# core/beta_engine.py
"""
Trinitas Beta 引擎：定向攻击器 (Adversarial Reviewer)。
接收 Alpha 的多假设文本，不对问题重新解题，仅针对每个假设进行挑刺，
输出纯文本格式的攻击块，并由正则解析为与假设一一映射的结构供 Gamma 使用。
"""
from __future__ import annotations
import re as _re
from typing import Any, AsyncIterator, Dict, List, Optional
from core.protocol import ModelExecutionError
from core.config import BETA_MAX_TOKENS, BETA_TEMPERATURE, BETA_TOP_P, BETA_REPEAT_PENALTY

# ==========================================
# 对抗性审查模式：系统提示词（纯文本攻击块，与 Alpha 假设一一对应）
# ==========================================
BETA_ADVERSARIAL_SYSTEM = """你是 Trinitas-Beta，对抗性审查者 (Adversarial Reviewer)。你的任务不是重新解题，而是针对已给出的多个候选假设逐条挑刺：找出逻辑断裂、隐含假设冲突或反例，并说明攻击要点。

你必须用中文输出。
严禁输出 JSON。必须且仅能按以下纯文本分段格式输出（每个假设对应一个攻击块）：

=== ATTACK ON HYPOTHESIS 1 ===
Target: [攻击点：逻辑断裂 / 隐含假设冲突 / 反例]
Detail: [详细攻击说明]

=== ATTACK ON HYPOTHESIS 2 ===
Target: [攻击点]
Detail: [详细说明]

（若有第三个假设，继续 === ATTACK ON HYPOTHESIS 3 === 同上格式。）

要求：
- 对每一个给出的假设至少输出一个攻击块，顺序与假设编号一致。
- Target 用一句话概括攻击类型或要点；Detail 展开说明。
- 不要输出 JSON、不要用 ``` 包裹；不要暴露内部标签。"""

# 宽容正则：匹配 "=== ATTACK ON HYPOTHESIS N ===" 标题行（允许等号数量波动、前后空白）
_ATTACK_HEADER_RE = _re.compile(
    r"={2,}\s*ATTACK\s+ON\s+HYPOTHESIS\s*\d+\s*={2,}",
    _re.IGNORECASE | _re.MULTILINE
)
# 从单块内提取 Target: / Detail:（允许中英文冒号、换行与空白）
_TARGET_DETAIL_RE = _re.compile(
    r"Target\s*[：:]\s*(.*?)(?=Detail\s*[：:]|$)",
    _re.DOTALL | _re.IGNORECASE
)
_DETAIL_RE = _re.compile(
    r"Detail\s*[：:]\s*(.*?)(?===|$)",
    _re.DOTALL | _re.IGNORECASE
)


def parse_attacks(raw: str) -> List[Dict[str, Any]]:
    """
    从 Beta 对抗性输出的纯文本中，按 === ATTACK ON HYPOTHESIS N === 切分为多个攻击块，
    并解析 Target / Detail，与 Alpha 的假设列表在结构上做好映射（hypothesis_index 对应假设编号）。

    :param raw: Beta 模型输出的完整原始文本
    :return: 攻击块列表，每项为 dict：hypothesis_index（1-based 假设编号）、target、detail、raw（该块全文）；
             若某块未解析出 target/detail 则对应键为空字符串；未匹配到任何块时返回空列表。
    """
    if not raw or not raw.strip():
        return []
    blocks: List[Dict[str, Any]] = []
    matches = list(_ATTACK_HEADER_RE.finditer(raw))
    if not matches:
        return []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        chunk = raw[start:end].strip()
        if not chunk:
            continue
        # 从标题中提取假设编号（1-based），便于与 alpha_hypotheses[hypothesis_index - 1] 映射
        header = m.group(0)
        num_m = _re.search(r"HYPOTHESIS\s*(\d+)", header, _re.IGNORECASE)
        hypothesis_index = int(num_m.group(1)) if num_m else (i + 1)
        target = ""
        detail = ""
        tm = _TARGET_DETAIL_RE.search(chunk)
        if tm:
            target = tm.group(1).strip()
        dm = _DETAIL_RE.search(chunk)
        if dm:
            detail = dm.group(1).strip()
        blocks.append({
            "hypothesis_index": hypothesis_index,
            "target": target,
            "detail": detail,
            "raw": chunk,
        })
    return blocks


class BetaEngine:
    DEFAULT_OPTIONS: Dict[str, Any] = {
        "num_predict": BETA_MAX_TOKENS,
        "temperature": BETA_TEMPERATURE,
        "top_p": BETA_TOP_P,
        "repeat_penalty": BETA_REPEAT_PENALTY,
        "repeat_last_n": 128,
        "num_ctx": 6144,
    }

    def __init__(self, executor_or_model, model_or_executor):
        if hasattr(executor_or_model, "stream") and isinstance(model_or_executor, str):
            self.executor = executor_or_model; self.model_name = model_or_executor
        elif hasattr(model_or_executor, "stream") and isinstance(executor_or_model, str):
            self.executor = model_or_executor; self.model_name = executor_or_model
        else:
            self.executor = executor_or_model; self.model_name = str(model_or_executor)

    def _ensure_executor(self):
        if not hasattr(self.executor, "stream"):
            raise ModelExecutionError(f"BetaEngine.executor 缺少 stream()")
        return self.executor

    async def execute(self, prompt, *, options=None, keep_alive=None, system=None, **kw) -> AsyncIterator[str]:
        try:
            ex = self._ensure_executor()
            merged = {**self.DEFAULT_OPTIONS}; 
            if options: merged.update(options)
            async for chunk in ex.stream(self.model_name, prompt, options=merged, keep_alive=keep_alive, system=system):
                yield chunk
        except Exception as e:
            yield f"[Beta执行异常] {e}"

    async def execute_buffered(self, prompt, *, options=None, keep_alive=None, system=None, **kw) -> str:
        try:
            ex = self._ensure_executor()
            merged = {**self.DEFAULT_OPTIONS}; 
            if options: merged.update(options)
            return await ex.generate_text(self.model_name, prompt, options=merged, keep_alive=keep_alive, system=system)
        except Exception as e:
            return f"[Beta执行异常] {e}"

    async def run(self, prompt, **kw) -> AsyncIterator[str]:
        async for c in self.execute(prompt, **kw): yield c

    async def generate_once(self, prompt, **kw) -> str:
        return await self.execute_buffered(prompt, **kw)
