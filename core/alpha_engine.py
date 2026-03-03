# core/alpha_engine.py
"""
Trinitas Alpha 引擎：多假设生成器。
在 Pro/Auto 模式下负责发散思维，输出 2～3 个带依赖前提与自身弱点的假设（纯文本分段格式），
并由正则解析为分块列表供后续 Beta/Gamma 使用。
"""
from __future__ import annotations
import re as _re
from typing import Any, AsyncIterator, Dict, List, Optional
from core.protocol import ModelExecutionError
from core.config import ALPHA_MAX_TOKENS, ALPHA_TEMPERATURE, ALPHA_TOP_P, ALPHA_REPEAT_PENALTY

# ==========================================
# 多假设生成模式：系统提示词（严禁 JSON，强制纯文本分段）
# ==========================================
ALPHA_MULTI_HYPOTHESIS_SYSTEM = """你是 Trinitas-Alpha，多假设生成器。你的任务不是直接给出最终答案，而是针对用户问题发散思考，给出 2～3 个候选假设，每个假设需包含结论、依赖前提与自我评估的弱点。

你必须用中文输出。
严禁输出 JSON。必须且仅能按以下纯文本分段格式输出（保留标题行与字段名）：

=== HYPOTHESIS 1 ===
Answer: [结论内容]
Assumptions:
- [依赖前提 1]
- [依赖前提 2]
Weakness:
- [自我评估的不确定点或易被攻击处]

=== HYPOTHESIS 2 ===
Answer: [结论内容]
Assumptions:
- [依赖前提]
Weakness:
- [不确定点]

（如有第三个假设，继续 === HYPOTHESIS 3 === 同上格式。）

要求：
- 至少输出 2 个假设，至多 3 个。
- 每个假设的 Answer 要具体可判；Assumptions 与 Weakness 用列表项列出。
- 不要输出任何 JSON、不要用 ``` 包裹上述结构；不要暴露内部标签（如 TaskPack、RISK 等）。"""

# 宽容正则：匹配 "=== HYPOTHESIS N ===" 标题行（允许等号数量波动、前后空白、换行、Markdown 加粗等）
_HYPOTHESIS_HEADER_RE = _re.compile(
    r"={2,}\s*HYPOTHESIS\s*\d+\s*={2,}",
    _re.IGNORECASE | _re.MULTILINE
)


def parse_hypotheses(raw: str) -> List[str]:
    """
    从 Alpha 多假设生成的纯文本中，按 === HYPOTHESIS N === 切分为多个假设块。
    正则宽容：允许跨行、等号数量波动、前后空白及 Markdown 符号。

    :param raw: Alpha 模型输出的完整原始文本
    :return: 各假设块的原文列表（每项为包含该假设标题及内容的整段文本），顺序与原文一致；
             若未匹配到任何块则返回空列表；解析失败时返回空列表以保证调用方可降级。
    """
    if not raw or not raw.strip():
        return []
    blocks: List[str] = []
    # 找到所有标题的起始位置（含匹配对象，便于截取区间）
    matches = list(_HYPOTHESIS_HEADER_RE.finditer(raw))
    if not matches:
        return []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        chunk = raw[start:end]
        chunk_stripped = chunk.strip()
        if chunk_stripped:
            blocks.append(chunk_stripped)
    return blocks


class AlphaEngine:
    DEFAULT_OPTIONS: Dict[str, Any] = {
        "num_predict": ALPHA_MAX_TOKENS,
        "temperature": ALPHA_TEMPERATURE,
        "top_p": ALPHA_TOP_P,
        "repeat_penalty": ALPHA_REPEAT_PENALTY,
        "repeat_last_n": 256,
        "num_ctx": 8192,
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
            raise ModelExecutionError(f"AlphaEngine.executor 缺少 stream()")
        return self.executor

    async def execute(self, prompt, *, options=None, keep_alive=None, system=None, **kw) -> AsyncIterator[str]:
        try:
            ex = self._ensure_executor()
            merged = {**self.DEFAULT_OPTIONS}; 
            if options: merged.update(options)
            async for chunk in ex.stream(self.model_name, prompt, options=merged, keep_alive=keep_alive, system=system):
                yield chunk
        except Exception as e:
            yield f"[Alpha执行异常] {e}"

    async def execute_buffered(self, prompt, *, options=None, keep_alive=None, system=None, **kw) -> str:
        """非流式执行，返回完整文本（用于审查模式下 Alpha 内部缓冲）"""
        try:
            ex = self._ensure_executor()
            merged = {**self.DEFAULT_OPTIONS}; 
            if options: merged.update(options)
            return await ex.generate_text(self.model_name, prompt, options=merged, keep_alive=keep_alive, system=system)
        except Exception as e:
            return f"[Alpha执行异常] {e}"

    async def run(self, prompt, **kw) -> AsyncIterator[str]:
        async for c in self.execute(prompt, **kw): yield c

    async def generate_once(self, prompt, *, options=None, keep_alive=None, system=None, **kw) -> str:
        return await self.execute_buffered(prompt, options=options, keep_alive=keep_alive, system=system)

    async def summarize(self, prompt, **kw): return await self.generate_once(prompt, **kw)
    async def summarize_text(self, prompt, **kw): return await self.generate_once(prompt, **kw)
    async def finalize_from_review(self, prompt, **kw): return await self.generate_once(prompt, **kw)
    async def build_final_answer(self, prompt, **kw): return await self.generate_once(prompt, **kw)
