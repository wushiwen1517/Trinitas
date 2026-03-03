# core/gamma_engine.py
"""
Trinitas Gamma 引擎：冲突裁决者。
接收 Alpha 的原始多假设文本与 Beta 的攻击报告（结构化拼接后输入），
删除被 Beta 击穿的假设，对幸存假设进行局部再推理，输出唯一的最终 Markdown 答案；
该输出作为系统向用户展示的最终回答，并保持流式输出以接入 Orchestrator 下游。
"""
from __future__ import annotations
import re as _re
from typing import Any, AsyncIterator, Dict, List, Optional
from core.protocol import ModelExecutionError
from core.config import GAMMA_MAX_TOKENS, GAMMA_TEMPERATURE, GAMMA_TOP_P, GAMMA_REPEAT_PENALTY

# ==========================================
# 冲突裁决模式：系统提示词与任务定义
# ==========================================
GAMMA_CONFLICT_RESOLVER_SYSTEM = """你是 Trinitas-Gamma，冲突裁决者。你接收 Alpha 的多个候选假设与 Beta 对每个假设的攻击报告，完成逻辑闭环并给出唯一最终答案。

你的任务：
1. 根据 Beta 的攻击内容，判断哪些假设被有效击穿（逻辑断裂、隐含假设冲突或反例成立），予以删除。
2. 对未被击穿的幸存假设进行局部再推理与综合，必要时修正或收敛。
3. 输出唯一的最终答案，以 Markdown 形式直接面向用户：结论清晰、结构分明（可用 ##/###、列表、加粗等），不要输出 JSON 或内部标签。

你必须用中文输出。
要求：最终答案必须是自洽、可执行的结论性内容，且为系统展示给用户的唯一结果；不要暴露 Alpha/Beta/假设编号等内部信息。"""


def build_adversarial_prompt(
    user_question: str,
    alpha_hypotheses_text: str,
    beta_attacks_text: str,
) -> str:
    """
    将 Alpha 的多假设文本与 Beta 的攻击报告拼接为 Gamma 的输入内容。
    Orchestrator 在 Pro/Auto 多假设链路中调用此函数生成 Gamma 的 prompt 正文。

    :param user_question: 用户原始问题
    :param alpha_hypotheses_text: Alpha 输出的多假设结构化文本（通常为 parse_hypotheses 后拼接或原始缓冲）
    :param beta_attacks_text: Beta 输出的攻击报告结构化文本（通常为 parse_attacks 后拼接或原始缓冲）
    :return: 拼接后的完整 prompt 字符串，供 Gamma 作为用户消息输入
    """
    return f"""[用户问题]
{user_question}

[Alpha 的多个候选假设]
{alpha_hypotheses_text.strip()}

[Beta 对各假设的攻击报告]
{beta_attacks_text.strip()}

请根据上述假设与攻击报告，删除被击穿的假设，对幸存假设进行局部再推理，并输出唯一的最终 Markdown 答案（直接面向用户，不要 JSON 或内部标签）。"""


class GammaEngine:
    DEFAULT_OPTIONS: Dict[str, Any] = {
        "num_predict": GAMMA_MAX_TOKENS,
        "temperature": GAMMA_TEMPERATURE,
        "top_p": GAMMA_TOP_P,
        "repeat_penalty": GAMMA_REPEAT_PENALTY,
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
            raise ModelExecutionError(f"GammaEngine.executor 缺少 stream()")
        return self.executor

    async def execute(self, prompt, *, options=None, keep_alive=None, system=None, **kw) -> AsyncIterator[str]:
        try:
            ex = self._ensure_executor()
            merged = {**self.DEFAULT_OPTIONS}; 
            if options: merged.update(options)
            async for chunk in ex.stream(self.model_name, prompt, options=merged, keep_alive=keep_alive, system=system):
                yield chunk
        except Exception as e:
            yield f"[Gamma执行异常] {e}"

    async def execute_buffered(self, prompt, *, options=None, keep_alive=None, system=None, **kw) -> str:
        try:
            ex = self._ensure_executor()
            merged = {**self.DEFAULT_OPTIONS}; 
            if options: merged.update(options)
            return await ex.generate_text(self.model_name, prompt, options=merged, keep_alive=keep_alive, system=system)
        except Exception as e:
            return f"[Gamma执行异常] {e}"

    async def generate_once(self, prompt, **kw) -> str:
        return await self.execute_buffered(prompt, **kw)

    async def run(self, prompt, **kw) -> AsyncIterator[str]:
        async for c in self.execute(prompt, **kw): yield c
