# core/alpha_engine.py
from __future__ import annotations
from typing import Any, AsyncIterator, Dict, Optional
from core.protocol import ModelExecutionError
from core.config import ALPHA_MAX_TOKENS, ALPHA_TEMPERATURE, ALPHA_TOP_P, ALPHA_REPEAT_PENALTY


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
