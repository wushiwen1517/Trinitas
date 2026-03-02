# core/beta_engine.py
from __future__ import annotations
from typing import Any, AsyncIterator, Dict, Optional
from core.protocol import ModelExecutionError
from core.config import BETA_MAX_TOKENS, BETA_TEMPERATURE, BETA_TOP_P, BETA_REPEAT_PENALTY


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
