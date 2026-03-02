# core/gamma_engine.py
from __future__ import annotations
import re
from typing import Any, AsyncIterator, Dict, Optional
from core.protocol import ModelExecutionError
from core.config import GAMMA_MAX_TOKENS, GAMMA_TEMPERATURE, GAMMA_TOP_P, GAMMA_REPEAT_PENALTY


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
