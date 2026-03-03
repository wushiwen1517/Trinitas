# 语言: python
# 文件路径: core/oculus_engine.py

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

from core.protocol import ModelExecutionError


class OculusEngine:
    """
    Oculus（视觉模块）
    负责看图并输出 VisionPack（协议层会尝试截取 [VisionPack]）
    """

    def __init__(self, executor_or_model: Any, model_or_executor: Any) -> None:
        if hasattr(executor_or_model, "vision_extract") and isinstance(model_or_executor, str):
            self.executor = executor_or_model
            self.model_name = model_or_executor
        elif hasattr(model_or_executor, "vision_extract") and isinstance(executor_or_model, str):
            self.executor = model_or_executor
            self.model_name = executor_or_model
        else:
            self.executor = executor_or_model
            self.model_name = str(model_or_executor)

    def _ensure_executor(self) -> Any:
        if not hasattr(self.executor, "vision_extract"):
            raise ModelExecutionError(
                f"OculusEngine.executor 类型错误：{type(self.executor).__name__}，缺少 vision_extract() 方法"
            )
        return self.executor

    @staticmethod
    def to_base64(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    async def extract_from_base64(
        self,
        image_b64: str,
        *,
        prompt: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
        strict_vision_pack: bool = True,
    ) -> str:
        executor = self._ensure_executor()

        vision_prompt = (
            prompt
            or "请识别图片并结构化输出。请严格把提取文字、错误码、路径、UI元素包裹在[VisionPack]和[/VisionPack]之间。"
        )

        try:
            return await executor.vision_extract(
                self.model_name,
                vision_prompt,
                images=[image_b64],
                options=options,
                keep_alive=keep_alive,
                strict_vision_pack=strict_vision_pack,
            )
        except Exception as e:
            return f"[Oculus执行异常] {e}"

    async def extract_from_bytes(
        self,
        image_bytes: bytes,
        *,
        prompt: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
        strict_vision_pack: bool = True,
    ) -> str:
        return await self.extract_from_base64(
            self.to_base64(image_bytes),
            prompt=prompt,
            options=options,
            keep_alive=keep_alive,
            strict_vision_pack=strict_vision_pack,
        )

    # 修复点：新增 execute 方法，适配 orchestrator 的统一调用入口。
    # 自动过滤掉底层不支持的参数（如 stage_manager），防止抛出 TypeError。
    async def execute(self, image_bytes: bytes, **kwargs: Any) -> str:
        allowed_keys = {"prompt", "options", "keep_alive", "strict_vision_pack"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        return await self.extract_from_bytes(image_bytes, **filtered_kwargs)

    # 兼容调用名
    async def run_vision(self, image_bytes: bytes, **kwargs: Any) -> str:
        allowed_keys = {"prompt", "options", "keep_alive", "strict_vision_pack"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        return await self.extract_from_bytes(image_bytes, **filtered_kwargs)

    async def analyze_image(self, image_bytes: bytes, **kwargs: Any) -> str:
        allowed_keys = {"prompt", "options", "keep_alive", "strict_vision_pack"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
        return await self.extract_from_bytes(image_bytes, **filtered_kwargs)