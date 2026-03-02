# -*- coding: utf-8 -*-
"""
Trinitas 工业异常体系（完整版）
--------------------------------
用途：
1) 给 orchestrator / engine / router / protocol / validator 提供统一异常类型
2) 支持流式阶段标注（stage）
3) 支持 retryable / meta / cause 透传
4) 提供兼容别名，减少模块重构过程中的 import 崩溃

注意：
- 这是“完整替换版”，请直接覆盖 core/error_types.py
- 如果后续某个模块 import 了新的异常名，只需要在本文件补一个兼容别名即可
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ============================================================
# 基础异常模型
# ============================================================

@dataclass
class ErrorPayload:
    code: str
    message: str
    stage: Optional[str] = None
    retryable: bool = False
    http_status: int = 500
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "retryable": self.retryable,
            "http_status": self.http_status,
            "meta": self.meta or {},
        }


class TrinitasError(Exception):
    """
    Trinitas 所有业务异常的基类。
    """

    default_code = "TRINITAS_ERROR"
    default_http_status = 500
    default_retryable = False

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        stage: Optional[str] = None,
        retryable: Optional[bool] = None,
        http_status: Optional[int] = None,
        cause: Optional[BaseException] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.stage = stage
        self.retryable = self.default_retryable if retryable is None else retryable
        self.http_status = self.default_http_status if http_status is None else http_status
        self.cause = cause
        self.meta: Dict[str, Any] = meta or {}

    def to_payload(self) -> ErrorPayload:
        payload_meta = dict(self.meta)
        if self.cause is not None and "cause" not in payload_meta:
            payload_meta["cause"] = repr(self.cause)
        return ErrorPayload(
            code=self.code,
            message=self.message,
            stage=self.stage,
            retryable=self.retryable,
            http_status=self.http_status,
            meta=payload_meta,
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.to_payload().to_dict()

    def __str__(self) -> str:
        return self.message


# ============================================================
# 请求 / 鉴权 / JSON 入参异常
# ============================================================

class RequestValidationError(TrinitasError):
    default_code = "REQUEST_VALIDATION_ERROR"
    default_http_status = 400
    default_retryable = False


class JsonParseError(RequestValidationError):
    """
    你当前报错缺失的类：orchestrator 会 import 这个名字。
    """
    default_code = "JSON_PARSE_ERROR"


class MissingFieldError(RequestValidationError):
    default_code = "MISSING_FIELD_ERROR"

    def __init__(self, field_name: str, **kwargs: Any) -> None:
        super().__init__(f"缺少字段: {field_name}", meta={"field": field_name}, **kwargs)


class InvalidFieldError(RequestValidationError):
    default_code = "INVALID_FIELD_ERROR"

    def __init__(self, field_name: str, reason: str = "字段无效", **kwargs: Any) -> None:
        super().__init__(
            f"字段无效: {field_name} ({reason})",
            meta={"field": field_name, "reason": reason},
            **kwargs,
        )


class AuthenticationError(TrinitasError):
    default_code = "AUTHENTICATION_ERROR"
    default_http_status = 401
    default_retryable = False


class ApiKeyError(AuthenticationError):
    default_code = "API_KEY_ERROR"


class UnauthorizedError(AuthenticationError):
    default_code = "UNAUTHORIZED"


# ============================================================
# 路由 / 风险 / 编排异常
# ============================================================

class RouterDecisionError(TrinitasError):
    default_code = "ROUTER_DECISION_ERROR"
    default_http_status = 500
    default_retryable = False


class OrchestrationError(TrinitasError):
    default_code = "ORCHESTRATION_ERROR"
    default_http_status = 500
    default_retryable = False


class StageExecutionError(OrchestrationError):
    default_code = "STAGE_EXECUTION_ERROR"

    def __init__(self, stage_name: str, message: str, **kwargs: Any) -> None:
        super().__init__(
            f"[{stage_name}] {message}",
            stage=stage_name,
            meta={"stage_name": stage_name, **(kwargs.pop('meta', {}) or {})},
            **kwargs,
        )


# ============================================================
# 上下文 / 记忆 / 存储异常
# ============================================================

class ContextTrimmerError(TrinitasError):
    default_code = "CONTEXT_TRIMMER_ERROR"
    default_http_status = 500
    default_retryable = True


class MemoryErrorBase(TrinitasError):
    default_code = "MEMORY_ERROR"
    default_http_status = 500
    default_retryable = True


class DatabaseError(TrinitasError):
    default_code = "DATABASE_ERROR"
    default_http_status = 500
    default_retryable = True


class ChatNotFoundError(TrinitasError):
    default_code = "CHAT_NOT_FOUND"
    default_http_status = 404
    default_retryable = False


# ============================================================
# 模型 / Ollama / 上游 HTTP 异常
# ============================================================

class ModelExecutionError(TrinitasError):
    default_code = "MODEL_EXECUTION_ERROR"
    default_http_status = 502
    default_retryable = True


class OllamaError(ModelExecutionError):
    default_code = "OLLAMA_ERROR"


class UpstreamConnectionError(ModelExecutionError):
    default_code = "UPSTREAM_CONNECTION_ERROR"
    default_retryable = True

    def __init__(self, message: str = "无法连接上游模型服务", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)


class UpstreamTimeoutError(ModelExecutionError):
    default_code = "UPSTREAM_TIMEOUT_ERROR"
    default_retryable = True

    def __init__(self, message: str = "上游模型服务超时", **kwargs: Any) -> None:
        super().__init__(message, **kwargs)


class UpstreamHTTPStatusError(ModelExecutionError):
    default_code = "UPSTREAM_HTTP_STATUS_ERROR"
    default_retryable = False

    def __init__(
        self,
        status_code: int,
        message: Optional[str] = None,
        response_text: str = "",
        **kwargs: Any,
    ) -> None:
        msg = message or f"上游服务返回异常状态码: HTTP {status_code}"
        meta = kwargs.pop("meta", {}) or {}
        meta.update({"status_code": status_code, "response_text": response_text})
        super().__init__(msg, meta=meta, **kwargs)
        self.status_code = status_code
        self.response_text = response_text


class ModelNotFoundError(ModelExecutionError):
    default_code = "MODEL_NOT_FOUND"
    default_retryable = False

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        super().__init__(f"模型不存在或未安装: {model_name}", meta={"model": model_name}, **kwargs)


class ModelResponseFormatError(ModelExecutionError):
    default_code = "MODEL_RESPONSE_FORMAT_ERROR"
    default_retryable = True


# ============================================================
# 流式协议 / chunk 解析异常
# ============================================================

class StreamProtocolError(ModelExecutionError):
    default_code = "STREAM_PROTOCOL_ERROR"
    default_retryable = True


class StreamClosedError(ModelExecutionError):
    default_code = "STREAM_CLOSED_ERROR"
    default_retryable = True


class ChunkDecodeError(StreamProtocolError):
    default_code = "CHUNK_DECODE_ERROR"


class ChunkJsonParseError(StreamProtocolError):
    default_code = "CHUNK_JSON_PARSE_ERROR"


# ============================================================
# 审稿 / 结构化输出 / 置信度解析异常
# ============================================================

class ReviewValidationError(TrinitasError):
    default_code = "REVIEW_VALIDATION_ERROR"
    default_http_status = 500
    default_retryable = True


class ReviewFormatError(ReviewValidationError):
    default_code = "REVIEW_FORMAT_ERROR"


class MinimalFixMissingError(ReviewValidationError):
    default_code = "MINIMAL_FIX_MISSING"


class ConfidenceParseError(ReviewValidationError):
    default_code = "CONFIDENCE_PARSE_ERROR"


# ============================================================
# 视觉 / VisionPack 异常
# ============================================================

class VisionError(TrinitasError):
    default_code = "VISION_ERROR"
    default_http_status = 500
    default_retryable = True


class VisionPackParseError(VisionError):
    default_code = "VISIONPACK_PARSE_ERROR"


class VisionImageEncodeError(VisionError):
    default_code = "VISION_IMAGE_ENCODE_ERROR"
    default_retryable = False


# ============================================================
# 工具函数：统一包装未知异常
# ============================================================

def wrap_unknown_exception(
    exc: BaseException,
    *,
    stage: Optional[str] = None,
    code: str = "UNEXPECTED_ERROR",
    message_prefix: str = "未处理异常",
    retryable: bool = False,
    http_status: int = 500,
    meta: Optional[Dict[str, Any]] = None,
) -> TrinitasError:
    return TrinitasError(
        f"{message_prefix}: {exc}",
        code=code,
        stage=stage,
        retryable=retryable,
        http_status=http_status,
        cause=exc,
        meta=meta or {},
    )


# ============================================================
# 兼容别名（防止旧模块 import 名称不同）
# ============================================================
# 请求/JSON
InvalidJSONError = JsonParseError
BadRequestError = RequestValidationError

# 上游 / HTTP
UpstreamBadStatusError = UpstreamHTTPStatusError
OllamaConnectionError = UpstreamConnectionError
OllamaTimeoutError = UpstreamTimeoutError
OllamaBadStatusError = UpstreamHTTPStatusError

# 流
StreamParseError = ChunkJsonParseError
ProtocolError = StreamProtocolError

# 审稿 / 结构化
ValidatorError = ReviewValidationError
ReviewParseError = ReviewFormatError

# Vision
VisionParseError = VisionPackParseError


__all__ = [
    # base
    "ErrorPayload",
    "TrinitasError",
    "wrap_unknown_exception",

    # request/auth/json
    "RequestValidationError",
    "JsonParseError",
    "MissingFieldError",
    "InvalidFieldError",
    "AuthenticationError",
    "ApiKeyError",
    "UnauthorizedError",

    # orchestration/router
    "RouterDecisionError",
    "OrchestrationError",
    "StageExecutionError",

    # context/memory/db
    "ContextTrimmerError",
    "MemoryErrorBase",
    "DatabaseError",
    "ChatNotFoundError",

    # model/ollama/upstream
    "ModelExecutionError",
    "OllamaError",
    "UpstreamConnectionError",
    "UpstreamTimeoutError",
    "UpstreamHTTPStatusError",
    "ModelNotFoundError",
    "ModelResponseFormatError",

    # stream
    "StreamProtocolError",
    "StreamClosedError",
    "ChunkDecodeError",
    "ChunkJsonParseError",

    # review/validator/confidence
    "ReviewValidationError",
    "ReviewFormatError",
    "MinimalFixMissingError",
    "ConfidenceParseError",

    # vision
    "VisionError",
    "VisionPackParseError",
    "VisionImageEncodeError",

    # aliases
    "InvalidJSONError",
    "BadRequestError",
    "UpstreamBadStatusError",
    "OllamaConnectionError",
    "OllamaTimeoutError",
    "OllamaBadStatusError",
    "StreamParseError",
    "ProtocolError",
    "ValidatorError",
    "ReviewParseError",
    "VisionParseError",
]