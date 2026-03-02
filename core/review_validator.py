# core/review_validator.py

import json
from core.config import GAMMA_RETRY_LIMIT, GAMMA_CONFIDENCE_THRESHOLD

async def validate_gamma_json(call_func, prompt):
    """
    call_func(model, prompt) -> async generator
    """

    for _ in range(GAMMA_RETRY_LIMIT):
        result = ""
        async for chunk in call_func(prompt):
            result += chunk

        try:
            data = json.loads(result)
            if (
                "verdict" in data
                and "issues" in data
                and "minimal_fix" in data
                and "confidence" in data
                and data["confidence"] >= GAMMA_CONFIDENCE_THRESHOLD
            ):
                return data
        except:
            continue

    return {
        "verdict": "Gamma JSON 验证失败",
        "issues": ["输出非法或置信度不足"],
        "minimal_fix": "人工检查",
        "confidence": 0.0
    }