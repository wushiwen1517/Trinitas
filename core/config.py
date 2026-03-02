# 语言: python
# 文件路径: core/config.py

"""
Trinitas 全局配置中心（工业兼容版）
"""

import os

os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
os.environ["no_proxy"] = "127.0.0.1,localhost,::1"

# =========================
# 安全配置
# =========================
API_KEY = "wushiwen5170"

# =========================
# Ollama 服务配置
# =========================
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# =========================
# 数据库配置
# =========================
DB_NAME = "trinitas.db"

# =========================
# 模型名称
# =========================
MODEL_ALPHA = "Trinitas-Alpha:latest"
MODEL_BETA = "Trinitas-Beta:latest"
MODEL_GAMMA = "Trinitas-Gamma:latest"
MODEL_VISION = "minicpm-v:latest"
MODEL_OCULUS = MODEL_VISION

# =========================
# 模型推理参数
# =========================
ALPHA_MAX_TOKENS = 4096
ALPHA_TEMPERATURE = 0.7
ALPHA_TOP_P = 0.95
ALPHA_REPEAT_PENALTY = 1.1

BETA_MAX_TOKENS = 2048
BETA_TEMPERATURE = 0.3
BETA_TOP_P = 0.9
BETA_REPEAT_PENALTY = 1.05

GAMMA_MAX_TOKENS = 2048
GAMMA_TEMPERATURE = 0.2
GAMMA_TOP_P = 0.85
GAMMA_REPEAT_PENALTY = 1.05

VISION_MAX_TOKENS = 2048
VISION_TEMPERATURE = 0.2
VISION_TOP_P = 0.9
VISION_REPEAT_PENALTY = 1.05

# =========================
# Orchestrator / 审查控制参数
# =========================
CONFIDENCE_THRESHOLD = 0.6
MAX_RETRY_COUNT = 1

# 兼容旧模块
GAMMA_RETRY_LIMIT = 3
GAMMA_CONFIDENCE_THRESHOLD = 0.65

# =========================
# Memory / 上下文控制
# =========================
SHORT_TERM_LIMIT = 12
LONG_TERM_TRIGGER = 15000

# =========================
# 功能开关
# =========================
ENABLE_STREAMING = True
ENABLE_RETRY = True
ENABLE_BETA_REVIEW = True
ENABLE_GAMMA_JUDGE = True
ENABLE_VISION = True

# =========================
# 稳定性增强参数
# =========================
ALPHA_STAGE_TIMEOUT_SECONDS = 1800
BETA_STAGE_TIMEOUT_SECONDS = 900
GAMMA_STAGE_TIMEOUT_SECONDS = 900
VISION_STAGE_TIMEOUT_SECONDS = 600

ALPHA_STAGE_MAX_RETRY = 1
BETA_STAGE_MAX_RETRY = 1
GAMMA_STAGE_MAX_RETRY = 1
VISION_STAGE_MAX_RETRY = 1

STAGE_RETRY_BACKOFF_SECONDS = 1.5
ENABLE_L0_QUALITY_GUARD = True

# =========================
# 上下文限制（前端显示用）
# =========================
MAX_CONTEXT_TOKENS = 15000
