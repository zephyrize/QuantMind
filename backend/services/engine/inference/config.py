"""
Configuration for AI Inference Service
"""

import os
from pathlib import Path

from backend.shared.env_loader import PROJECT_ROOT

# Model Registry Paths
MODELS_DIR = PROJECT_ROOT / "models"
PRODUCTION_MODELS_DIR = MODELS_DIR / "production"
CANDIDATE_MODELS_DIR = MODELS_DIR / "candidates"
ARCHIVE_MODELS_DIR = MODELS_DIR / "archive"

# Service Configuration
INFERENCE_SERVICE_HOST = os.getenv("INFERENCE_SERVICE_HOST", "0.0.0.0")
INFERENCE_SERVICE_PORT = int(os.getenv("INFERENCE_SERVICE_PORT", "8007"))

# Model Loading Configuration
MAX_MODELS_IN_MEMORY = int(os.getenv("MAX_MODELS_IN_MEMORY", "5"))
MODEL_TIMEOUT_SECONDS = int(os.getenv("MODEL_TIMEOUT_SECONDS", "30"))

# History Buffer Configuration
HISTORY_WINDOW_SIZE = int(os.getenv("HISTORY_WINDOW_SIZE", "30"))

# Qlib Configuration — 通过 qlib_paths 统一解析，优先 QuantDB 缓存路径
from backend.shared.qlib_paths import resolve_qlib_provider_uri

QLIB_PROVIDER_URI = resolve_qlib_provider_uri()
QLIB_REGION = os.getenv("QLIB_REGION", "cn")
