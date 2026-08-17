"""Runtime helpers for native Qlib Alpha158 LightGBM models.

The training pipeline stores a native LightGBM booster plus Alpha158 metadata.
This module deliberately rebuilds the same Qlib handler at inference time instead
of attempting to reinterpret Alpha158 as a feature-snapshot model.
"""
from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from backend.shared.qlib_paths import resolve_qlib_provider_uri

logger = logging.getLogger(__name__)

# Qlib imports MLflow in every multiprocessing worker.  MLflow's current
# Pydantic-v2 compatibility warning is unrelated to inference and otherwise
# gets repeated once per Alpha158 feature worker.
warnings.filterwarnings(
    "ignore",
    message=r'^Field "model_name" has conflict with protected namespace "model_"\.',
    category=UserWarning,
    module=r"pydantic\._internal\._fields",
)


def read_metadata(model_dir: Path | str) -> dict[str, Any]:
    """Read model metadata, returning an empty mapping for invalid files."""
    path = Path(model_dir) / "metadata.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def is_qlib_alpha158_model(metadata: dict[str, Any]) -> bool:
    """Whether metadata describes the native Alpha158 LightGBM artifact."""
    return (
        str(metadata.get("data_source") or "").lower() in {"qlib", "qlib_bin", "bin"}
        and str(metadata.get("feature_mode") or "").lower() == "qlib_alpha158"
        and str(metadata.get("model_type") or "").lower() == "lightgbm"
    )


def resolve_alpha158_provider_uri(
    metadata: dict[str, Any], fallback: Path | str | None = None
) -> str:
    """Resolve a portable provider URI, preferring a valid model-specific path."""
    for key in ("qlib_provider_uri", "qlib_data_path"):
        raw = str(metadata.get(key) or "").strip()
        if raw and Path(raw).is_dir():
            return str(Path(raw).resolve())
    if fallback and Path(fallback).is_dir():
        return str(Path(fallback).resolve())
    return resolve_qlib_provider_uri()


def get_qlib_trading_dates(
    provider_uri: Path | str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[str]:
    """Return trading dates from the Qlib calendar without loading features."""
    calendar_path = Path(provider_uri) / "calendars" / "day.txt"
    if not calendar_path.is_file():
        return []
    try:
        dates = [
            line.strip()
            for line in calendar_path.read_text(encoding="utf-8").splitlines()
        ]
    except OSError:
        return []
    dates = [value for value in dates if value]
    if start_date:
        dates = [value for value in dates if value >= start_date]
    if end_date:
        dates = [value for value in dates if value <= end_date]
    return dates


def predict_alpha158_scores(
    model_dir: Path | str,
    start_date: str,
    end_date: str | None = None,
    provider_uri: Path | str | None = None,
) -> pd.DataFrame:
    """Generate native Alpha158 scores indexed by ``trade_date`` and ``symbol``.

    The Alpha158 handler and its raw feature matrix match the handler used by
    ``docker/training/train.py``.  LightGBM natively handles the missing values
    that are also present during training, so no snapshot-style filling or
    column reshaping is applied here.
    """
    model_dir = Path(model_dir)
    metadata = read_metadata(model_dir)
    if not is_qlib_alpha158_model(metadata):
        raise ValueError("模型不是原生 Qlib Alpha158 LightGBM 模型")

    end_date = end_date or start_date
    effective_uri = resolve_alpha158_provider_uri(metadata, provider_uri)
    model_file = str(metadata.get("model_file") or "model.lgb")
    model_path = model_dir / model_file
    if not model_path.is_file():
        raise FileNotFoundError(f"Alpha158 模型文件不存在: {model_path}")

    try:
        import lightgbm as lgb
        import qlib
        from qlib.contrib.data.handler import Alpha158
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("Alpha158 推理依赖不可用：需要 qlib 与 lightgbm") from exc

    # Alpha158 contains many expressions.  A small, bounded worker count avoids
    # Windows spawning one Python process per available CPU for a single
    # inference request; deployments may opt into more parallelism explicitly.
    default_kernels = min(os.cpu_count() or 1, 8)
    kernels = max(
        1, int(os.getenv("QLIB_INFERENCE_KERNELS", str(default_kernels)))
    )
    qlib.init(provider_uri=effective_uri, region="cn", kernels=kernels)
    train_start = str(metadata.get("train_start") or start_date)
    train_end = str(metadata.get("train_end") or end_date)
    label_expression = str(
        metadata.get("label_expression") or "Ref($close, -2)/Ref($close, -1) - 1"
    )
    handler = Alpha158(
        instruments=str(metadata.get("qlib_universe") or "all"),
        start_time=start_date,
        end_time=end_date,
        fit_start_time=train_start,
        fit_end_time=train_end,
        label=([label_expression], ["LABEL0"]),
    )
    dataset = DatasetH(
        handler=handler,
        segments={"infer": (start_date, end_date)},
    )
    features = dataset.prepare(
        "infer", col_set="feature", data_key=DataHandlerLP.DK_I
    )
    if features is None or features.empty:
        return pd.DataFrame(columns=["trade_date", "symbol", "score"])

    booster = lgb.Booster(model_file=str(model_path))
    scores = booster.predict(features.values)
    result = pd.DataFrame({"score": scores}, index=features.index).reset_index()
    result = result.rename(columns={"datetime": "trade_date", "instrument": "symbol"})
    if "trade_date" not in result or "symbol" not in result:
        raise RuntimeError("Alpha158 特征索引缺少 datetime 或 instrument")
    result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.strftime("%Y-%m-%d")
    result["symbol"] = result["symbol"].astype(str)
    result["score"] = pd.to_numeric(result["score"], errors="coerce")
    return result.dropna(subset=["score"]).reset_index(drop=True)
