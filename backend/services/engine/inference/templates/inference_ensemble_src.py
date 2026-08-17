#!/usr/bin/env python3
"""
QuantMind 融合模型推理脚本 (inference_ensemble_src.py)
=======================================================
用户从模型管理页多选已有模型创建融合模型后，该脚本负责对融合模型执行推理：
读取模型目录中的 ensemble_config.json，逐一加载源模型 → 预测 → 截面百分位
归一化 → 加权融合 → 共识度统计 → 输出标准信号 JSON。

支持的源模型类型：
  - LightGBM (.lgb / .txt)
  - XGBoost (.xgb)
  - CatBoost (.cbm)
  - sklearn (.pkl)
  - Stacking 融合 (is_ensemble + ensemble_method=stacking，加载基模型 + meta_model)

平台注入环境变量：
    MODEL_DIR      融合模型目录绝对路径（含 ensemble_config.json + metadata.json）
    TRADE_DATE     推理日期（同 --date 参数，互为备份）
    OUTPUT_FORMAT  固定值 json
    MODEL_TRAINING_DATA_DIR  特征 parquet 数据目录

调用方式（由 InferenceScriptRunner 自动调用）：
    python inference.py --date YYYY-MM-DD --output /path/to/out.json

输出格式（写入 --output 文件）：
    [{"symbol": "SH600519", "score": 0.15, "consensus": 3, "horizons": 3, "detail": {...}}, ...]

exit code：
    0  = 成功
    1  = 致命错误（模型/配置损坏）
    2  = 该日期无可用数据（触发 alpha158 兜底）
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from catboost import CatBoost, Pool
except ImportError:
    CatBoost = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("inference_ensemble_src")

_PROJECT_ROOT = Path(os.getenv("QUANTMIND_PROJECT_ROOT", Path.cwd()))
_DEFAULT_DATA_DIR = str(_PROJECT_ROOT / "db" / "feature_snapshots")


# ═══════════════════════════════════════════════════════════════════════════
# 1. 配置加载
# ═══════════════════════════════════════════════════════════════════════════

def load_ensemble_config(model_dir: Path) -> dict:
    """从融合模型目录读取 ensemble_config.json。"""
    config_path = model_dir / "ensemble_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"ensemble_config.json 不存在: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict) or not config.get("models"):
        raise ValueError(f"ensemble_config.json 格式无效或 models 为空: {config_path}")
    return config


# ═══════════════════════════════════════════════════════════════════════════
# 2. 数据加载
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_parquet_path(data_dir: Path, trade_date: str, market: str = "CN") -> Path | None:
    """解析年度/市场 parquet 文件路径。

    market 非 CN 时优先取各市场汇总文件（model_features_{market}.parquet），
    CN 回退到按年拆分文件。
    """
    _MARKET_PARQUET = {
        "HK": "model_features_hk.parquet",
        "US": "model_features_us.parquet",
        "CRYPTO": "model_features_crypto.parquet",
        "FUTURES": "model_features_futures.parquet",
    }
    market_upper = str(market or "CN").upper()
    if market_upper in _MARKET_PARQUET:
        p = data_dir / _MARKET_PARQUET[market_upper]
        if p.exists():
            return p

    year = int(trade_date[:4])
    p = data_dir / f"model_features_{year}.parquet"
    if p.exists():
        return p
    p = data_dir / "model_features.parquet"
    return p if p.exists() else None


def load_day_data(trade_date: str, data_dir: Path, market: str = "CN") -> pd.DataFrame | None:
    """加载指定交易日的全市场特征数据。"""
    parquet_path = _resolve_parquet_path(data_dir, trade_date, market=market)
    if parquet_path is None:
        logger.warning("找不到 parquet 文件 (data_dir=%s)", data_dir)
        return None

    df = pd.read_parquet(parquet_path, engine="pyarrow")
    if "symbol" not in df.columns and "instrument" in df.columns:
        df = df.rename(columns={"instrument": "symbol"})
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
    day_df = df[df["trade_date"] == trade_date].copy()

    if len(day_df) == 0:
        logger.warning("日期 %s 无数据", trade_date)
        return None

    # 过滤不可交易：价格/成交量为零或负
    if "close" in day_df.columns:
        day_df = day_df[pd.to_numeric(day_df["close"], errors="coerce") > 0]
    if "volume" in day_df.columns:
        day_df = day_df[pd.to_numeric(day_df["volume"], errors="coerce") > 0]

    if len(day_df) == 0:
        logger.warning("日期 %s 过滤后无可交易数据", trade_date)
        return None

    return day_df


# ═══════════════════════════════════════════════════════════════════════════
# 3. 源模型加载（支持单模型 + stacking 融合）
# ═══════════════════════════════════════════════════════════════════════════

def _load_base_model(model_path: Path, model_type: str):
    """按类型加载单个基模型权重。"""
    suffix = model_path.suffix.lower()
    if model_type == "lightgbm" or suffix in (".lgb", ".txt"):
        if lgb is None:
            raise ImportError("lightgbm 未安装")
        return lgb.Booster(model_file=str(model_path))
    if model_type == "xgboost" or suffix == ".xgb":
        if xgb is None:
            raise ImportError("xgboost 未安装")
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        return booster
    if model_type == "catboost" or suffix == ".cbm":
        if CatBoost is None:
            raise ImportError("catboost 未安装")
        cb = CatBoost()
        cb.load_model(str(model_path), format="cbm")
        return cb
    # .pkl → sklearn / pickle 模型
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_source_model(model_dir: Path) -> tuple[object, dict]:
    """加载源模型。返回 (model, meta)。

    对 stacking 融合源模型，加载全部基模型 + meta_model，包装为 StackingEnsemble。
    """
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json 不存在: {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    is_stacking = bool(meta.get("is_ensemble")) and str(meta.get("ensemble_method", "")).lower() == "stacking"
    if is_stacking:
        model = _load_stacking(model_dir, meta)
        return model, meta

    model_file = meta.get("model_file", "")
    model_path = model_dir / model_file if model_file else None
    if not model_path or not model_path.exists():
        for ext in ("*.xgb", "*.lgb", "*.cbm", "*.pkl", "*.txt"):
            candidates = list(model_dir.glob(ext))
            if candidates:
                model_path = candidates[0]
                break
    if not model_path or not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {model_dir}")

    model_type = str(meta.get("model_type", "")).lower()
    return _load_base_model(model_path, model_type), meta


class _StackingEnsemble:
    """Stacking 源模型推理包装：基模型预测 → 元学习器融合。"""

    def __init__(self, base_models: dict, meta_model, model_types: list[str],
                 fill_values: dict, features: list[str]):
        self.base_models = base_models
        self.meta_model = meta_model
        self.model_types = model_types
        self.fill_values = fill_values
        self.features = features

    def predict(self, X: np.ndarray) -> np.ndarray:
        base_preds = []
        for mt in self.model_types:
            model = self.base_models.get(mt)
            if model is None:
                continue
            # 基模型训练时使用与融合模型相同的特征矩阵（stacking 用统一特征），
            # 这里直接对已填充的 X 预测
            if mt == "lightgbm":
                pred = model.predict(X)
            elif mt == "xgboost":
                # XGBoost Booster 需要特征名匹配训练时的顺序
                names = model.feature_names
                dmat = xgb.DMatrix(X, feature_names=names) if names else xgb.DMatrix(X)
                pred = model.predict(dmat)
            elif mt == "catboost":
                pred = np.asarray(model.predict(Pool(X))).flatten()
            else:
                pred = model.predict(X).flatten()
            base_preds.append(pred)
        if not base_preds:
            raise ValueError("Stacking 基模型为空")
        meta_X = np.column_stack(base_preds)
        return self.meta_model.predict(meta_X)


def _load_stacking(model_dir: Path, meta: dict) -> _StackingEnsemble:
    """加载 stacking 融合源模型的全部基模型 + meta_model。"""
    model_types = meta.get("model_types", [])
    saved_models = meta.get("saved_models", {})
    base_fv = meta.get("base_model_fill_values", {})
    global_fv = meta.get("fill_values", {})
    features = meta.get("feature_columns") or meta.get("features", [])

    base_models = {}
    for mt in model_types:
        model_file = saved_models.get(mt, "")
        if not model_file:
            continue
        model_path = model_dir / model_file
        if not model_path.exists():
            logger.warning("Stacking 基模型缺失: %s", model_path)
            continue
        base_models[mt] = _load_base_model(model_path, mt)

    meta_model_path = model_dir / meta.get("meta_model_file", "meta_model.pkl")
    if not meta_model_path.exists():
        raise FileNotFoundError(f"Stacking meta_model 不存在: {meta_model_path}")
    with open(meta_model_path, "rb") as f:
        meta_data = pickle.load(f)
    meta_model = meta_data["model"] if isinstance(meta_data, dict) else meta_data

    fill_values = {}
    for mt in model_types:
        per_model_fv = base_fv.get(mt, {}) if isinstance(base_fv, dict) else {}
        fill_values[mt] = per_model_fv if per_model_fv else (global_fv if isinstance(global_fv, dict) else {})

    logger.info("加载 Stacking 融合源模型: %d 个基模型 + 元学习器", len(base_models))
    return _StackingEnsemble(
        base_models=base_models,
        meta_model=meta_model,
        model_types=model_types,
        fill_values=fill_values,
        features=features,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. 单源模型预测
# ═══════════════════════════════════════════════════════════════════════════

def predict_with_model(model, meta: dict, day_df: pd.DataFrame) -> dict[str, float]:
    """对单个源模型执行推理，返回 {symbol: raw_score}。

    meta 决定特征列与填充值；day_df 为当日全市场数据副本。
    """
    feature_cols = meta.get("feature_columns") or meta.get("features", [])
    fill_values = meta.get("fill_values", {})
    best_iter = meta.get("best_iteration")

    # features_daily.return_Nd 是未来 N 日收益，不能作为特征喂给模型（标签泄漏）
    _leaky = [c for c in ("return_1d", "return_3d", "return_5d",
                          "return_10d", "return_20d", "return_60d")
              if c in day_df.columns]
    if _leaky:
        day_df = day_df.drop(columns=_leaky, errors="ignore")

    # 缺失列补 0
    missing = [c for c in feature_cols if c not in day_df.columns]
    if missing:
        logger.warning("源模型缺 %d 个特征列，填 0: %s", len(missing), missing[:8])
        for c in missing:
            day_df[c] = 0.0

    X = day_df[feature_cols].copy()
    for col, val in fill_values.items():
        if col in X.columns:
            X[col] = X[col].fillna(val)
    X = X.fillna(0.0)
    X_values = X.values.astype(np.float32)
    symbols = day_df["symbol"].tolist()

    is_stacking = isinstance(model, _StackingEnsemble)
    if is_stacking:
        scores = model.predict(X_values)
    elif hasattr(model, "get_dump") or str(type(model).__name__) == "Booster":
        # LightGBM Booster 支持 num_iteration；XGBoost 3.x Booster 需要 DMatrix
        if str(type(model).__module__).startswith("xgboost"):
            dmat = xgb.DMatrix(X_values, feature_names=list(feature_cols)) if xgb else None
            scores = model.predict(dmat)
        else:
            scores = model.predict(X_values, num_iteration=best_iter)
    else:
        # sklearn / pickle 模型
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_values)[:, 1]
        else:
            scores = model.predict(X_values)

    scores = np.asarray(scores).flatten()

    # 方向纠正：训练时 IC<0 的模型分数已翻转
    if meta.get("score_direction") == "reversed":
        scores = -scores

    result = {}
    for sym, s in zip(symbols, scores):
        f = float(s)
        if f == f:  # 过滤 NaN
            result[sym] = f
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 5. 融合逻辑
# ═══════════════════════════════════════════════════════════════════════════

_Z_CLIP = 3.0  # Winsorize: 单模型 z-score 截尾到 ±3σ, 防单点劫持


def fuse_scores(
    all_scores: dict[str, dict[str, float]],
    weights: dict[str, float],
    fusion_strategy: str = "linear",
    strategy_config: dict | None = None,
    model_horizons: dict[str, int] | None = None,
) -> list[dict]:
    """多源模型分数融合（按 fusion_strategy 选择算法）。

    公共预处理：
    1. 每个源模型的原始分做 z-score 标准化
    2. Winsorize: 单模型 z-score 截尾到 ±_Z_CLIP, 防极端值劫持

    融合算法（fusion_strategy）：
      - linear            线性加权平均（默认，原逻辑）
      - majority_vote     方向一致性投票（仅方向一致的模型参与加权，乘一致率）
      - periodic_hierarchy 周期分层（长周期定方向，短周期定时，按 target_horizon 分界）
      - confidence_gate   共识度门控（共识不足按阈值降权/丢弃）

    Returns:
        [{"symbol": "...", "score": <fused_z>, "consensus": n, "zfusion": <fused_z>,
          "detail": {...}}, ...]
    """
    strategy_config = strategy_config or {}
    model_horizons = model_horizons or {}
    boundary = float(strategy_config.get("periodic_boundary", 10))
    gate_threshold = float(strategy_config.get("confidence_threshold", 0.6))

    all_symbols = set()
    for scores in all_scores.values():
        all_symbols.update(scores.keys())

    # 1. 每个源模型独立 z-score 标准化
    z_scores: dict[str, dict[str, float]] = {}
    for mid, scores in all_scores.items():
        if not scores:
            continue
        vals = list(scores.values())
        n = len(vals)
        if n < 2:
            continue
        mu = sum(vals) / n
        variance = sum((v - mu) ** 2 for v in vals) / n
        sigma = variance ** 0.5
        if sigma < 1e-12:
            z_scores[mid] = {sym: 0.0 for sym in scores}
        else:
            z_scores[mid] = {sym: (s - mu) / sigma for sym, s in scores.items()}

    if not z_scores:
        return []

    n_models = len(z_scores)
    results = []

    for sym in all_symbols:
        per_model: dict[str, dict] = {}
        z_clipped_map: dict[str, float] = {}
        w_map: dict[str, float] = {}

        for mid in sorted(z_scores):
            if sym in z_scores[mid]:
                raw = all_scores[mid][sym]
                z = z_scores[mid][sym]
                z_clipped = max(-_Z_CLIP, min(_Z_CLIP, z))
                w = weights.get(mid, 1.0 / n_models)
                per_model[mid] = {
                    "raw": round(raw, 6),
                    "z": round(z, 4),
                    "z_clipped": round(z_clipped, 4),
                    "horizon": model_horizons.get(mid),
                }
                z_clipped_map[mid] = z_clipped
                w_map[mid] = w

        if not z_clipped_map:
            continue

        # 公共: 共识度 = 独立看多模型数 (z > 0)
        consensus = sum(1 for z in z_clipped_map.values() if z > 0)

        # 按策略计算融合分数
        if fusion_strategy == "majority_vote":
            # 方向投票：看涨数 vs 看跌数
            n_bull = sum(1 for z in z_clipped_map.values() if z > 0)
            n_bear = sum(1 for z in z_clipped_map.values() if z < 0)
            majority_dir = 1 if n_bull >= n_bear else -1
            # 仅方向一致的模型参与加权
            aligned = {mid: z for mid, z in z_clipped_map.items() if (z > 0) == (majority_dir > 0) and z != 0}
            if not aligned:
                fused_z = 0.0
            else:
                tot_w = sum(w_map[mid] for mid in aligned)
                fused_z = sum(z * w_map[mid] for mid, z in aligned.items()) / tot_w if tot_w > 0 else 0.0
                # 乘一致率（方向一致的模型占比），一致率低则弱化
                fused_z *= (len(aligned) / n_models)
        elif fusion_strategy == "periodic_hierarchy":
            # 长周期(≥boundary)定方向，短周期(<boundary)定时
            long_z = [z for mid, z in z_clipped_map.items()
                      if (model_horizons.get(mid) or boundary) >= boundary]
            short_z = [z for mid, z in z_clipped_map.items()
                       if (model_horizons.get(mid) or boundary) < boundary]
            if not short_z:
                short_z = list(z_clipped_map.values())  # 无短周期则全用
            fused_z = sum(short_z) / len(short_z) if short_z else 0.0
            if long_z:
                long_dir = sum(1 for z in long_z if z > 0) - sum(1 for z in long_z if z < 0)
                if long_dir > 0:  # 长趋势向上
                    fused_z *= (1.5 if fused_z > 0 else 0.3)
                elif long_dir < 0:  # 长趋势向下
                    fused_z *= (1.5 if fused_z < 0 else 0.3)
                else:  # 长周期分裂，中立
                    fused_z *= 0.5
        elif fusion_strategy == "confidence_gate":
            # 共识度门控
            consensus_ratio = consensus / n_models
            tot_w = sum(w_map.values())
            fused_z = sum(z * w_map[mid] for mid, z in z_clipped_map.items()) / tot_w if tot_w > 0 else 0.0
            if consensus_ratio >= gate_threshold:
                pass  # 高共识，保留
            elif consensus_ratio >= 0.4:
                fused_z *= 0.5  # 分歧，降权
            else:
                fused_z = 0.0  # 剧烈分歧，丢弃
        else:
            # linear（默认）：加权平均
            tot_w = sum(w_map.values())
            fused_z = sum(z * w_map[mid] for mid, z in z_clipped_map.items()) / tot_w if tot_w > 0 else 0.0

        results.append({
            "symbol": sym,
            "score": round(float(fused_z), 6),
            "consensus": int(consensus),
            "zfusion": round(float(fused_z), 6),
            "detail": per_model,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 6. 主流程
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="融合模型推理")
    p.add_argument("--date", "-d", type=str, default=os.getenv("TRADE_DATE", ""))
    p.add_argument("--output", "-o", type=str, required=True)
    p.add_argument("--model-dir", type=str, default=os.getenv("MODEL_DIR", ""))
    p.add_argument("--data-dir", type=str, default=os.getenv("MODEL_TRAINING_DATA_DIR", _DEFAULT_DATA_DIR))
    p.add_argument("--market", type=str, default=os.getenv("MARKET", "CN"),
                   choices=["CN", "US", "HK", "CRYPTO", "FUTURES"],
                   help="目标市场，用于选择对应 parquet 数据")
    return p.parse_args()


def main():
    args = parse_args()

    trade_date = (args.date or "").strip()
    if not trade_date:
        logger.error("未指定推理日期（--date 或 TRADE_DATE 环境变量）")
        sys.exit(1)

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    out_path = Path(args.output)
    market = (args.market or "CN").upper().strip()

    logger.info("=== 融合模型推理 ===")
    logger.info("  date     : %s", trade_date)
    logger.info("  market   : %s", market)
    logger.info("  model_dir: %s", model_dir)
    logger.info("  data_dir : %s", data_dir)

    # 1. 加载融合配置
    config = load_ensemble_config(model_dir)
    model_configs = config.get("models", [])
    logger.info("融合 %d 个源模型: %s", len(model_configs),
                [m.get("model_id", "?") for m in model_configs])

    # 2. 加载当日数据
    day_df = load_day_data(trade_date, data_dir, market=market)
    if day_df is None:
        msg = f"日期 {trade_date} 无数据"
        logger.warning(msg)
        print(msg, file=sys.stderr)
        sys.exit(2)

    # 融合算法与参数（ensemble_config.json）
    fusion_strategy = str(config.get("fusion_strategy") or "linear")
    strategy_config = config.get("strategy_config") or {}

    # 3. 逐源模型推理（缺失模型优雅降级）
    all_scores: dict[str, dict[str, float]] = {}
    weights: dict[str, float] = {}
    model_horizons: dict[str, int] = {}

    for mc in model_configs:
        mid = str(mc.get("model_id") or mc.get("name") or "?")
        m_dir = Path(mc.get("model_dir") or "")
        w = float(mc.get("weight", 0.1))
        h = mc.get("target_horizon_days")
        if isinstance(h, (int, float)):
            model_horizons[mid] = int(h)

        if not m_dir.exists():
            logger.warning("源模型目录缺失，跳过 %s: %s", mid, m_dir)
            continue

        try:
            model, meta = load_source_model(m_dir)
            scores = predict_with_model(model, meta, day_df.copy())
            if scores:
                all_scores[mid] = scores
                weights[mid] = w
                logger.info("源模型 %s: %d 条信号, weight=%.3f", mid, len(scores), w)
            else:
                logger.warning("源模型 %s 预测为空，跳过", mid)
        except Exception as e:
            logger.warning("源模型 %s 推理失败: %s", mid, e)

    if not all_scores:
        logger.error("所有源模型推理均失败")
        sys.exit(1)

    # 4. 融合
    signals = fuse_scores(
        all_scores,
        weights,
        fusion_strategy=fusion_strategy,
        strategy_config=strategy_config,
        model_horizons=model_horizons,
    )
    signals.sort(key=lambda x: x["score"], reverse=True)
    logger.info("融合完成: %d 条信号 (strategy=%s)", len(signals), fusion_strategy)

    # 5. 输出
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(signals, f, ensure_ascii=False)

    logger.info("已写入融合信号: %s (%d 条)", out_path, len(signals))


if __name__ == "__main__":
    main()
