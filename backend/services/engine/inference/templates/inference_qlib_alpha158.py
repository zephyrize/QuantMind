#!/usr/bin/env python3
"""QuantMind 原生 Qlib Alpha158 推理脚本（平台托管模板）。

训练与推理均使用 Qlib Alpha158 Handler；不要将该模型改按 feature snapshot
的列集加载，否则会改变训练口径。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("inference_qlib_alpha158")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qlib Alpha158 推理")
    parser.add_argument("--date", default=os.getenv("TRADE_DATE", ""))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model-dir", default=os.getenv("MODEL_DIR", str(Path(__file__).parent))
    )
    parser.add_argument("--provider-uri", default=os.getenv("QLIB_PROVIDER_URI", ""))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.date:
        logger.error("缺少 --date / TRADE_DATE")
        return 1
    try:
        from backend.services.engine.inference.qlib_alpha158 import (
            predict_alpha158_scores,
        )

        scores = predict_alpha158_scores(
            model_dir=args.model_dir,
            start_date=args.date,
            provider_uri=args.provider_uri or None,
        )
    except Exception as exc:
        logger.exception("Alpha158 推理失败: %s", exc)
        return 1
    if scores.empty:
        logger.error("%s 无可用 Alpha158 特征", args.date)
        return 2

    payload = [
        {"symbol": row.symbol, "score": float(row.score)}
        for row in scores[["symbol", "score"]].itertuples(index=False)
    ]
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Alpha158 推理完成: date=%s, signals=%d", args.date, len(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
