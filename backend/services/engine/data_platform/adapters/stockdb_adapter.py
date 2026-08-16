"""StockDB HTTP adapter for the local A-share fallback source."""

from __future__ import annotations

import logging
import os
import re
from datetime import date
from typing import Any, Optional

import pandas as pd
import requests

from backend.services.engine.data_platform.base import (
    DataUnavailable,
    InvalidFieldRequest,
    OfflineDataSourceAdapter,
)
from backend.services.engine.data_platform.models import OHLCV_COLUMNS
from backend.shared.stock_utils import StockCodeUtil

logger = logging.getLogger(__name__)


class StockDBAdapter(OfflineDataSourceAdapter):
    """Read verified StockDB HTTP fields; QuantDB remains the primary source."""

    name = "stockdb"
    markets = ["A"]
    fields = {
        "daily_kline",
        "minute_kline",
        "stock_list",
        "calendar",
        "adj_factor",
        "valuation",
    }

    def __init__(self, session: requests.Session | None = None) -> None:
        host = os.getenv("STOCKDB_HOST", "localhost").strip() or "localhost"
        port = os.getenv("STOCKDB_PORT", "7899").strip() or "7899"
        try:
            timeout = max(1.0, float(os.getenv("STOCKDB_HTTP_TIMEOUT_SECONDS", "15")))
        except ValueError:
            timeout = 15.0
        self.base_url = f"http://{host}:{port}/"
        self.timeout_seconds = timeout
        self._session = session or requests.Session()

    def fetch_daily(
        self, symbol: str, start: date, end: date, *, adjust: str = "qfq"
    ) -> pd.DataFrame:
        code = _stockdb_code(symbol)
        rows = self._get_values("日k", code, f"fwd:{start:%Y%m%d},{end:%Y%m%d}")
        frame = _kline_frame(rows, symbol, minute=False)
        return _standardize_ohlcv(self._adjust(frame, code, adjust), symbol, self.name)

    def fetch_meta(self, market: str) -> pd.DataFrame:
        if market.upper() != "A":
            raise InvalidFieldRequest(f"StockDB 不支持 market={market}")
        keys = self._get_json(
            {"cmd": "keys", "t": "日k", "k1": "all:", "k2": f"key:{date.today():%Y%m%d}"}
        )
        codes = sorted({_key_code(value) for value in keys or [] if _key_code(value)})
        if not codes:
            raise DataUnavailable("StockDB 无当日股票列表")
        return pd.DataFrame(
            {
                "symbol": [StockCodeUtil.to_prefix(code) for code in codes],
                "code": codes,
                "exchange": [_exchange(code) for code in codes],
                "market": "A",
                "is_active": True,
                "source": self.name,
            }
        )

    def fetch_minute(
        self, symbol: str, start: date, end: date, *, freq: str = "1min"
    ) -> pd.DataFrame:
        if freq not in {"1min", "1m"}:
            raise InvalidFieldRequest("StockDB 只提供 1min 原始分钟K")
        code = _stockdb_code(symbol)
        selector = f"fwd:{start:%Y%m%d}000000,{end:%Y%m%d}235959"
        rows = self._get_values("分钟k", code, selector)
        return _standardize_ohlcv(_kline_frame(rows, symbol, minute=True), symbol, self.name)

    def fetch_field(
        self,
        field: str,
        symbol: str,
        *,
        start: Optional[date] = None,
        end: Optional[date] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if field == "adj_factor":
            code = _stockdb_code(symbol)
            factors = self._factor_events(code)
            if start and end and not factors.empty:
                rows = self._get_values("日k", code, f"fwd:{start:%Y%m%d},{end:%Y%m%d}")
                dates = _kline_frame(rows, symbol, minute=False)[["trade_date"]]
                factors = pd.merge_asof(
                    dates.sort_values("trade_date"), factors, on="trade_date", direction="backward"
                ).dropna(subset=["adj_factor"])
            else:
                if start:
                    factors = factors[factors["trade_date"] >= pd.Timestamp(start)]
                if end:
                    factors = factors[factors["trade_date"] <= pd.Timestamp(end)]
            if factors.empty:
                raise DataUnavailable(f"StockDB 无复权因子: {symbol}")
            factors = factors.copy()
            factors["symbol"] = StockCodeUtil.to_prefix(code)
            factors["source"] = self.name
            return factors[["symbol", "trade_date", "adj_factor", "source"]].reset_index(drop=True)
        if field == "valuation":
            if not start or not end:
                raise InvalidFieldRequest("StockDB valuation 需要 start 和 end")
            daily = self.fetch_daily(symbol, start, end, adjust="none")
            columns = [
                col for col in ("symbol", "trade_date", "pe_ttm", "pb", "total_mv", "float_mv", "turnover_rate")
                if col in daily.columns
            ]
            return daily[columns].copy()
        if field == "calendar":
            if not symbol:
                raise InvalidFieldRequest("StockDB calendar 需要一个标的代码")
            keys = self._get_json(
                {"cmd": "keys", "t": "日k", "k1": f"key:{_stockdb_code(symbol)}", "k2": "all:"}
            )
            days = sorted({_key_date(value) for value in keys or [] if _key_date(value)})
            if not days:
                raise DataUnavailable(f"StockDB 无交易日历: {symbol}")
            return pd.DataFrame({"trade_date": pd.to_datetime(days), "source": self.name})
        if field == "stock_list":
            return self.fetch_meta("A")
        raise InvalidFieldRequest(f"StockDB 不支持 field={field}")

    def _get_values(self, table: str, code: str, selector: str) -> list[dict[str, Any]]:
        payload = self._get_json({"cmd": "vals", "t": table, "k1": f"key:{code}", "k2": selector})
        rows = [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []
        if not rows:
            raise DataUnavailable(f"StockDB 无数据: {table}/{code}/{selector}")
        return rows

    def _get_json(self, params: dict[str, str]) -> Any:
        try:
            response = self._session.get(
                self.base_url, params=params, timeout=(3.05, self.timeout_seconds)
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise DataUnavailable(f"StockDB 请求失败 ({self.base_url}): {exc}") from exc

    def _factor_events(self, code: str) -> pd.DataFrame:
        payload = self._get_json({"cmd": "get", "t": "复权", "k1": f"key:{code}", "k2": "all:"})
        rows = []
        for item in payload if isinstance(payload, list) else []:
            if isinstance(item, list) and len(item) == 2 and isinstance(item[1], dict):
                event_date = _key_date(item[0])
                if event_date:
                    rows.append({"trade_date": event_date, "adj_factor": item[1].get("cum")})
        if not rows:
            return pd.DataFrame(columns=["trade_date", "adj_factor"])
        factors = pd.DataFrame(rows)
        factors["trade_date"] = pd.to_datetime(factors["trade_date"])
        factors["adj_factor"] = pd.to_numeric(factors["adj_factor"], errors="coerce")
        return factors.dropna(subset=["adj_factor"]).sort_values("trade_date").drop_duplicates("trade_date", keep="last")

    def _adjust(self, frame: pd.DataFrame, code: str, adjust: str) -> pd.DataFrame:
        mode = adjust.lower().strip()
        if mode in {"", "none", "raw"}:
            frame["adj_factor"] = 1.0
            return frame
        if mode not in {"qfq", "hfq"}:
            raise InvalidFieldRequest(f"StockDB 不支持复权方式: {adjust}")
        factors = self._factor_events(code)
        if factors.empty:
            frame["adj_factor"] = 1.0
            return frame
        result = pd.merge_asof(frame.sort_values("trade_date"), factors, on="trade_date", direction="backward")
        result["adj_factor"] = result["adj_factor"].fillna(1.0)
        if mode == "qfq":
            result["adj_factor"] /= float(factors["adj_factor"].iloc[-1])
        for column in ("open", "high", "low", "close", "pre_close"):
            if column in result.columns:
                result[column] *= result["adj_factor"]
        return result


def _stockdb_code(symbol: str) -> str:
    match = re.search(r"\d{6}", str(symbol or ""))
    if not match:
        raise InvalidFieldRequest(f"无效 A 股代码: {symbol}")
    return match.group(0)


def _kline_frame(rows: list[dict[str, Any]], symbol: str, *, minute: bool) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if "date" not in frame.columns:
        raise DataUnavailable("StockDB K线缺少 date 字段")
    fmt = "%Y%m%d%H%M%S" if minute else "%Y%m%d"
    frame["trade_date"] = pd.to_datetime(frame["date"].astype(str), format=fmt, errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).copy()
    if frame.empty:
        raise DataUnavailable("StockDB K线日期不可解析")
    for column in ("open", "high", "low", "close", "volume", "amount", "pre_close", "total_mv", "float_mv", "pe_ttm", "pb", "turnover", "pct_chg", "is_st"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "turnover" in frame.columns:
        frame["turnover_rate"] = frame["turnover"] / 100.0
    if "pct_chg" in frame.columns:
        frame["pctchange"] = frame["pct_chg"] / 100.0
    frame["symbol"] = StockCodeUtil.to_prefix(_stockdb_code(symbol))
    return frame.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True)


def _standardize_ohlcv(frame: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
    required = ("open", "high", "low", "close", "volume", "amount")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataUnavailable(f"StockDB K线缺少字段: {', '.join(missing)}")
    result = frame.copy()
    result["symbol"] = StockCodeUtil.to_prefix(_stockdb_code(symbol))
    result["source"] = source
    if "adj_factor" not in result.columns:
        result["adj_factor"] = 1.0
    ordered = [column for column in OHLCV_COLUMNS if column in result.columns]
    return result[ordered + [column for column in result.columns if column not in ordered]]


def _key_code(value: Any) -> str:
    parts = str(value).split(":")
    return parts[-2] if len(parts) >= 3 and re.fullmatch(r"\d{6}", parts[-2]) else ""


def _key_date(value: Any) -> str:
    parts = str(value).split(":")
    return parts[-1] if len(parts) >= 3 and re.fullmatch(r"\d{8}", parts[-1]) else ""


def _exchange(code: str) -> str:
    return {"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}.get(StockCodeUtil.to_prefix(code)[:2], "unknown")


def register() -> bool:
    """Register without probing StockDB during process startup."""
    from backend.services.engine.data_platform.registry import get_registry

    get_registry().register(StockDBAdapter, name=StockDBAdapter.name)
    logger.info("StockDBAdapter 已注册 (base_url=%s)", StockDBAdapter().base_url)
    return True
