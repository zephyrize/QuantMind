from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from backend.services.engine.data_platform.adapters.stockdb_adapter import StockDBAdapter
from backend.services.engine.data_platform.base import DataUnavailable


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Session:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append(params)
        return _Response(self.payloads.pop(0))


def test_stockdb_daily_normalizes_descending_rows_and_qfq(monkeypatch):
    monkeypatch.setenv("STOCKDB_HOST", "stockdb.test")
    session = _Session([
        [
            {"date": 20260103, "open": 20, "high": 22, "low": 19, "close": 21, "volume": 200, "amount": 4200, "turnover": 2, "is_st": False},
            {"date": 20260102, "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100, "amount": 1000, "turnover": 1, "is_st": False},
        ],
        [["复权:600000:20260101", {"cum": 2.0}]],
    ])
    out = StockDBAdapter(session=session).fetch_daily(
        "SH600000", date(2026, 1, 2), date(2026, 1, 3)
    )
    assert out["trade_date"].tolist() == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")]
    assert out["symbol"].unique().tolist() == ["SH600000"]
    assert out["turnover_rate"].tolist() == [0.01, 0.02]
    assert out["close"].tolist() == [10.0, 21.0]
    assert session.calls[0]["t"] == "日k"


def test_stockdb_adj_factor_and_empty_daily_response():
    adapter = StockDBAdapter(session=_Session([[ ["复权:000001:20260102", {"cum": 1.2}] ], [{"date": 20260102, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "amount": 1}, {"date": 20260103, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "amount": 1}]]))
    out = adapter.fetch_field(
        "adj_factor", "SZ000001", start=date(2026, 1, 1), end=date(2026, 1, 3)
    )
    assert out["symbol"].unique().tolist() == ["SZ000001"]
    assert out["trade_date"].tolist() == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-03")]
    assert out["adj_factor"].tolist() == pytest.approx([1.2, 1.2])

    with pytest.raises(DataUnavailable):
        StockDBAdapter(session=_Session([[]])).fetch_daily(
            "SZ000001", date(2026, 1, 1), date(2026, 1, 2)
        )
