from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "data"
    / "ingestion"
    / "sync_stockdb_to_qlib.py"
)
SPEC = importlib.util.spec_from_file_location("sync_stockdb_to_qlib", SCRIPT_PATH)
assert SPEC and SPEC.loader
sync_stockdb_to_qlib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sync_stockdb_to_qlib
SPEC.loader.exec_module(sync_stockdb_to_qlib)

DailyBar = sync_stockdb_to_qlib.DailyBar
SymbolBars = sync_stockdb_to_qlib.SymbolBars


def _record(*bars: DailyBar, events: tuple[tuple[str, float], ...]) -> SymbolBars:
    return SymbolBars("sh600000", bars, events)


def test_full_materialization_writes_qfq_ohlcv_and_vwap(tmp_path):
    calendar = ["2026-01-02", "2026-01-03"]
    record = _record(
        DailyBar("2026-01-02", 10, 11, 9, 10, 100, 1000),
        DailyBar("2026-01-03", 20, 22, 19, 21, 200, 4200),
        events=(("2026-01-01", 2.0), ("2026-01-03", 4.0)),
    )

    first, last = sync_stockdb_to_qlib.materialize_symbol(
        tmp_path, record, calendar, incremental=False
    )

    assert (first, last) == ("2026-01-02", "2026-01-03")
    feature_dir = tmp_path / "features" / "sh600000"
    _, close = sync_stockdb_to_qlib.read_bin(feature_dir / "close.day.bin")
    _, vwap = sync_stockdb_to_qlib.read_bin(feature_dir / "vwap.day.bin")
    _, factor = sync_stockdb_to_qlib.read_bin(feature_dir / "factor.day.bin")
    _, volume = sync_stockdb_to_qlib.read_bin(feature_dir / "volume.day.bin")
    assert close == pytest.approx([5.0, 21.0])
    assert vwap == pytest.approx([5.0, 21.0])
    assert factor == pytest.approx([0.5, 1.0])
    assert volume == pytest.approx([100.0, 200.0])


def test_incremental_materialization_rescales_existing_qfq_prices(tmp_path):
    initial_calendar = ["2026-01-02", "2026-01-03"]
    initial = _record(
        DailyBar("2026-01-02", 10, 10, 10, 10, 100, 1000),
        DailyBar("2026-01-03", 10, 10, 10, 10, 100, 1000),
        events=(("2026-01-01", 2.0), ("2026-01-03", 4.0)),
    )
    sync_stockdb_to_qlib.materialize_symbol(
        tmp_path, initial, initial_calendar, incremental=False
    )
    updated_calendar = [*initial_calendar, "2026-01-04"]
    update = _record(
        DailyBar("2026-01-04", 20, 20, 20, 20, 100, 2000),
        events=(
            ("2026-01-01", 2.0),
            ("2026-01-03", 4.0),
            ("2026-01-04", 8.0),
        ),
    )

    sync_stockdb_to_qlib.materialize_symbol(
        tmp_path, update, updated_calendar, incremental=True
    )

    feature_dir = tmp_path / "features" / "sh600000"
    _, close = sync_stockdb_to_qlib.read_bin(feature_dir / "close.day.bin")
    _, factor = sync_stockdb_to_qlib.read_bin(feature_dir / "factor.day.bin")
    assert close == pytest.approx([2.5, 5.0, 20.0])
    assert factor == pytest.approx([0.25, 0.5, 1.0])


def test_incremental_rejects_historical_rows(tmp_path):
    initial_calendar = ["2026-01-02", "2026-01-03"]
    initial = _record(
        DailyBar("2026-01-02", 10, 10, 10, 10, 100, 1000),
        DailyBar("2026-01-03", 10, 10, 10, 10, 100, 1000),
        events=(),
    )
    sync_stockdb_to_qlib.materialize_symbol(
        tmp_path, initial, initial_calendar, incremental=False
    )
    historical = _record(
        DailyBar("2026-01-03", 10, 10, 10, 10, 100, 1000), events=()
    )

    with pytest.raises(ValueError, match="use --rebuild"):
        sync_stockdb_to_qlib.materialize_symbol(
            tmp_path, historical, initial_calendar, incremental=True
        )
