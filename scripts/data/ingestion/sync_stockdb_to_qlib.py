#!/usr/bin/env python3
"""Synchronize StockDB daily bars into the local Qlib binary data layout.

The task is deliberately independent from the online data-source routing: QuantDB
remains the default source for existing services.  This script is an explicit,
offline StockDB -> ``db/qlib_data`` materialization step for Qlib training and
backtesting.

It writes these Qlib-native fields for every symbol: open, high, low, close,
vwap, volume, amount and factor.  Prices and VWAP are QFQ-adjusted using
StockDB's cumulative adjustment factor; ``factor`` is the same QFQ multiplier.

Examples:
  # Preview a small initial build (does not write files)
  python scripts/data/ingestion/sync_stockdb_to_qlib.py \
    --start 2016-01-01 --max-symbols 10

  # Create db/qlib_data from scratch
  python scripts/data/ingestion/sync_stockdb_to_qlib.py \
    --start 2016-01-01 --apply

  # Append newly available trading days to an existing Qlib directory
  python scripts/data/ingestion/sync_stockdb_to_qlib.py --apply
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import math
import os
import re
import shutil
import struct
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QLIB_DIR = PROJECT_ROOT / "db" / "qlib_data"
FEATURE_FIELDS = ("open", "high", "low", "close", "vwap", "volume", "amount", "factor")
PRICE_FIELDS = ("open", "high", "low", "close", "vwap")
REQUIRED_DAILY_FIELDS = ("open", "high", "low", "close", "volume", "amount")
DEFAULT_START = date(2000, 1, 1)


def load_backend_env() -> None:
    """Load non-secret StockDB settings from backend/.env without python-dotenv.

    Existing process environment variables always win, matching normal deployment
    behavior. This lets a direct offline invocation use the same StockDB host
    and port configured for the backend service.
    """
    env_file = PROJECT_ROOT / "backend" / ".env"
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip(chr(34)).strip(chr(39))
        if key:
            os.environ.setdefault(key, value)


class StockDBError(RuntimeError):
    """The local StockDB endpoint returned unusable data."""


@dataclasses.dataclass(frozen=True)
class DailyBar:
    day: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float


@dataclasses.dataclass(frozen=True)
class SymbolBars:
    instrument: str
    bars: tuple[DailyBar, ...]
    factor_events: tuple[tuple[str, float], ...]


@dataclasses.dataclass(frozen=True)
class SyncSummary:
    selected_symbols: int
    fetched_symbols: int
    skipped_symbols: int
    calendar_days: int
    calendar_start: str
    calendar_end: str
    mode: str
    applied: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize StockDB A-share daily data to Qlib binary files."
    )
    parser.add_argument(
        "--qlib-dir",
        type=Path,
        default=DEFAULT_QLIB_DIR,
        help=f"Target Qlib data directory (default: {DEFAULT_QLIB_DIR})",
    )
    parser.add_argument(
        "--start",
        type=parse_day,
        help="Inclusive start date (YYYY-MM-DD). Defaults to 2000-01-01 for a new build.",
    )
    parser.add_argument(
        "--end",
        type=parse_day,
        default=date.today(),
        help="Inclusive end date (YYYY-MM-DD, default: today).",
    )
    parser.add_argument(
        "--symbols",
        help="Comma-separated A-share symbols, e.g. SH600000,SZ000001.",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        help="Text file containing one A-share symbol per line.",
    )
    parser.add_argument(
        "--max-symbols",
        type=positive_int,
        help="Only process the first N selected symbols (useful for a smoke test).",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=8,
        help="Concurrent StockDB requests (default: 8).",
    )
    parser.add_argument(
        "--lookup-days",
        type=positive_int,
        default=14,
        help="How far backward to find a recent StockDB stock list (default: 14).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the target directory atomically instead of incrementally appending.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write files. Without this flag the script only fetches and validates data.",
    )
    parser.add_argument("--host", help="StockDB host (overrides STOCKDB_HOST).")
    parser.add_argument("--port", type=positive_int, help="StockDB port (overrides STOCKDB_PORT).")
    parser.add_argument(
        "--timeout",
        type=positive_float,
        help="Per-request read timeout in seconds (overrides STOCKDB_HTTP_TIMEOUT_SECONDS).",
    )
    args = parser.parse_args()
    if args.end < (args.start or DEFAULT_START):
        parser.error("--end must not be earlier than --start")
    if args.symbols and args.symbols_file:
        parser.error("--symbols and --symbols-file cannot be used together")
    return args


def parse_day(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


class StockDBClient:
    """Small HTTP client for only the StockDB endpoints used by this ETL."""

    def __init__(self, host: str, port: int, timeout_seconds: float) -> None:
        self.base_url = f"http://{host}:{port}/"
        self.timeout_seconds = timeout_seconds

    def request(self, params: dict[str, str]) -> Any:
        try:
            response = requests.get(
                self.base_url, params=params, timeout=(3.05, self.timeout_seconds)
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise StockDBError(f"StockDB request failed ({self.base_url}): {exc}") from exc

    def recent_stock_codes(self, end: date, lookback_days: int) -> list[str]:
        for offset in range(lookback_days):
            candidate = end - timedelta(days=offset)
            payload = self.request(
                {
                    "cmd": "keys",
                    "t": "日k",
                    "k1": "all:",
                    "k2": f"key:{candidate:%Y%m%d}",
                }
            )
            codes = sorted(
                {
                    code
                    for item in payload if isinstance(payload, list)
                    for code in [stockdb_key_code(item)]
                    if is_a_share_code(code)
                }
            )
            if codes:
                print(f"Using StockDB universe from {candidate.isoformat()}: {len(codes)} A-share symbols")
                return codes
        raise StockDBError(
            f"No A-share symbols found in the {lookback_days} days up to {end.isoformat()}"
        )

    def daily_bars(self, code: str, start: date, end: date) -> tuple[DailyBar, ...]:
        payload = self.request(
            {
                "cmd": "vals",
                "t": "日k",
                "k1": f"key:{code}",
                "k2": f"fwd:{start:%Y%m%d},{end:%Y%m%d}",
            }
        )
        by_day: dict[str, DailyBar] = {}
        for row in payload if isinstance(payload, list) else []:
            if not isinstance(row, dict):
                continue
            bar = parse_daily_bar(row)
            if bar:
                by_day[bar.day] = bar
        return tuple(by_day[day] for day in sorted(by_day))

    def factor_events(self, code: str) -> tuple[tuple[str, float], ...]:
        payload = self.request(
            {"cmd": "get", "t": "复权", "k1": f"key:{code}", "k2": "all:"}
        )
        events: dict[str, float] = {}
        for item in payload if isinstance(payload, list) else []:
            if not (isinstance(item, list) and len(item) == 2 and isinstance(item[1], dict)):
                continue
            event_day = stockdb_key_day(item[0])
            value = as_finite_float(item[1].get("cum"))
            if event_day and value is not None and value > 0:
                events[event_day] = value
        return tuple(sorted(events.items()))


def stockdb_key_code(value: Any) -> str:
    parts = str(value).split(":")
    return parts[-2] if len(parts) >= 3 and re.fullmatch(r"\d{6}", parts[-2]) else ""


def stockdb_key_day(value: Any) -> str:
    parts = str(value).split(":")
    raw = parts[-1] if len(parts) >= 3 else ""
    return normalize_day(raw)


def normalize_day(value: Any) -> str:
    raw = str(value).strip()
    if re.fullmatch(r"\d{8}", raw):
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw
    return ""


def is_a_share_code(code: str) -> bool:
    return code.startswith(("600", "601", "603", "605", "688", "000", "001", "002", "003", "300", "301", "4", "8"))


def to_qlib_symbol(value: str) -> str:
    digits = re.search(r"\d{6}", str(value))
    if not digits:
        raise ValueError(f"invalid A-share symbol: {value}")
    code = digits.group(0)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("0", "2", "3")):
        return f"sz{code}"
    if code.startswith(("4", "8")):
        return f"bj{code}"
    raise ValueError(f"cannot infer exchange for: {value}")


def read_symbols(args: argparse.Namespace, client: StockDBClient) -> list[str]:
    if args.symbols:
        raw = args.symbols.split(",")
    elif args.symbols_file:
        raw = args.symbols_file.read_text(encoding="utf-8").splitlines()
    else:
        raw = client.recent_stock_codes(args.end, args.lookup_days)
    symbols = sorted({to_qlib_symbol(item) for item in raw if str(item).strip()})
    if not symbols:
        raise StockDBError("No valid A-share symbols selected")
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]
    return symbols


def as_finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_daily_bar(row: dict[str, Any]) -> DailyBar | None:
    day = normalize_day(row.get("date"))
    if not day:
        return None
    values = {field: as_finite_float(row.get(field)) for field in REQUIRED_DAILY_FIELDS}
    if any(value is None for value in values.values()):
        return None
    return DailyBar(day=day, **values)  # type: ignore[arg-type]


def fetch_symbol(client: StockDBClient, symbol: str, start: date, end: date) -> SymbolBars:
    code = symbol[2:]
    bars = client.daily_bars(code, start, end)
    if not bars:
        raise StockDBError(f"{symbol}: no daily bars in requested range")
    return SymbolBars(
        instrument=symbol,
        bars=bars,
        factor_events=client.factor_events(code),
    )


def load_calendar(qlib_dir: Path) -> list[str]:
    path = qlib_dir / "calendars" / "day.txt"
    if not path.is_file():
        return []
    return sorted({line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()})


def read_bin(path: Path) -> tuple[int, list[float]]:
    raw = path.read_bytes()
    if len(raw) < 4 or len(raw) % 4:
        raise ValueError(f"invalid Qlib binary file: {path}")
    start_raw = struct.unpack("<f", raw[:4])[0]
    start_index = int(start_raw)
    if start_index != start_raw or start_index < 0:
        raise ValueError(f"invalid Qlib start index in {path}: {start_raw}")
    count = (len(raw) - 4) // 4
    return start_index, list(struct.unpack(f"<{count}f", raw[4:])) if count else []


def write_bin(path: Path, start_index: int, values: Iterable[float]) -> None:
    packed = [float(start_index), *values]
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_bytes(struct.pack(f"<{len(packed)}f", *packed))
    temp_path.replace(path)


def factor_map(events: tuple[tuple[str, float], ...], days: Iterable[str]) -> dict[str, float]:
    event_iter = iter(events)
    current = 1.0
    next_event = next(event_iter, None)
    latest = events[-1][1] if events else 1.0
    result: dict[str, float] = {}
    for day in sorted(days):
        while next_event and next_event[0] <= day:
            current = next_event[1]
            next_event = next(event_iter, None)
        result[day] = current / latest if latest > 0 else 1.0
    return result


def raw_vwap(bar: DailyBar) -> float:
    return bar.amount / bar.volume if bar.volume > 0 else math.nan


def aligned_new_values(
    data: SymbolBars, calendar: list[str]
) -> tuple[int, dict[str, list[float]]]:
    bars_by_day = {bar.day: bar for bar in data.bars}
    positions = [index for index, day in enumerate(calendar) if day in bars_by_day]
    if not positions:
        raise ValueError(f"{data.instrument} has no bars in the Qlib calendar")
    start_index, end_index = min(positions), max(positions)
    factors = factor_map(data.factor_events, calendar[start_index : end_index + 1])
    values = {field: [math.nan] * (end_index - start_index + 1) for field in FEATURE_FIELDS}
    for index in positions:
        bar = bars_by_day[calendar[index]]
        offset = index - start_index
        multiplier = factors[bar.day]
        values["open"][offset] = bar.open * multiplier
        values["high"][offset] = bar.high * multiplier
        values["low"][offset] = bar.low * multiplier
        values["close"][offset] = bar.close * multiplier
        values["vwap"][offset] = raw_vwap(bar) * multiplier
        values["volume"][offset] = bar.volume
        values["amount"][offset] = bar.amount
        values["factor"][offset] = multiplier
    return start_index, values


def rescale_existing_prices(
    values: dict[str, list[float]], old_factors: list[float], new_factors: list[float]
) -> None:
    for field in PRICE_FIELDS:
        for index, old_factor in enumerate(old_factors):
            value = values[field][index]
            new_factor = new_factors[index]
            if math.isfinite(value) and math.isfinite(old_factor) and old_factor != 0:
                values[field][index] = value * new_factor / old_factor


def materialize_symbol(
    root: Path,
    data: SymbolBars,
    calendar: list[str],
    *,
    incremental: bool,
) -> tuple[str, str]:
    """Write one symbol and return its actual first/last observed dates.

    Incremental mode accepts appended calendar days only.  It still refreshes all
    previous QFQ prices when a new corporate-action factor changes the QFQ base.
    """
    start_index, incoming = aligned_new_values(data, calendar)
    feature_dir = root / "features" / data.instrument
    existing_factor_path = feature_dir / "factor.day.bin"
    if not incremental or not existing_factor_path.is_file():
        feature_dir.mkdir(parents=True, exist_ok=True)
        for field, values in incoming.items():
            write_bin(feature_dir / f"{field}.day.bin", start_index, values)
        return data.bars[0].day, data.bars[-1].day

    old_start, old_factors = read_bin(existing_factor_path)
    old_end = old_start + len(old_factors) - 1
    if old_start < 0 or old_end >= len(calendar):
        raise ValueError(f"{data.instrument}: existing factor file is outside calendar")
    old_days = calendar[old_start : old_end + 1]
    current_factor_by_day = factor_map(data.factor_events, old_days)
    current_factors = [current_factor_by_day[day] for day in old_days]
    existing: dict[str, list[float]] = {}
    for field in FEATURE_FIELDS:
        path = feature_dir / f"{field}.day.bin"
        if not path.is_file():
            raise ValueError(f"{data.instrument}: missing existing field {field}")
        field_start, values = read_bin(path)
        if field_start != old_start or len(values) != len(old_factors):
            raise ValueError(f"{data.instrument}: inconsistent field length for {field}")
        existing[field] = values
    rescale_existing_prices(existing, old_factors, current_factors)
    existing["factor"] = current_factors

    # Incoming data must start strictly after old Qlib data.  Earlier backfills
    # can change bin offsets, so require --rebuild for that safe full rewrite.
    incoming_first = data.bars[0].day
    if incoming_first <= old_days[-1]:
        raise ValueError(
            f"{data.instrument}: received historical data through {old_days[-1]}; use --rebuild"
        )
    append_begin = old_end + 1
    new_last = start_index + len(incoming["factor"]) - 1
    if new_last < append_begin:
        for field, values in existing.items():
            write_bin(feature_dir / f"{field}.day.bin", old_start, values)
        return data.bars[0].day, data.bars[-1].day
    for field in FEATURE_FIELDS:
        extension = [math.nan] * (new_last - append_begin + 1)
        for calendar_index in range(max(append_begin, start_index), new_last + 1):
            extension[calendar_index - append_begin] = incoming[field][calendar_index - start_index]
        existing[field].extend(extension)
        write_bin(feature_dir / f"{field}.day.bin", old_start, existing[field])
    return old_days[0], data.bars[-1].day


def load_instruments(root: Path) -> dict[str, tuple[str, str]]:
    path = root / "instruments" / "all.txt"
    if not path.is_file():
        return {}
    result: dict[str, tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3:
            result[parts[0].lower()] = (parts[1], parts[2])
    return result


def write_text_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def write_calendar_and_instruments(
    root: Path, calendar: list[str], instruments: dict[str, tuple[str, str]]
) -> None:
    write_text_atomically(root / "calendars" / "day.txt", "\n".join(calendar) + "\n")
    lines = [
        f"{symbol}\t{start}\t{end}"
        for symbol, (start, end) in sorted(instruments.items())
    ]
    write_text_atomically(root / "instruments" / "all.txt", "\n".join(lines) + "\n")


def collect_data(
    client: StockDBClient, symbols: list[str], start: date, end: date, workers: int
) -> tuple[list[SymbolBars], dict[str, str]]:
    collected: list[SymbolBars] = []
    errors: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_symbol, client, symbol, start, end): symbol for symbol in symbols
        }
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                collected.append(future.result())
            except Exception as exc:  # noqa: BLE001 - report every failed symbol
                errors[symbol] = str(exc)
    return sorted(collected, key=lambda item: item.instrument), errors


def require_append_only(existing_calendar: list[str], records: list[SymbolBars]) -> None:
    if not existing_calendar:
        return
    existing_last = existing_calendar[-1]
    old_dates = sorted({bar.day for record in records for bar in record.bars if bar.day <= existing_last})
    if old_dates:
        raise ValueError(
            "Existing Qlib data can only be incrementally appended. "
            f"StockDB returned historical date {old_dates[0]}; rerun with --rebuild."
        )


def sync(args: argparse.Namespace) -> SyncSummary:
    load_backend_env()
    host = args.host or os.getenv("STOCKDB_HOST", "localhost")
    port = args.port or int(os.getenv("STOCKDB_PORT", "7899"))
    timeout = args.timeout or float(os.getenv("STOCKDB_HTTP_TIMEOUT_SECONDS", "15"))
    client = StockDBClient(host, port, timeout)
    qlib_dir = args.qlib_dir.expanduser().resolve()
    existing_calendar = load_calendar(qlib_dir)
    is_incremental = bool(existing_calendar) and not args.rebuild
    start = args.start or (
        datetime.strptime(existing_calendar[-1], "%Y-%m-%d").date() + timedelta(days=1)
        if is_incremental
        else DEFAULT_START
    )
    symbols = read_symbols(args, client)
    print(
        f"StockDB: {client.base_url} | mode={'incremental' if is_incremental else 'rebuild'} | "
        f"range={start.isoformat()}..{args.end.isoformat()} | symbols={len(symbols)}"
    )
    records, errors = collect_data(client, symbols, start, args.end, args.workers)
    for symbol, message in sorted(errors.items()):
        print(f"WARN {symbol}: {message}", file=sys.stderr)
    if not records:
        raise StockDBError("No symbol data was fetched; Qlib output was not changed")
    if is_incremental:
        require_append_only(existing_calendar, records)
    calendar = sorted({*existing_calendar, *(bar.day for record in records for bar in record.bars)})
    if not calendar:
        raise StockDBError("No valid trading days were found")

    if not args.apply:
        return SyncSummary(
            selected_symbols=len(symbols),
            fetched_symbols=len(records),
            skipped_symbols=len(errors),
            calendar_days=len(calendar),
            calendar_start=calendar[0],
            calendar_end=calendar[-1],
            mode="incremental" if is_incremental else "rebuild",
            applied=False,
        )

    if is_incremental:
        target_root = qlib_dir
    else:
        target_root = Path(tempfile.mkdtemp(prefix="qlib-stockdb-", dir=qlib_dir.parent))
    instruments = load_instruments(qlib_dir) if is_incremental else {}
    try:
        for record in records:
            first_day, last_day = materialize_symbol(
                target_root, record, calendar, incremental=is_incremental
            )
            instruments[record.instrument] = (first_day, last_day)
        write_calendar_and_instruments(target_root, calendar, instruments)
        if not is_incremental:
            if qlib_dir.exists():
                backup = qlib_dir.with_name(f"{qlib_dir.name}.pre-stockdb-backup")
                if backup.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite an existing backup directory: {backup}"
                    )
                qlib_dir.replace(backup)
                try:
                    target_root.replace(qlib_dir)
                except Exception:
                    backup.replace(qlib_dir)
                    raise
                shutil.rmtree(backup)
            else:
                target_root.replace(qlib_dir)
    finally:
        if not is_incremental and target_root.exists():
            shutil.rmtree(target_root)
    return SyncSummary(
        selected_symbols=len(symbols),
        fetched_symbols=len(records),
        skipped_symbols=len(errors),
        calendar_days=len(calendar),
        calendar_start=calendar[0],
        calendar_end=calendar[-1],
        mode="incremental" if is_incremental else "rebuild",
        applied=True,
    )


def main() -> int:
    args = parse_args()
    try:
        summary = sync(args)
    except (StockDBError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    status = "applied" if summary.applied else "dry-run"
    print(
        f"{status}: mode={summary.mode}, selected={summary.selected_symbols}, "
        f"fetched={summary.fetched_symbols}, skipped={summary.skipped_symbols}, "
        f"calendar={summary.calendar_days} ({summary.calendar_start}..{summary.calendar_end})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
