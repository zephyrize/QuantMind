#!/usr/bin/env python3
"""Inspect and compare Qlib feature .bin files without modifying any data.

Examples:
  python scripts/data/processing/inspect_qlib_bin.py \
    --symbol SH600027 --field close --tail 10
  python scripts/data/processing/inspect_qlib_bin.py \
    --file db/qlib_data/features/sh600027/close.day.bin \
    --compare /tmp/new_qlib/features/sh600027/close.day.bin
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path



DEFAULT_QLIB_DIR = Path("db/qlib_data")


@dataclass(frozen=True)
class BinSeries:
    path: Path
    start_index: int
    values: tuple[float, ...]
    calendar: list[str]

    @property
    def end_index(self) -> int:
        return self.start_index + len(self.values) - 1

    @property
    def dates(self) -> list[str]:
        return self.calendar[self.start_index : self.end_index + 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read or compare Qlib float32 feature .bin files (read-only)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path, help="Exact .bin file path")
    source.add_argument("--symbol", help="Instrument, e.g. SH600027 or sh600027")
    parser.add_argument("--field", help="Qlib field for --symbol, e.g. close/factor")
    parser.add_argument(
        "--qlib-dir", type=Path, default=DEFAULT_QLIB_DIR, help="Qlib data root"
    )
    parser.add_argument("--compare", type=Path, help="Second .bin file to compare")
    parser.add_argument("--tail", type=int, default=20, help="Rows to print (default: 20)")
    parser.add_argument("--csv", type=Path, help="Write parsed primary series to CSV")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute comparison tolerance")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative comparison tolerance")
    args = parser.parse_args()
    if args.symbol and not args.field:
        parser.error("--field is required with --symbol")
    if args.tail < 0:
        parser.error("--tail must be non-negative")
    return args


def resolve_path(args: argparse.Namespace) -> Path:
    if args.file:
        return args.file
    return args.qlib_dir / "features" / args.symbol.lower() / f"{args.field.lower()}.day.bin"


def load_calendar(qlib_dir: Path) -> list[str]:
    path = qlib_dir / "calendars" / "day.txt"
    if not path.is_file():
        raise ValueError(f"Qlib calendar not found: {path}")
    calendar = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    calendar = [value for value in calendar if value]
    if not calendar:
        raise ValueError(f"Qlib calendar is empty: {path}")
    return calendar


def read_bin(path: Path, calendar: list[str]) -> BinSeries:
    if not path.is_file():
        raise ValueError(f"Feature file not found: {path}")
    raw = path.read_bytes()
    if len(raw) < 4 or len(raw) % 4:
        raise ValueError(f"Feature file has an invalid float32 byte length: {path}")
    start_index_float = struct.unpack("<f", raw[:4])[0]
    start_index = int(start_index_float)
    if start_index != start_index_float or start_index < 0:
        raise ValueError(f"Invalid start index {start_index_float!r} in {path}")
    value_count = (len(raw) - 4) // 4
    values = struct.unpack(f"<{value_count}f", raw[4:]) if value_count else ()
    if start_index + len(values) > len(calendar):
        raise ValueError(
            f"{path}: index range {start_index}..{start_index + len(values) - 1} "
            f"exceeds calendar length {len(calendar)}"
        )
    return BinSeries(path=path, start_index=start_index, values=values, calendar=calendar)


def print_summary(series: BinSeries) -> None:
    finite = [value for value in series.values if math.isfinite(value)]
    first_date = series.dates[0] if len(series.values) else "-"
    last_date = series.dates[-1] if len(series.values) else "-"
    print(f"file:        {series.path}")
    print(f"start_index: {series.start_index}")
    print(f"end_index:   {series.end_index if len(series.values) else '-'}")
    print(f"rows:        {len(series.values)}")
    print(f"date_range:  {first_date} .. {last_date}")
    print(f"nan_rows:    {sum(math.isnan(value) for value in series.values)}")
    if finite:
        print(
            "value_range: "
            f"{min(finite):.8g} .. {max(finite):.8g} "
            f"(mean={sum(finite) / len(finite):.8g})"
        )


def print_rows(series: BinSeries, tail: int) -> None:
    if not tail or not len(series.values):
        return
    begin = max(0, len(series.values) - tail)
    print("\ndate,value")
    for day, value in zip(series.dates[begin:], series.values[begin:]):
        text = "nan" if math.isnan(value) else f"{value:.10g}"
        print(f"{day},{text}")


def write_csv(series: BinSeries, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "value"])
        writer.writerows(zip(series.dates, series.values))
    print(f"\nCSV written: {output}")


def compare_series(left: BinSeries, right: BinSeries, *, atol: float, rtol: float) -> int:
    left_values = dict(zip(left.dates, left.values))
    right_values = dict(zip(right.dates, right.values))
    left_dates, right_dates = set(left_values), set(right_values)
    common_dates = sorted(left_dates & right_dates)
    only_left, only_right = sorted(left_dates - right_dates), sorted(right_dates - left_dates)

    mismatches: list[tuple[str, float, float]] = []
    for day in common_dates:
        old, new = left_values[day], right_values[day]
        both_nan = math.isnan(old) and math.isnan(new)
        if not both_nan and not math.isclose(old, new, abs_tol=atol, rel_tol=rtol):
            mismatches.append((day, float(old), float(new)))

    print("\ncomparison")
    print(f"common_dates: {len(common_dates)}")
    print(f"only_primary: {len(only_left)}")
    print(f"only_compare: {len(only_right)}")
    print(f"value_mismatch: {len(mismatches)} (atol={atol:g}, rtol={rtol:g})")
    for day, old, new in mismatches[:20]:
        print(f"  {day}: primary={old:.10g}, compare={new:.10g}, delta={new - old:.10g}")
    if len(mismatches) > 20:
        print(f"  ... {len(mismatches) - 20} more mismatches")
    return 1 if only_left or only_right or mismatches else 0


def main() -> int:
    args = parse_args()
    try:
        calendar = load_calendar(args.qlib_dir)
        primary = read_bin(resolve_path(args), calendar)
        print_summary(primary)
        print_rows(primary, args.tail)
        if args.csv:
            write_csv(primary, args.csv)
        if args.compare:
            compared = read_bin(args.compare, calendar)
            return compare_series(primary, compared, atol=args.atol, rtol=args.rtol)
        return 0
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
