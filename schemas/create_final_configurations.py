#!/usr/bin/env python3
"""Create final configuration CSVs from locally tested configurations.

Every ``tested_*.csv`` below this script's directory is ranked by
``test_accuracy`` in descending order.  SCFE outputs contain the four
highest-accuracy rows; every other output contains the lowest-accuracy row.

This script only reads and writes local CSV files.  It does not use the W&B
API, so API authentication, pagination, and rate limits do not apply.

Example:
    python schemas/create_final_configurations.py
"""

from __future__ import annotations

import csv
import io
import os
import stat
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path


SCHEMAS_DIR = Path(__file__).resolve().parent
SOURCE_PREFIX = "tested_best_"
OUTPUT_PREFIX = "final_"
ACCURACY_COLUMN = "test_accuracy"
SCFE_ROW_COUNT = 4


class FinalConfigurationError(RuntimeError):
    """Raised when a final CSV cannot be produced unambiguously."""


@dataclass(frozen=True)
class CsvTable:
    path: Path
    header: list[str]
    rows: list[list[str]]
    newline: str


@dataclass(frozen=True)
class PlannedOutput:
    source: Path
    destination: Path
    header: list[str]
    rows: list[list[str]]
    newline: str


def discover_tested_csvs(root: Path = SCHEMAS_DIR) -> list[Path]:
    """Return tested CSVs contained in subdirectories of ``root``."""

    paths = sorted(
        path
        for path in root.rglob("tested_*.csv")
        if path.is_file() and path.parent != root
    )
    if not paths:
        raise FinalConfigurationError(f"{root}: no tested_*.csv files found")
    return paths


def read_csv(path: Path) -> CsvTable:
    """Read one CSV and validate its header and row widths."""

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        text = csv_file.read()
    newline = "\r\n" if "\r\n" in text else "\n"

    parsed_rows: list[tuple[int, list[str]]] = []
    try:
        reader = csv.reader(io.StringIO(text, newline=""))
        for row in reader:
            if any(cell.strip() for cell in row):
                parsed_rows.append((reader.line_num, row))
    except csv.Error as error:
        raise FinalConfigurationError(f"{path}: invalid CSV: {error}") from error

    if not parsed_rows:
        raise FinalConfigurationError(f"{path}: CSV is empty")

    header = parsed_rows[0][1]
    if not header or any(not column.strip() for column in header):
        raise FinalConfigurationError(f"{path}: header contains an empty column name")
    if len(set(header)) != len(header):
        raise FinalConfigurationError(f"{path}: header contains duplicate column names")

    rows = [row for _line_number, row in parsed_rows[1:]]
    if not rows:
        raise FinalConfigurationError(f"{path}: CSV has no data rows")

    for line_number, row in parsed_rows[1:]:
        if len(row) != len(header):
            raise FinalConfigurationError(
                f"{path}:{line_number}: expected {len(header)} columns, "
                f"found {len(row)}"
            )

    return CsvTable(path=path, header=header, rows=rows, newline=newline)


def rank_rows(table: CsvTable) -> list[list[str]]:
    """Return rows sorted from highest to lowest test accuracy."""

    try:
        accuracy_index = table.header.index(ACCURACY_COLUMN)
    except ValueError as error:
        raise FinalConfigurationError(
            f"{table.path}: missing required column {ACCURACY_COLUMN!r}"
        ) from error

    ranked: list[tuple[Decimal, list[str]]] = []
    for line_number, row in enumerate(table.rows, start=2):
        raw_accuracy = row[accuracy_index].strip()
        try:
            accuracy = Decimal(raw_accuracy)
        except InvalidOperation as error:
            raise FinalConfigurationError(
                f"{table.path}:{line_number}: {ACCURACY_COLUMN} must be numeric, "
                f"found {raw_accuracy!r}"
            ) from error
        if not accuracy.is_finite():
            raise FinalConfigurationError(
                f"{table.path}:{line_number}: {ACCURACY_COLUMN} must be finite, "
                f"found {raw_accuracy!r}"
            )
        ranked.append((accuracy, row))

    # Python's sort is stable, so equal-accuracy rows retain their input order.
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [row for _accuracy, row in ranked]


def destination_for(source: Path) -> Path:
    """Map tested_best_<name>.csv to final_<name>.csv."""

    if not source.name.startswith(SOURCE_PREFIX):
        raise FinalConfigurationError(
            f"{source}: expected a name starting with {SOURCE_PREFIX!r}"
        )
    return source.with_name(OUTPUT_PREFIX + source.name.removeprefix(SOURCE_PREFIX))


def plan_output(source: Path) -> PlannedOutput:
    table = read_csv(source)
    ranked_rows = rank_rows(table)

    if "scfe" in source.name.casefold():
        if len(ranked_rows) < SCFE_ROW_COUNT:
            raise FinalConfigurationError(
                f"{source}: SCFE input needs at least {SCFE_ROW_COUNT} data rows, "
                f"found {len(ranked_rows)}"
            )
        selected_rows = ranked_rows[:SCFE_ROW_COUNT]
    else:
        selected_rows = [ranked_rows[-1]]

    return PlannedOutput(
        source=source,
        destination=destination_for(source),
        header=table.header,
        rows=selected_rows,
        newline=table.newline,
    )


def write_output(output: PlannedOutput) -> None:
    """Atomically write one planned output next to its source CSV."""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output.destination.parent,
            prefix=f".{output.destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.writer(temporary_file, lineterminator=output.newline)
            writer.writerow(output.header)
            writer.writerows(output.rows)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        mode_source = output.destination if output.destination.exists() else output.source
        os.chmod(temporary_path, stat.S_IMODE(mode_source.stat().st_mode))
        os.replace(temporary_path, output.destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    try:
        # Validate and rank every input before writing any output.
        outputs = [plan_output(path) for path in discover_tested_csvs()]
        for output in outputs:
            write_output(output)
            print(f"Wrote {output.destination.relative_to(SCHEMAS_DIR)}")
    except (FinalConfigurationError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Created {len(outputs)} final configuration CSV files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
