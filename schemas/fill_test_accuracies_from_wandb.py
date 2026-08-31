#!/usr/bin/env python3
"""Fill tested schema CSVs with chronologically matched W&B test accuracies.

By default this script processes every ``tested_*.csv`` below its own
directory.  A CSV row whose ``id`` is 1 receives ``test/accuracy`` from the
oldest run in that CSV's sweep, id 2 receives it from the second-oldest run,
and so on.

The W&B Public API reads credentials from the normal W&B configuration (for
example, ``WANDB_API_KEY`` or ``wandb login``).  All sweeps and rows are
fetched and validated before any CSV is written.

Examples:
    ../.venv/bin/python fill_test_accuracies_from_wandb.py --dry-run
    ../.venv/bin/python fill_test_accuracies_from_wandb.py \
        water_lr/tested_best_water_lr_scfe_configurations.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import stat
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DEFAULT_ENTITY = "counterfactual"
DEFAULT_PROJECT = "counterfactual_overfitting_experiments_new"
DEFAULT_WANDB_METRIC = "test/accuracy"
DEFAULT_CSV_COLUMN = "test_accuracy"
DEFAULT_PAGE_SIZE = 100
DEFAULT_RETRIES = 3
SCHEMAS_DIR = Path(__file__).resolve().parent


class FillError(RuntimeError):
    """Raised when data cannot be mapped to a CSV without ambiguity."""


class SnapshotChangedError(FillError):
    """Raised when a sweep changes while its cursor pages are being read."""


@dataclass(frozen=True)
class CsvTable:
    path: Path
    headers: list[str]
    rows: list[list[str]]
    newline: str


@dataclass(frozen=True)
class ChronologicalRun:
    run_id: str
    metric_value: Any


@dataclass(frozen=True)
class PlannedUpdate:
    table: CsvTable
    rows: list[list[str]]
    run_ids_by_ordinal: list[str]


def read_csv(path: Path) -> CsvTable:
    """Read and validate one target CSV while retaining its line ending."""

    text = path.read_text(encoding="utf-8-sig")
    newline = "\r\n" if "\r\n" in text else "\n"
    parsed_rows = [
        row
        for row in csv.reader(io.StringIO(text, newline=""))
        if any(cell.strip() for cell in row)
    ]

    if not parsed_rows:
        raise FillError(f"{path}: CSV is empty")

    headers = parsed_rows[0]
    if not headers or any(not header.strip() for header in headers):
        raise FillError(f"{path}: header contains an empty column name")
    if len(set(headers)) != len(headers):
        raise FillError(f"{path}: header contains duplicate column names")

    rows = parsed_rows[1:]
    if not rows:
        raise FillError(f"{path}: CSV has no nonblank data rows")
    for row_number, row in enumerate(rows, start=2):
        if len(row) != len(headers):
            raise FillError(
                f"{path}:{row_number}: expected {len(headers)} columns, "
                f"found {len(row)}"
            )

    return CsvTable(path=path, headers=headers, rows=rows, newline=newline)


def column_indexes(table: CsvTable) -> dict[str, int]:
    indexes = {header: index for index, header in enumerate(table.headers)}
    required = {"sweep_id", "id", DEFAULT_CSV_COLUMN}
    missing = sorted(required.difference(indexes))
    if missing:
        raise FillError(
            f"{table.path}: missing required columns: {', '.join(missing)}"
        )
    return indexes


def sweep_id_for(table: CsvTable) -> str:
    indexes = column_indexes(table)
    sweep_index = indexes["sweep_id"]
    values = [row[sweep_index].strip() for row in table.rows]

    if any(not value for value in values):
        raise FillError(f"{table.path}: every data row must contain its sweep_id")
    sweep_ids = set(values)
    if len(sweep_ids) != 1:
        raise FillError(
            f"{table.path}: expected one sweep_id, found {sorted(sweep_ids)!r}"
        )
    return values[0]


def row_ordinals(table: CsvTable) -> list[int]:
    """Return each row's one-based run ordinal and reject ambiguous IDs."""

    id_index = column_indexes(table)["id"]
    ordinals: list[int] = []
    locations: dict[int, int] = {}

    for row_number, row in enumerate(table.rows, start=2):
        raw_id = row[id_index].strip()
        try:
            ordinal = int(raw_id)
        except ValueError as error:
            raise FillError(
                f"{table.path}:{row_number}: id must be a positive integer, "
                f"found {raw_id!r}"
            ) from error
        if ordinal < 1 or str(ordinal) != raw_id:
            raise FillError(
                f"{table.path}:{row_number}: id must be a canonical positive "
                f"integer, found {raw_id!r}"
            )
        if ordinal in locations:
            raise FillError(
                f"{table.path}:{row_number}: duplicate id {ordinal}; it first "
                f"appeared on line {locations[ordinal]}"
            )
        locations[ordinal] = row_number
        ordinals.append(ordinal)

    expected = set(range(1, len(table.rows) + 1))
    actual = set(ordinals)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing ids {missing}")
        if unexpected:
            details.append(f"unexpected ids {unexpected}")
        raise FillError(f"{table.path}: ids must be 1..{len(table.rows)}; " + "; ".join(details))

    return ordinals


def _lookup_metric(summary: Mapping[str, Any], metric_name: str, context: str) -> Any:
    if metric_name in summary:
        return summary[metric_name]

    # Slash-delimited metrics normally appear as literal W&B summary keys, but
    # accepting an equivalent nested mapping makes this robust to API shapes.
    nested: Any = summary
    for component in metric_name.split("/"):
        if not isinstance(nested, Mapping) or component not in nested:
            raise FillError(f"{context}: missing W&B summary metric {metric_name!r}")
        nested = nested[component]
    return nested


def _format_metric(value: Any, context: str, metric_name: str) -> str:
    if isinstance(value, bool) or value is None:
        raise FillError(
            f"{context}: W&B summary metric {metric_name!r} is not numeric: {value!r}"
        )
    try:
        number = Decimal(repr(value) if isinstance(value, float) else str(value))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise FillError(
            f"{context}: W&B summary metric {metric_name!r} is not numeric: {value!r}"
        ) from error
    if not number.is_finite():
        raise FillError(
            f"{context}: W&B summary metric {metric_name!r} is not finite: {value!r}"
        )
    return repr(value) if isinstance(value, float) else str(value)


def _fetch_runs_once(
    api: Any,
    entity: str,
    project: str,
    sweep_id: str,
    metric_name: str,
    page_size: int,
) -> list[ChronologicalRun]:
    """Fetch every run page in oldest-to-newest creation order."""

    paginated_runs = api.runs(
        f"{entity}/{project}",
        filters={"sweep": sweep_id},
        order="+created_at",
        per_page=page_size,
        lazy=False,
    )

    # len() obtains W&B's filtered runCount; list() then exhausts all cursor
    # pages.  Comparing them catches common active-sweep pagination races.
    reported_count = len(paginated_runs)
    fetched = list(paginated_runs)
    if len(fetched) != reported_count:
        raise SnapshotChangedError(
            f"sweep {sweep_id}: W&B reported {reported_count} runs but returned "
            f"{len(fetched)} while fetching all pages"
        )

    results: list[ChronologicalRun] = []
    fetched_ids: set[str] = set()
    for run in fetched:
        run_id = str(getattr(run, "id", getattr(run, "name", "<unknown>")))
        if run_id in fetched_ids:
            raise SnapshotChangedError(
                f"sweep {sweep_id}: run {run_id!r} appeared on more than one page"
            )
        fetched_ids.add(run_id)

        context = f"sweep {sweep_id}, run {run_id}"
        try:
            summary = dict(run.summary)
        except Exception as error:
            raise FillError(f"{context}: could not read the W&B summary: {error}") from error
        metric_value = _lookup_metric(summary, metric_name, context)
        # Validate now, while the run object is loaded, but retain the original
        # value so the CSV representation is not rounded unnecessarily.
        _format_metric(metric_value, context, metric_name)
        results.append(ChronologicalRun(run_id=run_id, metric_value=metric_value))

    return results


def fetch_runs(
    api: Any,
    entity: str,
    project: str,
    sweep_id: str,
    metric_name: str,
    page_size: int,
    retries: int,
) -> list[ChronologicalRun]:
    """Fetch a complete sweep snapshot, retrying API and pagination failures."""

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return _fetch_runs_once(
                api=api,
                entity=entity,
                project=project,
                sweep_id=sweep_id,
                metric_name=metric_name,
                page_size=page_size,
            )
        except FillError as error:
            if not isinstance(error, SnapshotChangedError):
                raise
            last_error = error
        except Exception as error:
            # W&B transport/GraphQL exception classes differ across SDK versions.
            last_error = error
        finally:
            flush = getattr(api, "flush", None)
            if callable(flush):
                flush()

        if attempt < retries:
            delay = 2**attempt
            print(
                f"  warning: sweep {sweep_id} fetch failed ({last_error}); "
                f"retrying in {delay}s ({attempt + 1}/{retries})...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

    raise FillError(
        f"sweep {sweep_id}: fetch failed after {retries + 1} attempts: {last_error}"
    ) from last_error


def build_update(
    table: CsvTable,
    chronological_runs: Sequence[ChronologicalRun],
    metric_name: str = DEFAULT_WANDB_METRIC,
) -> PlannedUpdate:
    indexes = column_indexes(table)
    ordinals = row_ordinals(table)

    if len(chronological_runs) != len(table.rows):
        raise FillError(
            f"{table.path}: has {len(table.rows)} rows, but its sweep has "
            f"{len(chronological_runs)} runs; chronological mapping is ambiguous"
        )

    output_rows: list[list[str]] = []
    for source_row, ordinal in zip(table.rows, ordinals):
        run = chronological_runs[ordinal - 1]
        row = source_row.copy()
        row[indexes[DEFAULT_CSV_COLUMN]] = _format_metric(
            run.metric_value,
            context=f"{table.path}, id {ordinal}, run {run.run_id}",
            metric_name=metric_name,
        )
        output_rows.append(row)

    return PlannedUpdate(
        table=table,
        rows=output_rows,
        run_ids_by_ordinal=[run.run_id for run in chronological_runs],
    )


def render_update(update: PlannedUpdate) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator=update.table.newline)
    writer.writerow(update.table.headers)
    writer.writerows(update.rows)
    return output.getvalue()


def write_atomic(path: Path, content: str) -> None:
    """Replace one CSV atomically while retaining its permission bits."""

    mode = stat.S_IMODE(path.stat().st_mode)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def discover_csvs(
    arguments: Sequence[str], schemas_dir: Path = SCHEMAS_DIR
) -> list[Path]:
    """Resolve only tested_*.csv files contained below schemas_dir."""

    schemas_dir = schemas_dir.resolve()
    if not schemas_dir.is_dir():
        raise FillError(f"schemas directory does not exist: {schemas_dir}")

    if arguments:
        paths = [
            (
                Path(argument)
                if Path(argument).is_absolute()
                else schemas_dir / argument
            ).resolve()
            for argument in arguments
        ]
    else:
        paths = sorted(path.resolve() for path in schemas_dir.rglob("tested_*.csv"))

    if not paths:
        raise FillError(f"no tested_*.csv files found below {schemas_dir}")

    for path in paths:
        try:
            relative_path = path.relative_to(schemas_dir)
        except ValueError as error:
            raise FillError(f"CSV must be inside {schemas_dir}: {path}") from error
        if len(relative_path.parts) < 2:
            raise FillError(f"CSV must be inside a schemas subfolder: {path}")
        if (
            not path.is_file()
            or path.suffix.casefold() != ".csv"
            or not path.name.startswith("tested_")
        ):
            raise FillError(f"not a tested_*.csv file: {path}")

    if len(set(paths)) != len(paths):
        raise FillError("the same tested CSV was supplied more than once")
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_files",
        nargs="*",
        metavar="CSV",
        help=(
            "tested CSV paths relative to schemas/ "
            "(default: every tested_*.csv recursively)"
        ),
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--wandb-metric", default=DEFAULT_WANDB_METRIC)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and validate all mappings without writing any CSV",
    )
    args = parser.parse_args(argv)

    if args.page_size < 1:
        parser.error("--page-size must be positive")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    if not args.entity.strip():
        parser.error("--entity cannot be blank")
    if not args.project.strip():
        parser.error("--project cannot be blank")
    if not args.wandb_metric.strip():
        parser.error("--wandb-metric cannot be blank")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        paths = discover_csvs(args.csv_files)
        tables = [read_csv(path) for path in paths]
        table_sweeps = [(table, sweep_id_for(table)) for table in tables]

        expected_counts: dict[str, int] = {}
        for table, sweep_id in table_sweeps:
            row_ordinals(table)
            previous = expected_counts.setdefault(sweep_id, len(table.rows))
            if previous != len(table.rows):
                raise FillError(
                    f"sweep {sweep_id} is referenced by CSVs with conflicting "
                    f"row counts ({previous} and {len(table.rows)})"
                )

        try:
            import wandb
        except ImportError as error:
            raise FillError(
                "wandb is not installed; run this script with the repository's .venv"
            ) from error

        try:
            api = wandb.Api(timeout=args.timeout)
        except Exception as error:
            raise FillError(f"could not initialize the W&B API: {error}") from error

        sweep_cache: dict[str, list[ChronologicalRun]] = {}
        updates: list[PlannedUpdate] = []

        for table, sweep_id in table_sweeps:
            relative_path = table.path.relative_to(SCHEMAS_DIR)
            print(
                f"Fetching sweep {sweep_id} for {relative_path} "
                f"({len(table.rows)} rows)...",
                flush=True,
            )
            if sweep_id not in sweep_cache:
                sweep_cache[sweep_id] = fetch_runs(
                    api=api,
                    entity=args.entity,
                    project=args.project,
                    sweep_id=sweep_id,
                    metric_name=args.wandb_metric,
                    page_size=args.page_size,
                    retries=args.retries,
                )

            update = build_update(
                table,
                sweep_cache[sweep_id],
                metric_name=args.wandb_metric,
            )
            updates.append(update)
            print(
                f"  matched {len(update.rows)} rows in +created_at order",
                flush=True,
            )

        # Finish every network read and validation before replacing any file.
        rendered = [(update.table.path, render_update(update)) for update in updates]
        if args.dry_run:
            print(
                f"Dry run complete: validated {len(rendered)} tested CSV file(s); "
                "nothing written."
            )
        else:
            for path, content in rendered:
                write_atomic(path, content)
            print(f"Updated {len(rendered)} tested CSV file(s).")
        return 0
    except (FillError, OSError, csv.Error, UnicodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
