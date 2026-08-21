#!/usr/bin/env python3
"""Fill schema CSV blanks from the highest-accuracy runs in their W&B sweeps.

By default, every CSV below this script's directory is processed.  No file is
written until every sweep has been fetched and every output row has passed
validation.

Example:
    ../.venv/bin/python fill_schemas_from_wandb.py

To inspect one file without changing it:
    ../.venv/bin/python fill_schemas_from_wandb.py \
        --dry-run water_lr/best_water_lr_l1_configurations.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import stat
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DEFAULT_ENTITY = "counterfactual"
DEFAULT_PROJECT = "counterfactual_overfitting_experiments_new"
DEFAULT_WANDB_METRIC = "validation/accuracy"
DEFAULT_CSV_METRIC = "validation_accuracy"
DEFAULT_PAGE_SIZE = 100
DEFAULT_RETRIES = 3
SCHEMAS_DIR = Path(__file__).resolve().parent

_BRACKET_COMMA = "\x1f"
_PARAMETER_NAME_RE = re.compile(r"^parameter_(\d+)_name$")
_NON_ALPHANUMERIC_RE = re.compile(r"[^0-9A-Za-z]+")


class SchemaError(RuntimeError):
    """Raised when a schema or a W&B result cannot be mapped safely."""


class MissingValueError(SchemaError):
    """Raised when a requested W&B metric or config value is absent."""


class SnapshotChangedError(SchemaError):
    """Raised when a sweep changes while its cursor pages are being read."""


@dataclass(frozen=True)
class SchemaTable:
    path: Path
    headers: list[str]
    rows: list[list[str]]
    newline: str


@dataclass(frozen=True)
class RankedRun:
    run_id: str
    metric_value: Any
    metric_decimal: Decimal
    config: Mapping[str, Any]


@dataclass(frozen=True)
class SweepResult:
    ranked_runs: list[RankedRun]
    fetched_runs: int
    eligible_runs: int


@dataclass(frozen=True)
class PlannedUpdate:
    table: SchemaTable
    rows: list[list[str]]
    fetched_runs: int
    eligible_runs: int
    selected_run_ids: list[str]


def _protect_bracket_commas(text: str, path: Path) -> str:
    """Hide commas inside unquoted list/dict/tuple cells from csv.reader.

    Some existing schemas contain cells such as ``[150, 1000, 150, 30]``
    without CSV quoting.  Treating every comma as a delimiter shifts all the
    columns after ``model_hidden_layers``.  Quoted CSV fields continue to be
    handled by the standard library reader.
    """

    if _BRACKET_COMMA in text:
        raise SchemaError(f"{path}: contains the parser's reserved control character")

    output: list[str] = []
    closing_brackets: list[str] = []
    in_quotes = False
    index = 0
    bracket_pairs = {"[": "]", "{": "}", "(": ")"}

    while index < len(text):
        character = text[index]

        if character == "," and closing_brackets:
            output.append(_BRACKET_COMMA)
            index += 1
            continue

        if character == '"':
            if in_quotes and index + 1 < len(text) and text[index + 1] == '"':
                output.extend(('"', '"'))
                index += 2
                continue
            in_quotes = not in_quotes
        elif not in_quotes:
            if character in bracket_pairs:
                closing_brackets.append(bracket_pairs[character])
            elif character in "]})":
                if not closing_brackets or closing_brackets[-1] != character:
                    raise SchemaError(f"{path}: unmatched bracket {character!r}")
                closing_brackets.pop()

        output.append(character)
        index += 1

    if in_quotes:
        raise SchemaError(f"{path}: unterminated quoted CSV field")
    if closing_brackets:
        raise SchemaError(f"{path}: unterminated bracketed CSV field")

    return "".join(output)


def read_schema(path: Path) -> SchemaTable:
    text = path.read_text(encoding="utf-8-sig")
    newline = "\r\n" if "\r\n" in text else "\n"
    protected = _protect_bracket_commas(text, path)

    parsed_rows: list[list[str]] = []
    reader = csv.reader(io.StringIO(protected, newline=""), skipinitialspace=True)
    for row in reader:
        restored = [cell.replace(_BRACKET_COMMA, ",").strip() for cell in row]
        if any(restored):
            parsed_rows.append(restored)

    if not parsed_rows:
        raise SchemaError(f"{path}: CSV is empty")

    headers = parsed_rows[0]
    if not headers or any(not header for header in headers):
        raise SchemaError(f"{path}: header contains an empty column name")
    if len(set(headers)) != len(headers):
        raise SchemaError(f"{path}: header contains duplicate column names")

    data_rows = parsed_rows[1:]
    for row_number, row in enumerate(data_rows, start=2):
        if len(row) != len(headers):
            raise SchemaError(
                f"{path}:{row_number}: expected {len(headers)} columns, found {len(row)}"
            )

    if not data_rows:
        raise SchemaError(f"{path}: CSV has no nonblank data rows")

    return SchemaTable(path=path, headers=headers, rows=data_rows, newline=newline)


def _canonical_key(key: str) -> str:
    """Map W&B paths such as loss.alpha to CSV names such as loss_alpha."""

    return _NON_ALPHANUMERIC_RE.sub("_", key).strip("_").casefold()


def _leaf_items(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if not prefix and key.startswith("_"):
                continue
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(child, Mapping):
                yield from _leaf_items(child, path)
            else:
                yield path, child
    elif prefix:
        yield prefix, value


def lookup_value(mapping: Mapping[str, Any], requested_name: str, context: str) -> Any:
    requested = _canonical_key(requested_name)
    matches = [
        (path, value)
        for path, value in _leaf_items(mapping)
        if _canonical_key(path) == requested
    ]

    if not matches:
        raise MissingValueError(
            f"{context}: W&B has no value matching {requested_name!r}"
        )

    _first_path, first_value = matches[0]
    for _other_path, other_value in matches[1:]:
        if type(other_value) is not type(first_value) or repr(other_value) != repr(first_value):
            paths = ", ".join(repr(path) for path, _ in matches)
            raise SchemaError(
                f"{context}: {requested_name!r} ambiguously matches W&B paths {paths}"
            )

    return first_value


def _as_finite_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = Decimal(repr(value) if isinstance(value, float) else str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return number if number.is_finite() else None


def format_cell(value: Any) -> str:
    """Serialize values without rounding numeric data returned by W&B."""

    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if value is None:
        return "None"
    return str(value)


def _column_indexes(headers: Sequence[str], path: Path) -> dict[str, int]:
    indexes = {header: index for index, header in enumerate(headers)}
    required = {"sweep_id", "id", DEFAULT_CSV_METRIC}
    missing = sorted(required.difference(indexes))
    if missing:
        raise SchemaError(f"{path}: missing required columns: {', '.join(missing)}")
    return indexes


def sweep_id_for(table: SchemaTable) -> str:
    indexes = _column_indexes(table.headers, table.path)
    sweep_index = indexes["sweep_id"]
    sweep_ids = {row[sweep_index].strip() for row in table.rows if row[sweep_index].strip()}
    if len(sweep_ids) != 1:
        raise SchemaError(
            f"{table.path}: expected one nonblank sweep_id, found {sorted(sweep_ids)!r}"
        )
    if any(not row[sweep_index].strip() for row in table.rows):
        raise SchemaError(f"{table.path}: every data row must contain its sweep_id")
    return next(iter(sweep_ids))


def _fetch_ranked_runs_once(
    api: Any,
    entity: str,
    project: str,
    sweep_id: str,
    wandb_metric: str,
    page_size: int,
    max_results: int,
) -> SweepResult:
    """Fetch one complete, creation-ordered snapshot and rank it locally."""

    paginated_runs = api.runs(
        f"{entity}/{project}",
        filters={"sweep": sweep_id},
        # Unlike a live summary metric, creation time cannot move an existing
        # run between cursor pages while this sweep is still active.
        order="+created_at",
        per_page=page_size,
        lazy=False,
    )

    # len() reads W&B's filtered runCount. Iterating the collection exhausts
    # every GraphQL page rather than silently accepting the first page.
    reported_run_count = len(paginated_runs)
    fetched = list(paginated_runs)
    if len(fetched) != reported_run_count:
        raise SnapshotChangedError(
            f"sweep {sweep_id}: W&B reported {reported_run_count} runs but returned "
            f"{len(fetched)} after pagination"
        )

    ranked: list[RankedRun] = []
    fetched_ids: set[str] = set()
    for run in fetched:
        run_id = str(getattr(run, "id", getattr(run, "name", "<unknown>")))
        if run_id in fetched_ids:
            raise SnapshotChangedError(
                f"sweep {sweep_id}: run {run_id!r} appeared on more than one page"
            )
        fetched_ids.add(run_id)
        summary = dict(run.summary)
        try:
            metric_value = lookup_value(
                summary,
                wandb_metric,
                context=f"sweep {sweep_id}, run {run_id}",
            )
        except MissingValueError:
            # Failed, cancelled, or still-starting runs commonly have no final
            # validation metric. They cannot participate in a numeric ranking.
            continue

        metric_decimal = _as_finite_decimal(metric_value)
        if metric_decimal is None:
            continue

        ranked.append(
            RankedRun(
                run_id=run_id,
                metric_value=metric_value,
                metric_decimal=metric_decimal,
                config=dict(run.config),
            )
        )

    # Accuracy is the primary key. Run ID provides a deterministic order for
    # exact ties, which W&B's API does not otherwise guarantee.
    ranked.sort(key=lambda run: (-run.metric_decimal, run.run_id))
    return SweepResult(
        ranked_runs=ranked[:max_results],
        fetched_runs=len(fetched),
        eligible_runs=len(ranked),
    )


def fetch_ranked_runs(
    api: Any,
    entity: str,
    project: str,
    sweep_id: str,
    wandb_metric: str,
    page_size: int,
    max_results: int,
    retries: int,
) -> SweepResult:
    """Fetch all pages, retrying transient or changing-snapshot failures."""

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return _fetch_ranked_runs_once(
                api=api,
                entity=entity,
                project=project,
                sweep_id=sweep_id,
                wandb_metric=wandb_metric,
                page_size=page_size,
                max_results=max_results,
            )
        except SchemaError as error:
            if not isinstance(error, SnapshotChangedError):
                raise
            last_error = error
        except Exception as error:
            # W&B transport/GraphQL exception types vary by SDK version.
            last_error = error
        finally:
            # A failed paginator must not be reused, and successful full Run
            # objects are much larger than the compact result returned above.
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

    raise SchemaError(
        f"sweep {sweep_id}: fetch failed after {retries + 1} attempts: {last_error}"
    ) from last_error


def build_update(
    table: SchemaTable,
    ranked_runs: Sequence[RankedRun],
    fetched_runs: int,
    eligible_runs: int | None = None,
    csv_metric: str = DEFAULT_CSV_METRIC,
) -> PlannedUpdate:
    indexes = _column_indexes(table.headers, table.path)
    row_count = len(table.rows)
    eligible_count = len(ranked_runs) if eligible_runs is None else eligible_runs
    if eligible_count < row_count or len(ranked_runs) < row_count:
        raise SchemaError(
            f"{table.path}: needs {row_count} ranked runs, but only "
            f"{eligible_count} of {fetched_runs} fetched runs have a numeric {csv_metric}"
        )

    name_columns = {
        match.group(1): index
        for header, index in indexes.items()
        if (match := _PARAMETER_NAME_RE.fullmatch(header))
    }
    value_columns = {
        match.group(1): index
        for header, index in indexes.items()
        if (match := re.fullmatch(r"parameter_(\d+)_value", header))
    }
    if name_columns.keys() != value_columns.keys():
        missing_values = sorted(name_columns.keys() - value_columns.keys())
        missing_names = sorted(value_columns.keys() - name_columns.keys())
        details: list[str] = []
        if missing_values:
            details.append(f"missing value columns for parameters {missing_values}")
        if missing_names:
            details.append(f"missing name columns for parameters {missing_names}")
        raise SchemaError(f"{table.path}: {'; '.join(details)}")
    parameter_pairs = [
        (name_columns[number], value_columns[number])
        for number in sorted(name_columns, key=int)
    ]

    output_rows: list[list[str]] = []
    for rank, (source_row, run) in enumerate(
        zip(table.rows, ranked_runs[:row_count]), start=1
    ):
        row = source_row.copy()
        row[indexes["id"]] = str(rank)
        row[indexes[csv_metric]] = format_cell(run.metric_value)

        for name_index, value_index in parameter_pairs:
            parameter_name = row[name_index].strip()
            if not parameter_name:
                raise SchemaError(
                    f"{table.path}, rank {rank}: parameter name cannot be blank"
                )
            parameter_value = lookup_value(
                run.config,
                parameter_name,
                context=f"{table.path}, rank {rank}, run {run.run_id}",
            )
            formatted_value = format_cell(parameter_value)
            existing_value = row[value_index].strip()
            if existing_value and existing_value != formatted_value:
                raise SchemaError(
                    f"{table.path}, rank {rank}, run {run.run_id}: existing "
                    f"{table.headers[value_index]}={existing_value!r} does not match "
                    f"W&B value {formatted_value!r}; blank the cell before refreshing"
                )
            if not existing_value:
                row[value_index] = formatted_value

        output_rows.append(row)

    return PlannedUpdate(
        table=table,
        rows=output_rows,
        fetched_runs=fetched_runs,
        eligible_runs=eligible_count,
        selected_run_ids=[run.run_id for run in ranked_runs[:row_count]],
    )


def render_update(update: PlannedUpdate) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator=update.table.newline)
    writer.writerow(update.table.headers)
    writer.writerows(update.rows)
    return output.getvalue()


def write_atomic(path: Path, content: str) -> None:
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


def discover_csvs(arguments: Sequence[str], schemas_dir: Path) -> list[Path]:
    if not schemas_dir.is_dir():
        raise SchemaError(f"schemas directory does not exist: {schemas_dir}")
    if arguments:
        paths = [
            (Path(argument) if Path(argument).is_absolute() else schemas_dir / argument).resolve()
            for argument in arguments
        ]
    else:
        paths = sorted(path.resolve() for path in schemas_dir.rglob("*.csv"))

    if not paths:
        raise SchemaError(f"no CSV files found below {schemas_dir}")
    for path in paths:
        try:
            path.relative_to(schemas_dir)
        except ValueError as error:
            raise SchemaError(f"CSV must be inside {schemas_dir}: {path}") from error
        if path.suffix.casefold() != ".csv" or not path.is_file():
            raise SchemaError(f"not a CSV file: {path}")
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_files",
        nargs="*",
        metavar="CSV",
        help="CSV paths relative to --schemas-dir (default: every CSV recursively)",
    )
    parser.add_argument("--schemas-dir", type=Path, default=SCHEMAS_DIR)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--wandb-metric", default=DEFAULT_WANDB_METRIC)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and validate everything, but do not write any CSV",
    )
    args = parser.parse_args(argv)
    if args.page_size < 1:
        parser.error("--page-size must be positive")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    schemas_dir = args.schemas_dir.resolve()

    try:
        paths = discover_csvs(args.csv_files, schemas_dir)
        tables = [read_schema(path) for path in paths]

        try:
            import wandb
        except ImportError as error:
            raise SchemaError(
                "wandb is not installed; run this script with the repository's .venv"
            ) from error

        try:
            api = wandb.Api(timeout=args.timeout)
        except Exception as error:
            raise SchemaError(f"could not initialize the W&B API: {error}") from error

        table_sweeps = [(table, sweep_id_for(table)) for table in tables]
        required_by_sweep: dict[str, int] = {}
        for table, sweep_id in table_sweeps:
            required_by_sweep[sweep_id] = max(
                required_by_sweep.get(sweep_id, 0), len(table.rows)
            )

        sweep_cache: dict[str, SweepResult] = {}
        updates: list[PlannedUpdate] = []

        for table, sweep_id in table_sweeps:
            row_count = len(table.rows)
            print(
                f"Fetching sweep {sweep_id} for {table.path.relative_to(schemas_dir)} "
                f"({row_count} rows)...",
                flush=True,
            )
            if sweep_id not in sweep_cache:
                sweep_cache[sweep_id] = fetch_ranked_runs(
                    api=api,
                    entity=args.entity,
                    project=args.project,
                    sweep_id=sweep_id,
                    wandb_metric=args.wandb_metric,
                    page_size=args.page_size,
                    max_results=required_by_sweep[sweep_id],
                    retries=args.retries,
                )
            sweep_result = sweep_cache[sweep_id]
            update = build_update(
                table,
                sweep_result.ranked_runs,
                sweep_result.fetched_runs,
                sweep_result.eligible_runs,
            )
            updates.append(update)
            print(
                f"  selected {row_count} of {update.eligible_runs} ranked runs "
                f"({update.fetched_runs} total fetched)",
                flush=True,
            )

        rendered = [(update.table.path, render_update(update)) for update in updates]
        if args.dry_run:
            print(f"Dry run complete: validated {len(rendered)} CSV file(s); nothing written.")
        else:
            for path, content in rendered:
                write_atomic(path, content)
            print(f"Updated {len(rendered)} CSV file(s).")
        return 0
    except (SchemaError, OSError, csv.Error, UnicodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
