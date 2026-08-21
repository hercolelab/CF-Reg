#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
CONFIG_DIR="$PROJECT_ROOT/wandb_sweeps_configs"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
MAIN_PY="$PROJECT_ROOT/main.py"

SESSION_NAME="${SESSION_NAME:-schema_grid_sweeps}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
if [[ "$LOG_DIR" != /* ]]; then
    LOG_DIR="$PROJECT_ROOT/$LOG_DIR"
fi

declare -a CSV_FILES=()
declare -a CONFIG_NAMES=()

usage() {
    cat <<EOF
Usage: $(basename -- "$0") [--check]

Without arguments, start all schema-derived grid sweeps sequentially in one
detached tmux session. The next sweep starts only after the current one exits.

Options:
  --check   Validate and print the CSV-to-YAML queue without starting tmux.
  -h, --help
            Show this help text.

Optional environment variables:
  SESSION_NAME  tmux session name (default: schema_grid_sweeps)
  LOG_DIR       log directory (default: schemas/logs)
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

timestamp() {
    date '+%Y-%m-%d %H:%M:%S %z'
}

collect_configurations() {
    local csv_file stem yaml_file
    local -A seen_stems=()

    CSV_FILES=()
    CONFIG_NAMES=()

    while IFS= read -r -d '' csv_file; do
        stem="$(basename -- "${csv_file%.csv}")"

        if [[ -n "${seen_stems[$stem]+present}" ]]; then
            die "duplicate CSV basename '$stem' in '$csv_file' and '${seen_stems[$stem]}'"
        fi
        seen_stems["$stem"]="$csv_file"

        yaml_file="$CONFIG_DIR/$stem.yaml"
        [[ -f "$yaml_file" ]] || die "missing YAML for '$csv_file': $yaml_file"

        CSV_FILES+=("$csv_file")
        CONFIG_NAMES+=("$stem")
    done < <(
        find "$SCRIPT_DIR" -mindepth 2 -type f -name '*.csv' -print0 \
            | LC_ALL=C sort -z
    )

    ((${#CONFIG_NAMES[@]} > 0)) || die "no CSV files found below $SCRIPT_DIR"
}

validate_environment() {
    [[ -f "$MAIN_PY" ]] || die "main.py not found at $MAIN_PY"
    [[ -x "$PYTHON_BIN" ]] || die "virtualenv Python is not executable: $PYTHON_BIN"
    [[ -d "$CONFIG_DIR" ]] || die "configuration directory not found: $CONFIG_DIR"

    collect_configurations
}

print_queue() {
    local index

    printf 'Validated %d CSV/YAML pair(s):\n' "${#CONFIG_NAMES[@]}"
    for index in "${!CONFIG_NAMES[@]}"; do
        printf '  %02d  %s\n' "$((index + 1))" "${CONFIG_NAMES[$index]}"
    done
}

run_worker() {
    local log_file="$1"
    local total index config_name exit_code
    local -a failures=()

    [[ -n "$log_file" ]] || die "worker log path is empty"
    [[ -d "$(dirname -- "$log_file")" ]] || die "worker log directory does not exist"
    command -v tee >/dev/null 2>&1 || die "tee is required but was not found in PATH"

    # Preserve both stdout and stderr in the log while keeping live tmux output.
    exec > >(tee -a -- "$log_file") 2>&1

    validate_environment
    cd -- "$PROJECT_ROOT"
    export PYTHONUNBUFFERED=1

    total="${#CONFIG_NAMES[@]}"
    printf '[%s] Starting %d sequential grid sweep(s).\n' "$(timestamp)" "$total"

    for index in "${!CONFIG_NAMES[@]}"; do
        config_name="${CONFIG_NAMES[$index]}"
        printf '\n[%s] [%d/%d] Starting %s\n' \
            "$(timestamp)" "$((index + 1))" "$total" "$config_name"

        if "$PYTHON_BIN" -u "$MAIN_PY" \
            run_mode=sweep "logger.config=$config_name"; then
            printf '[%s] [%d/%d] Finished %s\n' \
                "$(timestamp)" "$((index + 1))" "$total" "$config_name"
        else
            exit_code=$?
            failures+=("$config_name (exit $exit_code)")
            printf '[%s] [%d/%d] FAILED %s (exit %d); continuing with the next sweep.\n' \
                "$(timestamp)" "$((index + 1))" "$total" "$config_name" "$exit_code" >&2
        fi
    done

    if ((${#failures[@]} > 0)); then
        printf '\n[%s] Queue completed with %d failed sweep(s):\n' \
            "$(timestamp)" "${#failures[@]}" >&2
        printf '  - %s\n' "${failures[@]}" >&2
        return 1
    fi

    printf '\n[%s] All %d grid sweeps completed successfully.\n' "$(timestamp)" "$total"
}

launch_tmux() {
    local log_file worker_command pane_id

    command -v tmux >/dev/null 2>&1 || die "tmux is required but was not found in PATH"
    command -v tee >/dev/null 2>&1 || die "tee is required but was not found in PATH"
    [[ "$SESSION_NAME" =~ ^[A-Za-z0-9_-]+$ ]] \
        || die "SESSION_NAME may contain only letters, numbers, underscores, and hyphens"

    if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
        die "tmux session '$SESSION_NAME' already exists; attach with: tmux attach-session -t $SESSION_NAME"
    fi

    mkdir -p -- "$LOG_DIR"
    log_file="$LOG_DIR/schema_grid_sweeps_$(date '+%Y%m%d_%H%M%S')_$$.log"
    : > "$log_file"

    printf -v worker_command 'exec bash %q --worker %q' "$SCRIPT_PATH" "$log_file"

    # Configure the pane before starting the worker so even an immediate worker
    # failure remains visible in tmux.
    pane_id="$(
        tmux new-session -d -P -F '#{pane_id}' \
            -s "$SESSION_NAME" -c "$PROJECT_ROOT"
    )"
    tmux set-option -w -t "$pane_id" remain-on-exit on >/dev/null
    if ! tmux respawn-pane -k -t "$pane_id" -c "$PROJECT_ROOT" "$worker_command"; then
        tmux kill-session -t "=$SESSION_NAME" 2>/dev/null || true
        die "could not start the sweep worker in tmux session '$SESSION_NAME'"
    fi

    printf 'Started %d sequential grid sweep(s) in tmux session %q.\n' \
        "${#CONFIG_NAMES[@]}" "$SESSION_NAME"
    printf 'Attach: tmux attach-session -t %q\n' "$SESSION_NAME"
    printf 'Log:    tail -f %q\n' "$log_file"
    printf 'After completion, remove the retained session with:\n'
    printf '        tmux kill-session -t %q\n' "$SESSION_NAME"
}

main() {
    case "${1:-}" in
        '')
            validate_environment
            launch_tmux
            ;;
        --check)
            [[ $# -eq 1 ]] || die "--check takes no arguments"
            validate_environment
            print_queue
            ;;
        --worker)
            [[ $# -eq 2 ]] || die "internal --worker mode requires exactly one log path"
            run_worker "$2"
            ;;
        -h|--help)
            [[ $# -eq 1 ]] || die "$1 takes no arguments"
            usage
            ;;
        *)
            usage >&2
            die "unknown argument: $1"
            ;;
    esac
}

main "$@"
