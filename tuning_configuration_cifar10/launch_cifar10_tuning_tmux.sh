#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/wandb_sweeps_configs"

CONFIGS=(
    "01_tuning_cifar10_bresnet_l1.yaml"
    "02_tuning_cifar10_bresnet_l2.yaml"
    "03_tuning_cifar10_bresnet_scfe.yaml"
    "04_tuning_cifar10_bresnet_earlystop.yaml"
)

if ! command -v tmux >/dev/null 2>&1; then
    printf 'Error: tmux is not installed or is not in PATH.\n' >&2
    exit 1
fi

if [[ ! -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
    printf 'Error: virtual environment activation script not found: %s\n' \
        "$PROJECT_ROOT/.venv/bin/activate" >&2
    exit 1
fi

if [[ ! -f "$PROJECT_ROOT/main.py" ]]; then
    printf 'Error: entry point not found: %s\n' "$PROJECT_ROOT/main.py" >&2
    exit 1
fi

# Check every configuration before starting any sessions, avoiding a partial launch.
for config in "${CONFIGS[@]}"; do
    if [[ ! -f "$CONFIG_DIR/$config" ]]; then
        printf 'Error: configuration not found: %s\n' "$CONFIG_DIR/$config" >&2
        exit 1
    fi
done

for config in "${CONFIGS[@]}"; do
    config_name="${config%.yaml}"
    session_name="$config_name"

    if tmux has-session -t "$session_name" 2>/dev/null; then
        printf 'Skipping existing tmux session: %s\n' "$session_name"
        continue
    fi

    # Stagger W&B startup so runs are registered in configuration-list order.
    sleep 2

    tmux new-session \
        -d \
        -s "$session_name" \
        -c "$PROJECT_ROOT" \
        "bash -c 'source .venv/bin/activate && exec python main.py run_mode=sweep logger.config=$config_name'"

    printf 'Started tmux session %-40s (%s)\n' "$session_name" "$config"
done

printf '\nUse "tmux ls" to list sessions and "tmux attach -t <session-name>" to attach.\n'
