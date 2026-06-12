#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/innov-xc/Projects/lhy/data-agents"
TEMPLATE_CONFIG_PATH="$PROJECT_ROOT/configs/psrr_baseline.example.yaml"

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

INPUT_DIR="${INPUT_DIR:-/input}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
LOG_DIR="${LOG_DIR:-/logs}"
MODEL_API_URL="${MODEL_API_URL:-}"
MODEL_API_KEY="${MODEL_API_KEY:-}"
MODEL_NAME="${MODEL_NAME:-}"
AGENT_MODE="${AGENT_MODE:-psrr}"
MAX_STEPS="${MAX_STEPS:-16}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_WORKERS="${MAX_WORKERS:-$(python - <<'PY'
import os
count = os.cpu_count() or 1
print(min(max(count, 1), 16))
PY
)}"
TASK_TIMEOUT_SECONDS="${TASK_TIMEOUT_SECONDS:-900}"
ENABLE_REFLEXION="${ENABLE_REFLEXION:-true}"
REFLEXION_RETRY_ON_FAILURE="${REFLEXION_RETRY_ON_FAILURE:-true}"
REFLEXION_MAX_RETRIES="${REFLEXION_MAX_RETRIES:-1}"
MEMORY_TOP_K="${MEMORY_TOP_K:-3}"
MEMORY_MAX_ITEMS="${MEMORY_MAX_ITEMS:-2000}"
RUN_ID="${RUN_ID:-submission}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" /tmp/data-agent

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory not found: $INPUT_DIR" | tee -a "$LOG_DIR/runtime.log"
  exit 1
fi

if [[ -z "$MODEL_API_URL" || -z "$MODEL_API_KEY" || -z "$MODEL_NAME" ]]; then
  echo "MODEL_API_URL, MODEL_API_KEY, and MODEL_NAME must all be set." | tee -a "$LOG_DIR/runtime.log"
  exit 1
fi

if [[ ! -f "$TEMPLATE_CONFIG_PATH" ]]; then
  echo "Template config not found: $TEMPLATE_CONFIG_PATH" | tee -a "$LOG_DIR/runtime.log"
  exit 1
fi

INTERNAL_OUTPUT_DIR="/tmp/data-agent/runs"
INTERNAL_MEMORY_PATH="/tmp/data-agent/memory.jsonl"
CONFIG_PATH="/tmp/data-agent/config.yaml"

export TEMPLATE_CONFIG_PATH
export INPUT_DIR
export OUTPUT_DIR
export LOG_DIR
export MODEL_API_URL
export MODEL_API_KEY
export MODEL_NAME
export AGENT_MODE
export MAX_STEPS
export TEMPERATURE
export MAX_WORKERS
export TASK_TIMEOUT_SECONDS
export ENABLE_REFLEXION
export REFLEXION_RETRY_ON_FAILURE
export REFLEXION_MAX_RETRIES
export MEMORY_TOP_K
export MEMORY_MAX_ITEMS
export RUN_ID
export INTERNAL_OUTPUT_DIR
export INTERNAL_MEMORY_PATH
export CONFIG_PATH

python - <<'PY'
from pathlib import Path
import os

import yaml

template_path = Path(os.environ["TEMPLATE_CONFIG_PATH"])
config_path = Path(os.environ["CONFIG_PATH"])

payload = yaml.safe_load(template_path.read_text()) or {}
payload.setdefault("dataset", {})["root_path"] = os.environ["INPUT_DIR"]

agent = payload.setdefault("agent", {})
agent["mode"] = os.environ["AGENT_MODE"]
agent["model"] = os.environ["MODEL_NAME"]
agent["api_base"] = os.environ["MODEL_API_URL"]
agent["api_key"] = os.environ["MODEL_API_KEY"]
agent["max_steps"] = int(os.environ["MAX_STEPS"])
agent["temperature"] = float(os.environ["TEMPERATURE"])
agent["enable_reflexion"] = os.environ["ENABLE_REFLEXION"].lower() == "true"
agent["reflexion_retry_on_failure"] = os.environ["REFLEXION_RETRY_ON_FAILURE"].lower() == "true"
agent["reflexion_max_retries"] = int(os.environ["REFLEXION_MAX_RETRIES"])
agent["memory_path"] = os.environ["INTERNAL_MEMORY_PATH"]
agent["memory_top_k"] = int(os.environ["MEMORY_TOP_K"])
agent["memory_max_items"] = int(os.environ["MEMORY_MAX_ITEMS"])

run = payload.setdefault("run", {})
run["output_dir"] = os.environ["INTERNAL_OUTPUT_DIR"]
run["run_id"] = os.environ["RUN_ID"]
run["max_workers"] = int(os.environ["MAX_WORKERS"])
run["task_timeout_seconds"] = int(os.environ["TASK_TIMEOUT_SECONDS"])

config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
PY

echo "Using config template: $TEMPLATE_CONFIG_PATH" | tee -a "$LOG_DIR/runtime.log"
echo "Effective runtime config:" | tee -a "$LOG_DIR/runtime.log"
cat "$CONFIG_PATH" | tee -a "$LOG_DIR/runtime.log"
echo "Starting benchmark with input=$INPUT_DIR output=$OUTPUT_DIR workers=$MAX_WORKERS" | tee -a "$LOG_DIR/runtime.log"

if ! dabench run-benchmark --config "$CONFIG_PATH" 2>&1 | tee -a "$LOG_DIR/runtime.log"; then
  exit_code=${PIPESTATUS[0]}
  echo "Benchmark failed with exit code $exit_code" | tee -a "$LOG_DIR/runtime.log"
  exit "$exit_code"
fi

RUN_OUTPUT_DIR="$INTERNAL_OUTPUT_DIR/$RUN_ID"
if [[ ! -d "$RUN_OUTPUT_DIR" ]]; then
  echo "Expected run output directory missing: $RUN_OUTPUT_DIR" | tee -a "$LOG_DIR/runtime.log"
  exit 1
fi

find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
cp -a "$RUN_OUTPUT_DIR"/. "$OUTPUT_DIR"/

echo "Copied predictions to $OUTPUT_DIR" | tee -a "$LOG_DIR/runtime.log"