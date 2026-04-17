#!/bin/bash
# Hermes Self-Evolution Cron Runner
# Runs the actual Python evolution script and post-processes output to wiki.
# Called by the hermes-agent cron scheduler.
set -euo pipefail

VENV="/home/morderith/Corvus/AutoBot/hermes-agent/.venv"
EVOLUTION_DIR="/home/morderith/Corvus/self-evolution"
SCRIPTS_DIR="/home/morderith/Corvus/scripts"
LOG_DIR="/home/morderith/Corvus/logs"
WIKI_DIR="/home/morderith/wiki/queries"
TODAY=$(date +%Y-%m-%d)

# ── Environment ──────────────────────────────────────────────────────────────

source "$VENV/bin/activate"

# Read MiniMax API key from mmx config
API_KEY=$(python3 -c "import json; print(json.load(open('$HOME/.mmx/config.json'))['api_key'])" 2>/dev/null || echo "")
if [[ -z "$API_KEY" ]]; then
    echo "ERROR: No MiniMax API key found in ~/.mmx/config.json"
    exit 1
fi
export MINIMAX_API_KEY="$API_KEY"
export MINIMAX_BASE_URL="https://api.minimax.io/v1"
export HERMES_AGENT_REPO="/home/morderith/Corvus/AutoBot/hermes-agent"

# ── Run the evolution ────────────────────────────────────────────────────────

mkdir -p "$LOG_DIR"

SKILL="${1:-github-code-review}"
ITERATIONS="${2:-10}"
RUN_LOG="$LOG_DIR/self-evolution-$TODAY.log"

echo "[$(date)] Starting self-evolution for skill=$SKILL iterations=$ITERATIONS" >> "$RUN_LOG"

cd "$EVOLUTION_DIR"
python run-self-evolution.py "$SKILL" \
    --iterations "$ITERATIONS" \
    --eval-source synthetic \
    >> "$RUN_LOG" 2>&1

EXIT_CODE=$?

echo "[$(date)] Evolution finished with exit_code=$EXIT_CODE" >> "$RUN_LOG"

# ── Post-process: write summary to wiki ─────────────────────────────────────

EVOLUTION_SCRIPTS_DIR="$EVOLUTION_DIR/scripts"
python "$EVOLUTION_SCRIPTS_DIR/post-process-evolution.py" "$TODAY" "$RUN_LOG" && echo "[$(date)] Wiki summary written" >> "$RUN_LOG" || echo "[$(date)] Wiki summary FAILED" >> "$RUN_LOG"

exit $EXIT_CODE
