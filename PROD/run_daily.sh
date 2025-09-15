#!/bin/bash

# Simplified XGBoost Daily Runner
# Core functionality only

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/xgb_daily_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Configuration
SIGNAL_HOUR=${SIGNAL_HOUR:-12}
DRY_RUN=${DRY_RUN:-0}

log "=== XGBoost Daily Signal Generation ==="
log "Signal hour: $SIGNAL_HOUR"

# Build command
CMD="python xgb_daily_runner.py --signal-hour $SIGNAL_HOUR"
if [ "$DRY_RUN" = "1" ]; then
    CMD="$CMD --dry-run"
    log "DRY RUN mode"
fi

# Execute
log "Starting signal generation..."
if $CMD 2>&1 | tee -a "$LOG_FILE"; then
    log "✅ Signal generation completed"
    exit 0
else
    log "❌ Signal generation failed"
    exit 1
fi