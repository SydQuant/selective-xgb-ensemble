#!/bin/bash

# XGBoost PROD Daily Signal Runner for AWS Linux EC2
# Standalone production deployment script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Setup logging
LOG_DIR="logs/$(date +%Y%m%d)"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/xgb_prod_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Environment check
if ! command -v python3 &> /dev/null; then
    log "❌ Python3 not found"
    exit 1
fi

# Configuration (override with environment variables)
DRY_RUN=${DRY_RUN:-0}
PYTHON_CMD=${PYTHON_CMD:-python3}

log "=== XGBoost Production Daily Signal Generation ==="
log "Working directory: $SCRIPT_DIR"
log "Python command: $PYTHON_CMD"
log "Log file: $LOG_FILE"

if [ "$DRY_RUN" = "1" ]; then
    log "🧪 DRY RUN mode enabled"
    # For dry run, modify the script temporarily
    sed -i 's/dry_run = False/dry_run = True/' daily_signal_runner.py
fi

# Execute signal generation
log "🚀 Starting signal generation..."
if $PYTHON_CMD daily_signal_runner.py 2>&1 | tee -a "$LOG_FILE"; then
    log "✅ Signal generation completed successfully"

    # Restore production mode if dry run
    if [ "$DRY_RUN" = "1" ]; then
        sed -i 's/dry_run = True/dry_run = False/' daily_signal_runner.py
        log "🔄 Restored production mode"
    fi

    # Upload log file to S3 if configured
    if command -v aws &> /dev/null && [ -n "$AWS_S3_LOGS_BUCKET" ]; then
        aws s3 cp "$LOG_FILE" "s3://$AWS_S3_LOGS_BUCKET/$(basename "$LOG_FILE")"
        log "📤 Log uploaded to S3"
    fi

    exit 0
else
    log "❌ Signal generation failed"

    # Restore production mode if dry run
    if [ "$DRY_RUN" = "1" ]; then
        sed -i 's/dry_run = True/dry_run = False/' daily_signal_runner.py
    fi

    exit 1
fi