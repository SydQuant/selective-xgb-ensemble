#!/bin/bash

# XGBoost PROD Daily Signal Runner for AWS Linux EC2
# Standalone production deployment script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Environment for Xvfb/Wine (IQFeed)
export DISPLAY=:1
export WINEDEBUG=-all
export WINEPREFIX="${HOME}/.wine"

# Load environment variables from .env if present
ENV_LOADED=0
if [ -f .env ]; then
    # Export all variables except EMAIL_PASS (may contain spaces not shell-safe)
    # This prevents errors like: ".env: line 2: evho: command not found"
    # Safe-load by filtering the password line; Python will read it from .env directly.
    set -a
    # shellcheck disable=SC1090
    bash -c 'grep -v "^EMAIL_PASS=" ./.env' | sed '/^[[:space:]]*#/d' | sed 's/\r$//' > .env.shell.tmp
    # shellcheck disable=SC1091
    . ./.env.shell.tmp
    rm -f .env.shell.tmp
    set +a
    ENV_LOADED=1
fi

# Setup logging
LOG_DIR="logs/$(date +%Y%m%d)"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/xgb_prod_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

if [ "$ENV_LOADED" = "1" ]; then
    # log after LOG_FILE is ready
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Loaded .env from $SCRIPT_DIR" | tee -a "$LOG_FILE" >/dev/null 2>&1
fi

# Environment check
if ! command -v python3 &> /dev/null; then
    log "❌ Python3 not found"
    # Cleanup IQFeed/Xvfb on failure as well
    log "🧹 Cleaning up IQFeed/Xvfb after failure"
    pkill -f "IQCharts\.exe" || true
    pkill -f "IQConnect\.exe" || true
    if [ -n "${XVFB_PID}" ]; then
        kill ${XVFB_PID} 2>/dev/null || true
        pkill -f "Xvfb" || true
    fi
    rm -f /tmp/.X1-lock /tmp/.X11-unix/X1 || true

    exit 1
fi

# Configuration (override with environment variables)
DRY_RUN=${DRY_RUN:-0}
PYTHON_CMD=${PYTHON_CMD:-python3}

log "=== XGBoost Production Daily Signal Generation ==="
log "Working directory: $SCRIPT_DIR"
log "Python command: $PYTHON_CMD"
log "Log file: $LOG_FILE"

# --- IQFeed startup ---
log "🔧 Ensuring clean state for IQFeed/Xvfb"
pkill -f "IQCharts\.exe" || true
pkill -f "IQConnect\.exe" || true
pkill -f "Xvfb :1" || true
rm -f /tmp/.X1-lock /tmp/.X11-unix/X1 || true
sleep 2

log "🖥️  Starting Xvfb on $DISPLAY"
Xvfb :1 -screen 0 1024x768x16 >/dev/null 2>&1 &
XVFB_PID=$!
sleep 5
log "🪟 Starting IQFeed (IQCharts.exe) via wine"
if command -v wine >/dev/null 2>&1; then
  wine "C:\\Program Files\\DTN\\IQFeed\\IQCharts.exe" > "${LOG_DIR}/iqfeed_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
  IQFEED_PID=$!
  log "IQFeed PID: ${IQFEED_PID:-unknown}"
  log "⏳ Waiting for IQFeed to initialize..."
  sleep 15
else
  log "⚠️  wine not found; skipping IQFeed startup"
fi

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

    # Cleanup IQFeed/Xvfb
    log "🧹 Cleaning up IQFeed/Xvfb"
    pkill -f "IQCharts\.exe" || true
    pkill -f "IQConnect\.exe" || true
    if [ -n "${XVFB_PID}" ]; then
        kill ${XVFB_PID} 2>/dev/null || true
        pkill -f "Xvfb" || true
    fi
    rm -f /tmp/.X1-lock /tmp/.X11-unix/X1 || true

    exit 0
else
    log "❌ Signal generation failed"

    # Restore production mode if dry run
    if [ "$DRY_RUN" = "1" ]; then
        sed -i 's/dry_run = True/dry_run = False/' daily_signal_runner.py
    fi

    exit 1
fi