#!/bin/bash
# Check if IQFeed is ready to accept connections

HOST="localhost"
PORT=5009  # Default IQFeed port
TIMEOUT=300  # 5 minutes max
START_TIME=$(date +%s)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_iqfeed_processes() {
    # Check for both IQFeed processes
    local charts_running=0
    local connect_running=0
    
    if pgrep -f "IQCharts.exe" >/dev/null; then
        charts_running=1
    fi
    
    if pgrep -f "IQConnect.exe" >/dev/null; then
        connect_running=1
    fi
    
    log "IQCharts.exe running: ${charts_running}"
    log "IQConnect.exe running: ${connect_running}"
    
    if [ $charts_running -eq 1 ] && [ $connect_running -eq 1 ]; then
        return 0  # Both processes are running
    else
        return 1  # One or both processes are not running
    fi
}

check_port() {
    # Check if port is open using netcat
    if command -v nc &> /dev/null; then
        nc -z $HOST $PORT >/dev/null 2>&1
        return $?
    else
        # Fallback to checking if port is in use by any process
        lsof -i :$PORT >/dev/null 2>&1
        return $?
    fi
}

log "=== Starting IQFeed readiness check ==="
log "Host: $HOST, Port: $PORT, Timeout: ${TIMEOUT}s"

while :; do
    # Check if IQFeed processes are running
    if ! check_iqfeed_processes; then
        log "IQFeed processes not fully started yet..."
    else
        log "IQFeed processes are running"
        
        # Check if port is open
        if check_port; then
            log "IQFeed is accepting connections on port $PORT"
            break
        else
            log "Port $PORT is not yet accepting connections"
        fi
    fi
    
    # Check timeout
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    
    if [ $ELAPSED -gt $TIMEOUT ]; then
        log "ERROR: Timeout waiting for IQFeed to be ready after ${TIMEOUT}s"
        exit 1
    fi
    
    # Show progress every 10 seconds
    if [ $((ELAPSED % 10)) -eq 0 ]; then
        log "Still waiting for IQFeed... (${ELAPSED}s elapsed)"
    fi
    
    sleep 5
done

# Additional verification
log "Verifying IQFeed connection..."
if ! check_port; then
    log "WARNING: Port $PORT is no longer available after initial check"
    exit 1
fi

# Additional delay to ensure IQFeed is fully initialized
log "Giving IQFeed additional time to initialize..."
sleep 15

# Final check
if ! check_port; then
    log "ERROR: IQFeed stopped responding during initialization"
    exit 1
fi

log "=== IQFeed is ready ==="
exit 0
