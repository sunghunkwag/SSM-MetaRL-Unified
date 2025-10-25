#!/bin/bash
# RSI Daemon Control Script

DAEMON_SCRIPT="rsi_daemon.py"
PID_FILE="rsi_daemon.pid"
STOP_FILE="rsi_daemon.stop"
LOG_DIR="rsi_daemon_logs"

case "$1" in
    start)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p $PID > /dev/null 2>&1; then
                echo "❌ RSI Daemon is already running (PID: $PID)"
                exit 1
            else
                echo "⚠ Stale PID file found, removing..."
                rm "$PID_FILE"
            fi
        fi
        
        echo "🚀 Starting RSI Daemon..."
        nohup python3 "$DAEMON_SCRIPT" > /dev/null 2>&1 &
        sleep 2
        
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            echo "✅ RSI Daemon started successfully (PID: $PID)"
            echo "📁 Logs: $LOG_DIR/"
            echo "🛑 To stop: ./rsi_control.sh stop"
        else
            echo "❌ Failed to start RSI Daemon"
            exit 1
        fi
        ;;
        
    stop)
        if [ ! -f "$PID_FILE" ]; then
            echo "❌ RSI Daemon is not running"
            exit 1
        fi
        
        PID=$(cat "$PID_FILE")
        echo "🛑 Stopping RSI Daemon (PID: $PID)..."
        
        # Create stop signal file
        touch "$STOP_FILE"
        
        # Send SIGTERM
        kill -TERM $PID 2>/dev/null
        
        # Wait for graceful shutdown (max 30 seconds)
        for i in {1..30}; do
            if ! ps -p $PID > /dev/null 2>&1; then
                echo "✅ RSI Daemon stopped successfully"
                rm -f "$PID_FILE" "$STOP_FILE"
                exit 0
            fi
            sleep 1
        done
        
        # Force kill if still running
        echo "⚠ Daemon didn't stop gracefully, forcing..."
        kill -KILL $PID 2>/dev/null
        rm -f "$PID_FILE" "$STOP_FILE"
        echo "✅ RSI Daemon stopped (forced)"
        ;;
        
    status)
        if [ ! -f "$PID_FILE" ]; then
            echo "❌ RSI Daemon is not running"
            exit 1
        fi
        
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "✅ RSI Daemon is running (PID: $PID)"
            
            # Show latest log
            if [ -d "$LOG_DIR" ]; then
                LATEST_LOG=$(ls -t "$LOG_DIR"/rsi_daemon_*.log 2>/dev/null | head -1)
                if [ -n "$LATEST_LOG" ]; then
                    echo ""
                    echo "📊 Latest Status (last 20 lines):"
                    echo "=================================="
                    tail -20 "$LATEST_LOG"
                fi
            fi
        else
            echo "❌ RSI Daemon is not running (stale PID file)"
            rm "$PID_FILE"
            exit 1
        fi
        ;;
        
    logs)
        if [ ! -d "$LOG_DIR" ]; then
            echo "❌ No logs found"
            exit 1
        fi
        
        LATEST_LOG=$(ls -t "$LOG_DIR"/rsi_daemon_*.log 2>/dev/null | head -1)
        if [ -z "$LATEST_LOG" ]; then
            echo "❌ No log files found"
            exit 1
        fi
        
        echo "📄 Tailing log: $LATEST_LOG"
        echo "=================================="
        tail -f "$LATEST_LOG"
        ;;
        
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
        
    *)
        echo "Usage: $0 {start|stop|status|logs|restart}"
        echo ""
        echo "Commands:"
        echo "  start   - Start RSI daemon in background"
        echo "  stop    - Stop RSI daemon gracefully"
        echo "  status  - Check daemon status and show recent activity"
        echo "  logs    - Tail daemon logs in real-time"
        echo "  restart - Restart daemon"
        exit 1
        ;;
esac

exit 0

