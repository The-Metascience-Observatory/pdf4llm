#!/bin/bash
# Comprehensive GROBID Docker management script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

command="${1:-}"

case "$command" in
    start)
        MODE="${2:-delft}"
        echo "Starting GROBID ($MODE mode)..."
        docker compose -f "$COMPOSE_FILE" --profile "$MODE" up -d
        ;;

    stop)
        echo "Stopping GROBID..."
        docker compose -f "$COMPOSE_FILE" down
        ;;

    restart)
        MODE="${2:-delft}"
        echo "Restarting GROBID ($MODE mode)..."
        docker compose -f "$COMPOSE_FILE" down
        docker compose -f "$COMPOSE_FILE" --profile "$MODE" up -d
        ;;

    logs)
        docker compose -f "$COMPOSE_FILE" logs -f
        ;;

    status)
        echo "Checking GROBID status..."
        if curl -s http://localhost:8070/api/isalive > /dev/null 2>&1; then
            VERSION=$(curl -s http://localhost:8070/api/version)
            echo "✓ GROBID is running"
            echo "  Version: $VERSION"
            echo "  URL: http://localhost:8070"
        else
            echo "✗ GROBID is not running"
        fi
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|logs|status} [delft|crf]"
        echo ""
        echo "Commands:"
        echo "  start [mode]   - Start GROBID (delft or crf mode)"
        echo "  stop           - Stop GROBID"
        echo "  restart [mode] - Restart GROBID"
        echo "  logs           - View GROBID logs"
        echo "  status         - Check if GROBID is running"
        exit 1
        ;;
esac
