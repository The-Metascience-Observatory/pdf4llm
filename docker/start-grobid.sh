#!/bin/bash
# Start GROBID Docker container with DeLFT support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-delft}"  # delft or crf

echo "Starting GROBID in $MODE mode..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Stop any existing GROBID containers
docker compose -f "$SCRIPT_DIR/docker-compose.yml" down 2>/dev/null || true

# Start GROBID with selected profile
docker compose -f "$SCRIPT_DIR/docker-compose.yml" --profile "$MODE" up -d

echo "Waiting for GROBID to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8070/api/isalive > /dev/null 2>&1; then
        echo "✓ GROBID is ready!"

        # Get version info
        VERSION=$(curl -s http://localhost:8070/api/version)
        echo "GROBID version: $VERSION"
        echo "Mode: $MODE"
        echo "URL: http://localhost:8070"
        exit 0
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "Error: GROBID failed to start within 120 seconds"
echo "Check logs with: docker compose -f $SCRIPT_DIR/docker-compose.yml logs"
exit 1
