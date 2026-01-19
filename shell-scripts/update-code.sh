#!/bin/bash

# Quick update script - restarts services to pick up code changes
# Use this when you only changed Python code (not requirements.txt or Dockerfile)
# Usage: ./update-code.sh
#
# NOTE: Since source code is mounted as a volume with hot reload enabled,
#       many changes are picked up automatically. This script ensures all
#       processes reload the code by restarting the container.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

COMPOSE_FILE="docker-compose.staging.yml"
SERVICE_NAME="backend"
HEALTH_PORT="7070"

echo "🔄 Updating code without rebuild..."
echo "=================================================="

# Check if docker-compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "❌ Error: $COMPOSE_FILE not found!"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if containers are running
if ! docker compose -f ${COMPOSE_FILE} ps | grep -q "Up"; then
    echo "⚠️  Containers are not running. Use ./deploy-staging.sh instead."
    exit 1
fi

echo "✅ Containers are running"

# Pull latest code from Git (code is mounted as volume)
echo ""
echo "📥 Pulling latest code from Git..."
git pull --rebase || {
    echo "❌ Git pull failed. Please resolve conflicts and try again."
    exit 1
}

echo "✅ Code updated from Git"

# Restart backend service to reload code
# Note: With --reload flag, many changes are picked up automatically,
#       but restart ensures all modules are reloaded fresh
echo ""
echo "♻️  Restarting ${SERVICE_NAME} service..."
docker compose -f ${COMPOSE_FILE} restart ${SERVICE_NAME}

echo ""
echo "⏳ Waiting for service to be ready..."
sleep 3

# Check health
timeout 60 bash -c "until curl -f http://localhost:${HEALTH_PORT}/health > /dev/null 2>&1; do 
    echo -n '.'
    sleep 2
done" || {
    echo ""
    echo "⚠️  Health check timeout - checking logs..."
    docker compose -f ${COMPOSE_FILE} logs --tail=30 ${SERVICE_NAME}
    exit 1
}

echo ""
echo "✅ Code updated successfully!"
echo ""
echo "📝 Note: The ${SERVICE_NAME} service has been restarted."
echo "   Source code is mounted as a volume with hot reload enabled."
echo "   Most changes are picked up automatically by uvicorn --reload."
echo ""
echo "🔨 If you changed dependencies (pyproject.toml), run:"
echo "   ./deploy-staging.sh (to rebuild the image)"
echo ""
echo "=================================================="
