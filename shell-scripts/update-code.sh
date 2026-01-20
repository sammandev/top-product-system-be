#!/bin/bash

# Quick update script - restarts services to pick up code changes
# Use this when you only changed Python code (not requirements.txt or Dockerfile)
# Usage: ./update-code.sh [--skip-migrations]
#
# Options:
#   --skip-migrations : Skip running database migrations
#
# NOTE: Since source code is mounted as a volume with hot reload enabled,
#       many changes are picked up automatically. This script ensures all
#       processes reload the code by restarting the container.
#
# IMPORTANT: Run `git pull` manually before running this script.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

COMPOSE_FILE="docker-compose.staging.yml"
SERVICE_NAME="backend"
HEALTH_PORT="7070"
SKIP_MIGRATIONS=false

# Parse arguments
if [ "$1" = "--skip-migrations" ]; then
    SKIP_MIGRATIONS=true
fi

echo "🔄 Updating code without rebuild..."
echo "=================================================="
echo "Skip migrations: ${SKIP_MIGRATIONS}"
echo ""

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

# Check for uncommitted changes (informational)
if command -v git >/dev/null 2>&1 && [ -d ".git" ]; then
    echo ""
    if ! git diff-index --quiet HEAD -- 2>/dev/null; then
        echo "⚠️  You have uncommitted local changes:"
        git status --short
    else
        echo "✅ Git working directory is clean"
    fi
fi

# Restart container to pick up code changes
# Note: Using 'restart' is lightweight - no new images or containers created
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
echo "✅ Service restarted successfully!"

# Run migrations unless skipped
if [ "$SKIP_MIGRATIONS" = false ]; then
    echo ""
    echo "🔄 Running database migrations..."
    docker compose -f ${COMPOSE_FILE} exec -T ${SERVICE_NAME} alembic upgrade head
    echo "✅ Migrations complete!"
fi

echo ""
echo "✅ Code updated successfully!"
echo ""
echo "📝 Notes:"
echo "   - Service was restarted (no new images/containers created)"
echo "   - Source code is mounted as a volume with hot reload enabled"
echo "   - Most Python changes are picked up automatically by uvicorn --reload"
echo ""
echo "🔨 If you changed dependencies (pyproject.toml), run:"
echo "   ./deploy-staging.sh (to rebuild the image)"
echo ""
echo "📋 Useful commands:"
echo "   View logs: docker compose -f $COMPOSE_FILE logs -f $SERVICE_NAME"
echo "   Run migrations only: ./run-migrations-docker.sh"
echo "=================================================="
