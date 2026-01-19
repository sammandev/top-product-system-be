#!/bin/bash
# refresh-env.sh - Apply environment variable changes
# Usage: ./refresh-env.sh
# Note: Edit .env.staging file first, then run this script
#
# This script restarts the backend service to pick up new environment variables.
# It does NOT rebuild the Docker image.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

COMPOSE_FILE="docker-compose.staging.yml"
SERVICE_NAME="backend"
ENV_FILE=".env.staging"
HEALTH_PORT="7070"

echo "🔄 Applying environment variable changes..."
echo "=================================================="

# Check if docker-compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "❌ Error: $COMPOSE_FILE not found!"
    exit 1
fi

# Check if .env.staging exists
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: $ENV_FILE file not found!"
    exit 1
fi

echo "✅ Found $ENV_FILE file"

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

# Restart services to pick up new environment variables
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
echo "✅ Environment variables updated!"

# Verify key settings
echo ""
echo "📊 Current environment settings:"
echo "-----------------------------------"
echo "DB_HOST: $(docker compose -f ${COMPOSE_FILE} exec -T ${SERVICE_NAME} printenv DB_HOST 2>/dev/null || echo 'Not set')"
echo "DB_NAME: $(docker compose -f ${COMPOSE_FILE} exec -T ${SERVICE_NAME} printenv DB_NAME 2>/dev/null || echo 'Not set')"
echo "REDIS_HOST: $(docker compose -f ${COMPOSE_FILE} exec -T ${SERVICE_NAME} printenv REDIS_HOST 2>/dev/null || echo 'Not set')"
echo "LOG_LEVEL: $(docker compose -f ${COMPOSE_FILE} exec -T ${SERVICE_NAME} printenv LOG_LEVEL 2>/dev/null || echo 'Not set')"
echo "=================================================="
