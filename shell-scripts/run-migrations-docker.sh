#!/bin/bash
# run-migrations-docker.sh - Apply Alembic migrations inside Docker container
# Usage: ./run-migrations-docker.sh [docker-compose-file] [service-name]
# Example: ./run-migrations-docker.sh docker-compose.staging.yml backend
# Default: docker-compose.staging.yml backend

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Parse arguments
COMPOSE_FILE=${1:-docker-compose.staging.yml}
SERVICE_NAME=${2:-backend}

echo "🔄 Starting database migration process..."
echo "========================================"
echo "Compose file: ${COMPOSE_FILE}"
echo "Service: ${SERVICE_NAME}"
echo ""

# Check if docker-compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "❌ Error: $COMPOSE_FILE not found!"
    echo "Available files:"
    ls -1 docker-compose*.yml 2>/dev/null || echo "No docker-compose files found"
    exit 1
fi

echo "✅ Found compose file: ${COMPOSE_FILE}"

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Error: Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"

# Check if container is running
if ! docker compose -f "$COMPOSE_FILE" ps | grep -q "${SERVICE_NAME}.*Up"; then
    echo "❌ Error: ${SERVICE_NAME} container is not running."
    echo "Start it with: docker compose -f $COMPOSE_FILE up -d ${SERVICE_NAME}"
    exit 1
fi

echo "✅ Container '${SERVICE_NAME}' is running"

# Show current migration status
echo ""
echo "📋 Current migration status:"
docker compose -f "$COMPOSE_FILE" exec -T "${SERVICE_NAME}" alembic current

# Check for unapplied migrations
echo ""
echo "🔍 Checking for unapplied migrations..."
CURRENT_REV=$(docker compose -f "$COMPOSE_FILE" exec -T "${SERVICE_NAME}" alembic current | grep -oP '(?<=^)[a-f0-9]+' || echo "none")
HEAD_REV=$(docker compose -f "$COMPOSE_FILE" exec -T "${SERVICE_NAME}" alembic heads | grep -oP '(?<=^)[a-f0-9]+' || echo "none")

if [ "$CURRENT_REV" = "$HEAD_REV" ] && [ "$CURRENT_REV" != "none" ]; then
    echo "✅ All migrations are already applied. Database is up to date."
    exit 0
fi

echo "⚠️  Found unapplied migrations"
echo "   Current: ${CURRENT_REV}"
echo "   Target: ${HEAD_REV}"
echo ""

# Run migrations
echo "🚀 Applying migrations..."
docker compose -f "$COMPOSE_FILE" exec -T "${SERVICE_NAME}" alembic upgrade head

# Verify migrations were applied
echo ""
echo "✅ Migration complete! Current status:"
docker compose -f "$COMPOSE_FILE" exec -T "${SERVICE_NAME}" alembic current

echo ""
echo "========================================"
echo "✨ Database migrations applied successfully!"
echo ""
