#!/bin/bash

#==============================================================================
# Backend Staging Deployment Script
#==============================================================================
# This script performs a complete deployment of the backend to staging.
# It includes pre-flight checks, builds the image, and starts services.
#
# Usage: ./deploy-staging.sh [--no-build]
#   --no-build : Skip Docker image rebuild (just start/restart services)
#
# When to use this script:
#   - Initial deployment
#   - After changing dependencies (pyproject.toml)
#   - After changing Dockerfile
#
# For code-only changes, use ./update-code.sh instead (faster, no rebuild)
#==============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

COMPOSE_FILE="docker-compose.staging.yml"
SERVICE_NAME="backend"
HEALTH_PORT="7070"
DO_BUILD=true

# Parse arguments
if [ "$1" = "--no-build" ]; then
    DO_BUILD=false
fi

echo "🚀 Starting backend staging deployment..."
echo "========================================"
echo "Compose file: ${COMPOSE_FILE}"
echo "Service: ${SERVICE_NAME}"
echo "Build image: ${DO_BUILD}"
echo ""

# Pre-flight checks
echo "🔍 Running pre-flight checks..."
echo ""

# Check if docker-compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "❌ Error: $COMPOSE_FILE not found!"
    exit 1
fi
echo "✅ Found compose file"

# Check if .env file exists
ENV_FILE=".env.staging"
if [ -f "$ENV_FILE" ]; then
    echo "✅ Found environment file: ${ENV_FILE}"
else
    echo "❌ Error: ${ENV_FILE} not found!"
    echo "Please create the environment file before deploying."
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Error: Docker is not running. Please start Docker first."
    exit 1
fi
echo "✅ Docker is running"

# Check if Git is clean (warn if dirty)
if command -v git >/dev/null 2>&1; then
    if [ -d ".git" ]; then
        if ! git diff-index --quiet HEAD -- 2>/dev/null; then
            echo "⚠️  Warning: You have uncommitted changes"
            git status --short
            echo ""
        else
            echo "✅ Git working directory is clean"
        fi
    fi
fi

echo ""
echo "========================================"

# Build and start services
if [ "$DO_BUILD" = true ]; then
    echo "📦 Building and starting services..."
    docker compose -f "$COMPOSE_FILE" up -d --build "$SERVICE_NAME"
else
    echo "📦 Starting services (no rebuild)..."
    docker compose -f "$COMPOSE_FILE" up -d "$SERVICE_NAME"
fi

echo ""
echo "========================================"
echo "⏳ Waiting for backend to be ready..."
echo ""

# Wait for service to be healthy
RETRY_COUNT=0
MAX_RETRIES=30

until curl -fsS "http://localhost:${HEALTH_PORT}/health" > /dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo ""
        echo "❌ Health check timeout after ${MAX_RETRIES} attempts"
        echo ""
        echo "📋 Recent container logs:"
        docker compose -f "$COMPOSE_FILE" logs --tail=50 "$SERVICE_NAME"
        exit 1
    fi
    
    echo -n "."
    sleep 2
done

echo ""
echo "✅ Backend is healthy!"

echo ""
echo "========================================"
echo "✨ Deployment complete!"
echo ""
echo "📊 Service status:"
docker compose -f "$COMPOSE_FILE" ps "$SERVICE_NAME"

echo ""
echo "🔗 Service endpoints:"
echo "   Health: http://localhost:${HEALTH_PORT}/health"
echo "   API Docs: http://localhost:${HEALTH_PORT}/docs"
echo ""
echo "📋 Useful commands:"
echo "   View logs: docker compose -f $COMPOSE_FILE logs -f $SERVICE_NAME"
echo "   Update code: ./update-code.sh"
echo "   Update env: ./refresh-env.sh"
echo "   Run migrations: ./run-migrations-docker.sh"
echo ""
