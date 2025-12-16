#!/bin/bash

# Deployment script for AST Parser Backend
# Usage: ./deploy.sh [staging|production]

set -e

ENVIRONMENT=${1:-staging}
PROJECT_NAME="ast-tools"

echo "🚀 Starting deployment for ${ENVIRONMENT} environment..."

# Load environment variables
if [ -f ".env.${ENVIRONMENT}" ]; then
    export $(cat .env.${ENVIRONMENT} | grep -v '^#' | xargs)
    echo "✅ Environment variables loaded from .env.${ENVIRONMENT}"
else
    echo "⚠️  Warning: .env.${ENVIRONMENT} not found, using defaults"
fi

# Generate secure JWT secret if not set
if [ -z "$JWT_SECRET" ] || [ "$JWT_SECRET" = "CHANGE_THIS_TO_SECURE_RANDOM_STRING_IN_PRODUCTION" ]; then
    echo "⚠️  Generating secure JWT_SECRET..."
    JWT_SECRET=$(openssl rand -hex 32)
    echo "JWT_SECRET=${JWT_SECRET}" >> .env
    echo "✅ JWT_SECRET generated and saved to .env"
fi

# Build Docker images
echo "📦 Building Docker images..."
docker-compose build --no-cache

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for backend to be healthy
echo "⏳ Waiting for backend to be healthy..."
timeout 60 bash -c 'until curl -f http://localhost:7070/health > /dev/null 2>&1; do sleep 2; done' || {
    echo "❌ Backend failed to start"
    docker-compose logs backend
    exit 1
}

echo "✅ Backend is healthy"

# Run database migrations
echo "📊 Running database migrations..."
docker-compose exec -T backend alembic upgrade head || {
    echo "⚠️  Migration failed or no migrations to run"
}

# Check service status
echo "📊 Service status:"
docker-compose ps

echo "✅ Deployment complete!"
echo ""
echo "📝 Access points:"
echo "   - Frontend: http://172.18.220.56:9090"
echo "   - Backend API: http://172.18.220.56:9090/api"
echo "   - API Docs: http://172.18.220.56:9090/docs"
echo "   - Health Check: http://172.18.220.56:9090/health"
echo ""
echo "📊 Useful commands:"
echo "   - View logs: docker-compose logs -f"
echo "   - Stop services: docker-compose down"
echo "   - Restart: docker-compose restart"
echo "   - Shell access: docker-compose exec backend bash"
