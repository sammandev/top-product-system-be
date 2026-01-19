#!/bin/bash

#==============================================================================
# PostgreSQL Database Backup Script for Docker
#==============================================================================
# This script creates a backup of the PostgreSQL database from Docker container
# with timestamp and stores it in the backups directory.
#
# Usage: ./backup_db.sh [docker-compose-file] [service-name]
# Example: ./backup_db.sh docker-compose.staging.yml backend
# Default: docker-compose.staging.yml backend
#==============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Parse arguments
COMPOSE_FILE=${1:-docker-compose.staging.yml}
SERVICE_NAME=${2:-backend}

# Configuration
BACKUP_DIR="${PROJECT_ROOT}/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "🔄 Starting database backup..."
echo "========================================"
echo "Compose file: ${COMPOSE_FILE}"
echo "Service: ${SERVICE_NAME}"
echo ""

# Check if docker-compose file exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "❌ Error: $COMPOSE_FILE not found!"
    exit 1
fi

echo "✅ Found compose file"

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

# Create backup directory if it doesn't exist
mkdir -p "${BACKUP_DIR}"
echo "✅ Backup directory ready: ${BACKUP_DIR}"

# Get database connection info from container environment
echo ""
echo "📊 Reading database configuration..."
DB_HOST=$(docker compose -f "$COMPOSE_FILE" exec -T "${SERVICE_NAME}" printenv DB_HOST || echo "localhost")
DB_PORT=$(docker compose -f "$COMPOSE_FILE" exec -T "${SERVICE_NAME}" printenv DB_PORT || echo "5432")
DB_NAME=$(docker compose -f "$COMPOSE_FILE" exec -T "${SERVICE_NAME}" printenv DB_NAME || echo "")
DB_USER=$(docker compose -f "$COMPOSE_FILE" exec -T "${SERVICE_NAME}" printenv DB_USER || echo "postgres")
DB_PASSWORD=$(docker compose -f "$COMPOSE_FILE" exec -T "${SERVICE_NAME}" printenv DB_PASSWORD || echo "")

# Remove any carriage returns/newlines
DB_HOST=$(echo "$DB_HOST" | tr -d '\r\n')
DB_PORT=$(echo "$DB_PORT" | tr -d '\r\n')
DB_NAME=$(echo "$DB_NAME" | tr -d '\r\n')
DB_USER=$(echo "$DB_USER" | tr -d '\r\n')
DB_PASSWORD=$(echo "$DB_PASSWORD" | tr -d '\r\n')

if [ -z "$DB_NAME" ]; then
    echo "❌ Error: Could not determine database name from container environment"
    echo "Please check that DB_NAME is set in your environment file"
    exit 1
fi

echo "Database: ${DB_NAME}"
echo "Host: ${DB_HOST}:${DB_PORT}"
echo "User: ${DB_USER}"

# Define backup files
BACKUP_FILE="${BACKUP_DIR}/backup_${DB_NAME}_${TIMESTAMP}.sql"
COMPRESSED_FILE="${BACKUP_FILE}.gz"

echo ""
echo "💾 Creating backup..."
echo "Backup file: ${COMPRESSED_FILE}"

# Create backup using pg_dump inside the Docker container
# Export to the container, then copy to host
docker compose -f "$COMPOSE_FILE" exec -T "${SERVICE_NAME}" bash -c "
    PGPASSWORD='${DB_PASSWORD}' pg_dump \
        -h '${DB_HOST}' \
        -p '${DB_PORT}' \
        -U '${DB_USER}' \
        -d '${DB_NAME}' \
        -F p \
        --no-owner \
        --no-acl \
        --clean \
        --if-exists
" | gzip > "${COMPRESSED_FILE}"

# Check if backup was created successfully
if [ ! -f "${COMPRESSED_FILE}" ]; then
    echo "❌ Error: Backup file was not created"
    exit 1
fi

# Get backup file size
BACKUP_SIZE=$(du -h "${COMPRESSED_FILE}" | cut -f1)

echo ""
echo "========================================"
echo "✨ Backup created successfully!"
echo ""
echo "📁 Backup details:"
echo "   File: ${COMPRESSED_FILE}"
echo "   Size: ${BACKUP_SIZE}"
echo "   Timestamp: ${TIMESTAMP}"
echo ""
echo "🔍 To restore this backup, run:"
echo "   ./restore_db.sh ${COMPRESSED_FILE}"
echo ""

# List recent backups
echo "📋 Recent backups:"
ls -lht "${BACKUP_DIR}"/backup_*.sql.gz 2>/dev/null | head -5 || echo "   No other backups found"
echo ""
