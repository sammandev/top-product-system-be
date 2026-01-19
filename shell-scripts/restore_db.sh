#!/bin/bash

#==============================================================================
# PostgreSQL Database Restore Script for Docker
#==============================================================================
# This script restores a PostgreSQL database from a backup file using Docker.
#
# Usage: ./restore_db.sh <backup_file> [docker-compose-file] [service-name]
# Example: ./restore_db.sh ../backups/backup_ast_tools_20260119_143000.sql.gz
# Example: ./restore_db.sh ../backups/backup_ast_tools_20260119_143000.sql.gz docker-compose.staging.yml backend
#==============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Check if backup file is provided
if [ -z "$1" ]; then
    echo "❌ Error: Backup file not specified"
    echo ""
    echo "Usage: $0 <backup_file> [docker-compose-file] [service-name]"
    echo "Example: $0 ./backups/backup_ast_tools_20260119_143000.sql.gz"
    echo ""
    
    # List available backups
    echo "📋 Available backups:"
    ls -lht "${PROJECT_ROOT}/backups"/backup_*.sql.gz 2>/dev/null | head -10 || echo "   No backups found"
    exit 1
fi

BACKUP_FILE="$1"
COMPOSE_FILE=${2:-docker-compose.staging.yml}
SERVICE_NAME=${3:-backend}

echo "🔄 Starting database restore..."
echo "========================================"
echo "Backup file: ${BACKUP_FILE}"
echo "Compose file: ${COMPOSE_FILE}"
echo "Service: ${SERVICE_NAME}"
echo ""

# Check if backup file exists
if [ ! -f "${BACKUP_FILE}" ]; then
    echo "❌ Error: Backup file not found: ${BACKUP_FILE}"
    exit 1
fi

echo "✅ Found backup file"

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

# Confirm restoration
echo ""
echo "⚠️  WARNING: This will REPLACE the current database!"
echo ""
echo "📦 Backup file details:"
BACKUP_SIZE=$(du -h "${BACKUP_FILE}" | cut -f1)
echo "   File: ${BACKUP_FILE}"
echo "   Size: ${BACKUP_SIZE}"
echo ""
read -p "Do you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Restore cancelled"
    exit 0
fi

echo ""
echo "💾 Restoring database..."

# Restore database
# First, uncompress the backup and pipe it to psql inside the container
if [[ "$BACKUP_FILE" == *.gz ]]; then
    echo "📦 Decompressing and restoring..."
    gunzip -c "${BACKUP_FILE}" | docker compose -f "$COMPOSE_FILE" exec -T "${SERVICE_NAME}" bash -c "
        PGPASSWORD='${DB_PASSWORD}' psql \
            -h '${DB_HOST}' \
            -p '${DB_PORT}' \
            -U '${DB_USER}' \
            -d '${DB_NAME}' \
            -v ON_ERROR_STOP=1
    "
else
    echo "📥 Restoring from uncompressed backup..."
    cat "${BACKUP_FILE}" | docker compose -f "$COMPOSE_FILE" exec -T "${SERVICE_NAME}" bash -c "
        PGPASSWORD='${DB_PASSWORD}' psql \
            -h '${DB_HOST}' \
            -p '${DB_PORT}' \
            -U '${DB_USER}' \
            -d '${DB_NAME}' \
            -v ON_ERROR_STOP=1
    "
fi

echo ""
echo "========================================"
echo "✨ Database restored successfully!"
echo ""
echo "🔄 You may want to run migrations to ensure schema is up to date:"
echo "   ./run-migrations-docker.sh ${COMPOSE_FILE} ${SERVICE_NAME}"
echo ""
echo "🔄 You may also want to restart the backend service:"
echo "   docker compose -f ${COMPOSE_FILE} restart ${SERVICE_NAME}"
echo ""
