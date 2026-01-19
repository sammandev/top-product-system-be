#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

COMPOSE_FILE=${1:-docker-compose.staging.yml}
SERVICE_NAME=${2:-backend}

# Run Alembic migrations inside the running container

docker compose -f "${COMPOSE_FILE}" exec "${SERVICE_NAME}" alembic upgrade head
