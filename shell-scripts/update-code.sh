#!/bin/bash

# Backend code update (git pull + rebuild)
# Usage: ./update-code.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "🔄 Updating backend code..."

git pull --rebase

docker compose -f docker-compose.staging.yml up -d --build backend

echo "⏳ Waiting for backend health..."
if command -v curl >/dev/null 2>&1; then
  timeout 60 bash -c 'until curl -fsS http://localhost:7070/health > /dev/null 2>&1; do sleep 2; done'
else
  echo "⚠️ curl not found; skipping health wait"
fi

echo "✅ Backend update complete"
