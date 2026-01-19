#!/usr/bin/env bash
set -euo pipefail

MESSAGE=${1:-"auto_migration"}

cd "$(dirname "$0")/.."

alembic revision --autogenerate -m "$MESSAGE"
