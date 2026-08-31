"""Add encrypted PrimeUI license to application configuration.

Revision ID: 20260831a
Revises: 20260730a
Create Date: 2026-08-31
"""

import sqlalchemy as sa

from alembic import op

revision = "20260831a"
down_revision = "20260730a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("app_config", sa.Column("primeui_license_key", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("app_config", "primeui_license_key")
