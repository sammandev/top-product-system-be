"""Add admin and guest values to userrole enum

Revision ID: 20260223c
Revises: 20260223b
Create Date: 2026-02-23
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260223c"
down_revision = "20260223b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # PostgreSQL enums require explicit ALTER TYPE to add new values.
    # IF NOT EXISTS prevents errors when re-running on an already-upgraded DB.
    op.execute("ALTER TYPE userrole ADD VALUE IF NOT EXISTS 'admin'")
    op.execute("ALTER TYPE userrole ADD VALUE IF NOT EXISTS 'guest'")


def downgrade() -> None:
    # PostgreSQL does not support removing values from an enum type directly.
    # A full downgrade would require recreating the enum, which is destructive.
    # For safety, this is a no-op; manually handle if rollback is truly needed.
    pass
