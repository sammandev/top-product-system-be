"""add_description_to_menu_definitions

Revision ID: 20260223b
Revises: 20260223a
Create Date: 2026-02-23

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260223b"
down_revision = "20260223a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "menu_definitions",
        sa.Column("description", sa.String(length=256), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("menu_definitions", "description")
