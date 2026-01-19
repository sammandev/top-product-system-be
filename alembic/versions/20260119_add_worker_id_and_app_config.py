"""Add worker_id to users and app_config table.

Revision ID: 8f24c7d7c2b1
Revises: ce0545504c00
Create Date: 2026-01-19
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "8f24c7d7c2b1"
down_revision = "ce0545504c00"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Normalize existing usernames to lowercase to enforce case-insensitive logins
    # First, handle duplicates by removing all but the first occurrence (case-insensitive)
    # This uses a CTE to identify and delete duplicate usernames
    op.execute("""
        DELETE FROM users
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM users
            GROUP BY LOWER(username)
        )
    """)
    
    # Now lowercase all remaining usernames
    op.execute("UPDATE users SET username = LOWER(username) WHERE username != LOWER(username)")

    op.add_column("users", sa.Column("worker_id", sa.String(length=64), nullable=True))

    op.create_table(
        "app_config",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("description", sa.String(length=500), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_by", sa.String(length=150), nullable=True),
    )

    op.execute(
        """
        INSERT INTO app_config (name, version, description)
        SELECT 'DUT Management API', '1.0.0', 'Wireless Test Data Analysis Platform'
        WHERE NOT EXISTS (SELECT 1 FROM app_config)
        """
    )


def downgrade() -> None:
    op.drop_table("app_config")
    op.drop_column("users", "worker_id")