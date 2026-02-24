"""Add app config expansion tables (iplas_tokens, sfistsp_configs, guest_credentials) and extend app_config.

Revision ID: 20260224a
Revises: 20260223c
Create Date: 2025-01-24
"""

import sqlalchemy as sa  # noqa: I001
from alembic import op

# revision identifiers, used by Alembic.
revision = "20260224a"
down_revision = "20260223c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new columns to app_config
    op.add_column("app_config", sa.Column("tab_title", sa.String(200), nullable=True))
    op.add_column("app_config", sa.Column("favicon_path", sa.String(500), nullable=True))

    # Create iplas_tokens table
    op.create_table(
        "iplas_tokens",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("site", sa.String(10), nullable=False, index=True),
        sa.Column("base_url", sa.String(500), nullable=False),
        sa.Column("token_value", sa.Text(), nullable=False),
        sa.Column("label", sa.String(200), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_by", sa.String(150), nullable=True),
    )

    # Create sfistsp_configs table
    op.create_table(
        "sfistsp_configs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("base_url", sa.String(500), nullable=False),
        sa.Column("program_id", sa.String(200), nullable=False),
        sa.Column("program_password", sa.Text(), nullable=False),
        sa.Column("timeout", sa.Float(), nullable=False, server_default=sa.text("30.0")),
        sa.Column("label", sa.String(200), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_by", sa.String(150), nullable=True),
    )

    # Create guest_credentials table
    op.create_table(
        "guest_credentials",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("username", sa.Text(), nullable=False),
        sa.Column("password", sa.Text(), nullable=False),
        sa.Column("label", sa.String(200), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_by", sa.String(150), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("guest_credentials")
    op.drop_table("sfistsp_configs")
    op.drop_table("iplas_tokens")
    op.drop_column("app_config", "favicon_path")
    op.drop_column("app_config", "tab_title")
