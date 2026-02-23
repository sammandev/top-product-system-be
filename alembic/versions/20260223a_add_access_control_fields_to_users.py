"""add access control fields to users

Revision ID: 20260223a
Revises: 20260206a
Create Date: 2026-02-23
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260223a"
down_revision = "20260206a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create the UserRole enum type in PostgreSQL
    userrole_enum = postgresql.ENUM("developer", "superadmin", "user", name="userrole", create_type=False)
    userrole_enum.create(op.get_bind(), checkfirst=True)

    # Add new columns to users table
    op.add_column("users", sa.Column("is_superuser", sa.Boolean(), nullable=True, server_default=sa.text("false")))
    op.add_column("users", sa.Column("is_staff", sa.Boolean(), nullable=True, server_default=sa.text("false")))
    op.add_column(
        "users",
        sa.Column(
            "role",
            userrole_enum,
            nullable=True,
            server_default="user",
        ),
    )
    op.add_column("users", sa.Column("menu_permissions", postgresql.JSONB(), nullable=True))
    op.add_column("users", sa.Column("permission_updated_at", sa.DateTime(), nullable=True))

    # Backfill existing rows with default role
    op.execute("UPDATE users SET role = 'user' WHERE role IS NULL")
    op.execute("UPDATE users SET is_superuser = false WHERE is_superuser IS NULL")
    op.execute("UPDATE users SET is_staff = false WHERE is_staff IS NULL")

    # Make role NOT NULL after backfill
    op.alter_column("users", "role", nullable=False)
    op.alter_column("users", "is_superuser", nullable=False, server_default=sa.text("false"))
    op.alter_column("users", "is_staff", nullable=False, server_default=sa.text("false"))


def downgrade() -> None:
    op.drop_column("users", "permission_updated_at")
    op.drop_column("users", "menu_permissions")
    op.drop_column("users", "role")
    op.drop_column("users", "is_staff")
    op.drop_column("users", "is_superuser")

    # Drop the enum type
    userrole_enum = postgresql.ENUM("developer", "superadmin", "user", name="userrole", create_type=False)
    userrole_enum.drop(op.get_bind(), checkfirst=True)
