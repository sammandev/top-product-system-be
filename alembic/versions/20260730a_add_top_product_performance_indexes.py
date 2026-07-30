"""Add top product performance indexes.

Revision ID: 20260730a
Revises: 20260224a
Create Date: 2026-07-30
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20260730a"
down_revision = "20260224a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("ix_top_products_project_name", "top_products", ["project_name"], unique=False)
    op.create_index("ix_top_products_created_at", "top_products", ["created_at"], unique=False)
    op.create_index("ix_top_products_score", "top_products", ["score"], unique=False)
    op.create_index(
        "ix_top_products_project_station",
        "top_products",
        ["project_name", "station_name"],
        unique=False,
    )
    op.create_index(
        "ix_top_product_measurements_top_product_id",
        "top_product_measurements",
        ["top_product_id"],
        unique=False,
    )

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        op.create_index(
            "ix_top_products_dut_isn_trgm",
            "top_products",
            ["dut_isn"],
            unique=False,
            postgresql_using="gin",
            postgresql_ops={"dut_isn": "gin_trgm_ops"},
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.drop_index("ix_top_products_dut_isn_trgm", table_name="top_products")

    op.drop_index("ix_top_product_measurements_top_product_id", table_name="top_product_measurements")
    op.drop_index("ix_top_products_project_station", table_name="top_products")
    op.drop_index("ix_top_products_score", table_name="top_products")
    op.drop_index("ix_top_products_created_at", table_name="top_products")
    op.drop_index("ix_top_products_project_name", table_name="top_products")
