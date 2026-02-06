"""add_cached_test_items

Revision ID: 20260206a
Revises: 20260124_menu
Create Date: 2026-02-06

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20260206a'
# UPDATED: Fixed down_revision to match actual revision ID in 20260124_add_menu_access_tables.py
down_revision = '20260124_menu'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create cached_test_items table for caching iPLAS test item names."""
    op.create_table(
        'cached_test_items',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('site', sa.String(100), nullable=False),
        sa.Column('project', sa.String(100), nullable=False),
        sa.Column('station', sa.String(255), nullable=False),
        sa.Column('test_item_name', sa.String(255), nullable=False),
        sa.Column('is_value', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('is_bin', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('has_ucl', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('has_lcl', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    
    # Create indexes for efficient querying
    op.create_index(
        'ix_cached_test_items_site_project_station',
        'cached_test_items',
        ['site', 'project', 'station'],
        unique=False
    )
    op.create_index(
        'ix_cached_test_items_lookup',
        'cached_test_items',
        ['site', 'project', 'station', 'test_item_name'],
        unique=True
    )


def downgrade() -> None:
    """Drop cached_test_items table."""
    op.drop_index('ix_cached_test_items_lookup', table_name='cached_test_items')
    op.drop_index('ix_cached_test_items_site_project_station', table_name='cached_test_items')
    op.drop_table('cached_test_items')
