"""add_menu_access_tables

Revision ID: 20260124_menu
Revises: 6356ccd2f4ca
Create Date: 2025-01-24 10:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20260124_menu'
down_revision = '6356ccd2f4ca'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create menu_definitions table
    op.create_table(
        'menu_definitions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('menu_key', sa.String(length=100), nullable=False),
        sa.Column('title', sa.String(length=100), nullable=False),
        sa.Column('path', sa.String(length=255), nullable=True),
        sa.Column('icon', sa.String(length=100), nullable=True),
        sa.Column('parent_key', sa.String(length=100), nullable=True),
        sa.Column('section', sa.String(length=50), nullable=False),
        sa.Column('sort_order', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('menu_key')
    )
    op.create_index(op.f('ix_menu_definitions_id'), 'menu_definitions', ['id'], unique=False)
    op.create_index(op.f('ix_menu_definitions_menu_key'), 'menu_definitions', ['menu_key'], unique=True)
    op.create_index(op.f('ix_menu_definitions_section'), 'menu_definitions', ['section'], unique=False)

    # Create menu_role_access table
    op.create_table(
        'menu_role_access',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('menu_id', sa.Integer(), nullable=False),
        sa.Column('role_name', sa.String(length=50), nullable=False),
        sa.Column('can_view', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['menu_id'], ['menu_definitions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('menu_id', 'role_name', name='uq_menu_role')
    )
    op.create_index(op.f('ix_menu_role_access_id'), 'menu_role_access', ['id'], unique=False)
    op.create_index(op.f('ix_menu_role_access_role_name'), 'menu_role_access', ['role_name'], unique=False)


def downgrade() -> None:
    # Drop menu_role_access table first (due to foreign key)
    op.drop_index(op.f('ix_menu_role_access_role_name'), table_name='menu_role_access')
    op.drop_index(op.f('ix_menu_role_access_id'), table_name='menu_role_access')
    op.drop_table('menu_role_access')

    # Drop menu_definitions table
    op.drop_index(op.f('ix_menu_definitions_section'), table_name='menu_definitions')
    op.drop_index(op.f('ix_menu_definitions_menu_key'), table_name='menu_definitions')
    op.drop_index(op.f('ix_menu_definitions_id'), table_name='menu_definitions')
    op.drop_table('menu_definitions')
