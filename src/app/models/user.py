from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, Boolean, DateTime, Enum, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base
from app.models.rbac import user_roles

if TYPE_CHECKING:
    from app.models.rbac import Role


def _utc_now():
    """Helper function for SQLAlchemy default/onupdate."""
    return datetime.now(UTC)


class UserRole(enum.StrEnum):
    """User role hierarchy (highest to lowest privilege).

    - developer: Hardcoded identity, auto-assigned on login. Cannot be modified via UI. Full access.
    - superadmin: Granted by developer only. Full system access except modifying developer users.
    - admin: All pages except System Cleanup, App Configuration, and Menu Access.
    - user: Standard pages + Tools. No admin/system pages.
    - guest: Only Top Products Analysis + Data Explorer. Can be granted more via menu_permissions.
    """

    developer = "developer"
    superadmin = "superadmin"
    admin = "admin"
    user = "user"
    guest = "guest"


# Menu resources available in the system (used for menu_permissions dict keys)
MENU_RESOURCES: list[str] = [
    "dashboard",
    "parsing",
    "comparison",
    "top_products",
    "dut_analysis",
    "dut_management",
    "activity",
    "mastercontrol",
    "conversion",
    "admin_users",
    "admin_rbac",
    "admin_cleanup",
    "admin_config",
    "admin_menu_access",
    "admin_access_control",
]

# Valid CRUD actions for menu_permissions
MENU_ACTIONS: list[str] = ["create", "read", "update", "delete"]


def get_default_menu_permissions() -> dict[str, list[str]]:
    """Return default menu permissions for regular users.

    Grants read access to standard pages and no access to admin pages.
    """
    return {
        "dashboard": ["read"],
        "parsing": ["create", "read"],
        "comparison": ["create", "read"],
        "top_products": ["read"],
        "dut_analysis": ["read"],
        "dut_management": ["read"],
        "activity": ["read"],
        "mastercontrol": ["read"],
        "conversion": ["create", "read"],
    }


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(150), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)  # Optional email field
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)  # External worker_id (from DUT API)
    is_admin: Mapped[bool | None] = mapped_column(Boolean, default=False)  # Local admin privileges (manually granted)
    is_ptb_admin: Mapped[bool | None] = mapped_column(Boolean, default=False)  # External PTB admin status (synced from external API)
    is_superuser: Mapped[bool | None] = mapped_column(Boolean, default=False)  # Synced from external API (is_superuser)
    is_staff: Mapped[bool | None] = mapped_column(Boolean, default=False)  # Synced from external API (is_staff)
    is_active: Mapped[bool | None] = mapped_column(Boolean, default=True)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.user, nullable=False, server_default="user")  # Access control role
    menu_permissions: Mapped[dict[str, list[str]] | None] = mapped_column(JSON().with_variant(JSONB(), "postgresql"), nullable=True)
    permission_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # When permissions were last changed
    token_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    last_login: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # Track last login time
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now, nullable=False)  # Track creation time
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now, onupdate=_utc_now, nullable=False)  # Track update time
    roles: Mapped[list[Role]] = relationship("Role", secondary=user_roles, back_populates="users", lazy="selectin")
