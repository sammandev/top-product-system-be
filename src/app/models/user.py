import enum
from datetime import UTC, datetime

from sqlalchemy import Boolean, Column, DateTime, Enum, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.db import Base
from app.models.rbac import user_roles


def _utc_now():
    """Helper function for SQLAlchemy default/onupdate."""
    return datetime.now(UTC)


class UserRole(enum.StrEnum):
    """User role hierarchy (highest to lowest).

    - developer: Hardcoded identity, auto-assigned on login. Cannot be modified via UI.
    - superadmin: Granted by developer only. Full system access.
    - user: Regular authenticated user with menu_permissions-based access.
    """

    developer = "developer"
    superadmin = "superadmin"
    user = "user"


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
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), unique=True, index=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    email = Column(String(255), nullable=True)  # Optional email field
    worker_id = Column(String(64), nullable=True)  # External worker_id (from DUT API)
    is_admin = Column(Boolean, default=False)  # Local admin privileges (manually granted)
    is_ptb_admin = Column(Boolean, default=False)  # External PTB admin status (synced from external API)
    is_superuser = Column(Boolean, default=False)  # Synced from external API (is_superuser)
    is_staff = Column(Boolean, default=False)  # Synced from external API (is_staff)
    is_active = Column(Boolean, default=True)
    role = Column(Enum(UserRole), default=UserRole.user, nullable=False, server_default="user")  # Access control role
    menu_permissions = Column(JSONB, nullable=True)  # Per-resource CRUD permissions dict
    permission_updated_at = Column(DateTime, nullable=True)  # When permissions were last changed
    token_version = Column(Integer, default=1, nullable=False)
    last_login = Column(DateTime, nullable=True)  # Track last login time
    created_at = Column(DateTime, default=_utc_now, nullable=False)  # Track creation time
    updated_at = Column(DateTime, default=_utc_now, onupdate=_utc_now, nullable=False)  # Track update time
    roles = relationship("Role", secondary=user_roles, back_populates="users", lazy="selectin")
