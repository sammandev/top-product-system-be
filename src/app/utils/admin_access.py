from __future__ import annotations

from app.models.user import User, UserRole

# Hardcoded developer identities — these users are auto-assigned the developer role
# and cannot be modified via the UI. Only add trusted system administrators here.
DEVELOPER_USERNAMES = {
    "samuel_halomoan",
}

DEVELOPER_WORKER_IDS = {
    "MW2400549",
}

# Legacy admin allowlists (kept for backward compatibility with is_user_admin)
ADMIN_USERNAMES = DEVELOPER_USERNAMES
ADMIN_WORKER_IDS = DEVELOPER_WORKER_IDS


def _normalize_username(username: str | None) -> str:
    return (username or "").strip().lower()


def _normalize_worker_id(worker_id: str | None) -> str:
    return (worker_id or "").strip().upper()


def is_developer_identity(username: str | None, worker_id: str | None) -> bool:
    """Check if the given identity matches a hardcoded developer identity."""
    if _normalize_username(username) in DEVELOPER_USERNAMES:
        return True
    if _normalize_worker_id(worker_id) in DEVELOPER_WORKER_IDS:
        return True
    return False


def is_developer_user(user: User) -> bool:
    """Check if user has the developer role (highest privilege)."""
    return getattr(user, "role", None) == UserRole.developer or is_developer_identity(
        user.username, getattr(user, "worker_id", None)
    )


def is_superadmin_user(user: User) -> bool:
    """Check if user has superadmin role or higher."""
    if is_developer_user(user):
        return True
    return getattr(user, "role", None) == UserRole.superadmin


def is_admin_user(user: User) -> bool:
    """Check if user has admin role or higher (admin, superadmin, developer)."""
    if is_superadmin_user(user):
        return True
    return getattr(user, "role", None) == UserRole.admin


def is_user_admin(user: User) -> bool:
    """Check if user has any admin-level access.

    Returns True for developer, superadmin, admin, is_admin, is_ptb_admin,
    or hardcoded allowlist entries.
    """
    if is_admin_user(user):
        return True

    if user.is_admin or user.is_ptb_admin:
        return True

    if _normalize_username(user.username) in ADMIN_USERNAMES:
        return True

    if _normalize_worker_id(getattr(user, "worker_id", None)) in ADMIN_WORKER_IDS:
        return True

    return False


def check_menu_permission(user: User, resource: str, action: str = "read") -> bool:
    """Check if a user has permission for a specific resource and action.

    Priority chain (matches PTB OT pattern):
    1. Developer/Superadmin → full access (bypass)
    2. Explicit menu_permissions dict → per-resource CRUD check
    3. is_ptb_admin → full access
    4. is_admin → full access
    5. Default → deny
    """
    # 1. Developer or superadmin bypass
    if is_superadmin_user(user):
        return True

    # 2. Explicit menu_permissions (dict format: {"resource": ["create","read",...]})
    perms = getattr(user, "menu_permissions", None)
    if isinstance(perms, dict) and perms:
        resource_perms = perms.get(resource)
        if resource_perms is not None:
            return action in resource_perms
        # Resource not in permissions dict → denied
        return False

    # 3. PTB admin or local admin → full access
    if user.is_ptb_admin or user.is_admin:
        return True

    # 4. Default deny
    return False
