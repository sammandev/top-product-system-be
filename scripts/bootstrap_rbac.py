"""
Bootstrap default permissions, roles, and seed users for the AST Parser backend.

Usage (from repository root):

    uv run python scripts/bootstrap_rbac.py

Optional arguments let you override default passwords or skip user creation:

    uv run python scripts/bootstrap_rbac.py --admin-password S3cret --skip-default-users
"""

import argparse
from collections.abc import Iterable

from app.db import SessionLocal
from app.db.init_db import init_db
from app.models.rbac import Permission, Role
from app.utils.auth import create_user

PERMISSIONS: dict[str, str] = {
    "read": "Allows reading data resources",
    "write": "Allows modifying data resources",
    "analyze": "Allows running analysis workflows",
    "manage_users": "Allows managing application users",
}

ROLE_PERMISSION_MATRIX: dict[str, Iterable[str]] = {
    "admin": PERMISSIONS.keys(),
    "analyst": ("read", "analyze"),
    "viewer": ("read",),
}

DEFAULT_USERS: tuple[tuple[str, bool, Iterable[str], str], ...] = (
    ("admin", True, ("admin",), "admin123"),
    ("analyst", False, ("analyst",), "analyst123"),
    ("viewer", False, ("viewer",), "viewer123"),
)


def ensure_permission(db, name: str, description: str) -> Permission:
    perm = db.query(Permission).filter(Permission.name == name).one_or_none()
    if perm:
        if description and perm.description != description:
            perm.description = description
        return perm
    perm = Permission(name=name, description=description)
    db.add(perm)
    db.flush()
    return perm


def ensure_role(db, name: str, description: str = "") -> Role:
    role = db.query(Role).filter(Role.name == name).one_or_none()
    if role:
        if description and role.description != description:
            role.description = description
        return role
    role = Role(name=name, description=description)
    db.add(role)
    db.flush()
    return role


def assign_permissions_to_role(role: Role, permission_names: Iterable[str], permissions_by_name: dict[str, Permission]):
    desired = {permissions_by_name[name] for name in permission_names}
    current = set(role.permissions)
    to_add = desired - current
    to_remove = current - desired
    for perm in to_add:
        role.permissions.append(perm)
    for perm in to_remove:
        role.permissions.remove(perm)


def attach_roles_to_user(user, role_names: Iterable[str], roles_by_name: dict[str, Role]):
    desired = {roles_by_name[name] for name in role_names}
    current = set(user.roles)
    to_add = desired - current
    to_remove = current - desired
    for role in to_add:
        user.roles.append(role)
    for role in to_remove:
        user.roles.remove(role)


def bootstrap(default_passwords: dict[str, str], skip_users: bool = False) -> None:
    init_db()
    session = SessionLocal()
    try:
        permissions: dict[str, Permission] = {}
        for perm_name, description in PERMISSIONS.items():
            permissions[perm_name] = ensure_permission(session, perm_name, description)

        roles: dict[str, Role] = {}
        for role_name, perm_names in ROLE_PERMISSION_MATRIX.items():
            role = ensure_role(session, role_name, f"Auto-created role: {role_name}")
            assign_permissions_to_role(role, perm_names, permissions)
            roles[role_name] = role

        if not skip_users:
            for username, is_admin, role_names, default_password in DEFAULT_USERS:
                password = default_passwords.get(username, default_password)
                user = create_user(session, username, password, is_admin=is_admin)
                attach_roles_to_user(user, role_names, roles)
                session.add(user)

        session.commit()
    finally:
        session.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap default RBAC data.")
    parser.add_argument("--admin-password", help="Password for seeded 'admin' user.")
    parser.add_argument("--analyst-password", help="Password for seeded 'analyst' user.")
    parser.add_argument("--viewer-password", help="Password for seeded 'viewer' user.")
    parser.add_argument(
        "--skip-default-users",
        action="store_true",
        help="Only create permissions/roles; skip creating default users.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    password_overrides = {
        key: value
        for key, value in {
            "admin": args.admin_password,
            "analyst": args.analyst_password,
            "viewer": args.viewer_password,
        }.items()
        if value
    }
    bootstrap(password_overrides, skip_users=args.skip_default_users)
    print("Bootstrap completed successfully.")


if __name__ == "__main__":
    main()
