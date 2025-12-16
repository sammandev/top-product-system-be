from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi_cache.decorator import cache
from sqlalchemy.orm import Session

from app.db import get_db
from app.dependencies.authz import require_admin, require_permission
from app.models.rbac import Permission, Role
from app.models.user import User

router = APIRouter(prefix="/api/rbac", tags=["RBAC_Management"])

# Module-level dependency object to avoid calling Depends() in function defaults
dut_user_admin_dependency = Depends(require_admin)
dut_permission_dependency = Depends(require_permission)
dut_read_permission_dependency = Depends(require_permission("read"))
dut_db_dependency = Depends(get_db)


# ---- Roles ----
@router.post(
    "/roles",
    summary="(admin) Create role",
    description="Create a new role with an optional description. Admin privileges required.",
)
def create_role(
    name: Annotated[str, Form(description="Role name (unique identifier) (e.g., data_analyst)")],
    description: Annotated[str, Form(description="Role description (optional) (e.g., Can review dashboards)")] = "",
    db: Session = dut_db_dependency,
    _: User = dut_user_admin_dependency,
):
    if db.query(Role).filter(Role.name == name).one_or_none():
        raise HTTPException(409, "role already exists")
    r = Role(name=name, description=description)
    db.add(r)
    db.commit()
    db.refresh(r)
    return {"id": r.id, "name": r.name, "description": r.description}


@router.get(
    "/roles",
    summary="(admin) List roles",
    description="Return all roles with their assigned permissions. Admin privileges required.",
)
@cache(expire=120)  # Cache for 2 minutes (very stable metadata)
def list_roles(_: User = dut_user_admin_dependency, db: Session = dut_db_dependency):
    roles = db.query(Role).all()
    return {"roles": [{"id": r.id, "name": r.name, "permissions": [p.name for p in r.permissions]} for r in roles]}


@router.delete(
    "/roles/{role_id}",
    summary="(admin) Delete role",
    description="Delete an existing role by ID. Admin privileges required.",
)
def delete_role(role_id: int, db: Session = dut_db_dependency, _: User = dut_user_admin_dependency):
    r = db.get(Role, role_id)
    if not r:
        raise HTTPException(404, "role not found")
    db.delete(r)
    db.commit()
    return {"deleted": role_id}


# ---- Permissions ----
@router.post(
    "/permissions",
    summary="(admin) Create permission",
    description="Register a new permission with an optional description. Admin privileges required.",
)
def create_permission(
    name: Annotated[str, Form(description="Permission name (unique identifier) (e.g., read_reports)")],
    description: Annotated[
        str,
        Form(description="Permission description (optional) (e.g., Allows viewing report data)"),
    ] = "",
    db: Session = dut_db_dependency,
    _: User = dut_user_admin_dependency,
):
    if db.query(Permission).filter(Permission.name == name).one_or_none():
        raise HTTPException(409, "permission already exists")
    p = Permission(name=name, description=description)
    db.add(p)
    db.commit()
    db.refresh(p)
    return {"id": p.id, "name": p.name, "description": p.description}


@router.get(
    "/permissions",
    summary="(admin) List permissions",
    description="Return all available permissions in the system. Admin privileges required.",
)
@cache(expire=120)  # Cache for 2 minutes (very stable metadata)
def list_permissions(_: User = dut_user_admin_dependency, db: Session = dut_db_dependency):
    perms = db.query(Permission).all()
    return {"permissions": [{"id": p.id, "name": p.name} for p in perms]}


@router.delete(
    "/permissions/{perm_id}",
    summary="(admin) Delete permission",
    description="Delete a permission by ID. Admin privileges required.",
)
def delete_permission(perm_id: int, db: Session = dut_db_dependency, _: User = dut_user_admin_dependency):
    p = db.get(Permission, perm_id)
    if not p:
        raise HTTPException(404, "permission not found")
    db.delete(p)
    db.commit()
    return {"deleted": perm_id}


# ---- Role <-> Permission links ----
@router.post(
    "/roles/{role_id}/grant",
    summary="(admin) Grant permission to role",
    description="Attach an existing permission to the specified role. Admin privileges required.",
)
def grant_permission(
    role_id: int,
    perm_id: Annotated[int, Form(description="Permission ID to grant to the role (e.g., 12)")],
    db: Session = dut_db_dependency,
    _: User = dut_user_admin_dependency,
):
    r = db.get(Role, role_id)
    p = db.get(Permission, perm_id)
    if not r or not p:
        raise HTTPException(404, "role or permission not found")
    if p not in r.permissions:
        r.permissions.append(p)
        db.commit()
    return {"role": r.name, "granted": p.name}


@router.post(
    "/roles/{role_id}/revoke",
    summary="(admin) Revoke permission from role",
    description="Remove a permission from the specified role. Admin privileges required.",
)
def revoke_permission(
    role_id: int,
    perm_id: Annotated[int, Form(description="Permission ID to revoke from the role (e.g., 12)")],
    db: Session = dut_db_dependency,
    _: User = dut_user_admin_dependency,
):
    r = db.get(Role, role_id)
    p = db.get(Permission, perm_id)
    if not r or not p:
        raise HTTPException(404, "role or permission not found")
    if p in r.permissions:
        r.permissions.remove(p)
        db.commit()
    return {"role": r.name, "revoked": p.name}


# ---- User <-> Role links ----
@router.post(
    "/users/{user_id}/assign-role",
    summary="(admin) Assign role to user",
    description="Assign a role to a user. Admin privileges required.",
)
def assign_role(
    user_id: int,
    role_id: Annotated[int, Form(description="Role ID to assign to the user (e.g., 5)")],
    db: Session = dut_db_dependency,
    _: User = dut_user_admin_dependency,
):
    u = db.get(User, user_id)
    r = db.get(Role, role_id)
    if not u or not r:
        raise HTTPException(404, "user or role not found")
    if r not in u.roles:
        u.roles.append(r)
        db.commit()
    return {"user": u.username, "role": r.name, "assigned": True}


@router.post(
    "/users/{user_id}/remove-role",
    summary="(admin) Remove role from user",
    description="Detach a role from a user. Admin privileges required.",
)
def remove_role(
    user_id: int,
    role_id: Annotated[int, Form(description="Role ID to remove from the user (e.g., 5)")],
    db: Session = dut_db_dependency,
    _: User = dut_user_admin_dependency,
):
    u = db.get(User, user_id)
    r = db.get(Role, role_id)
    if not u or not r:
        raise HTTPException(404, "user or role not found")
    if r in u.roles:
        u.roles.remove(r)
        db.commit()
    return {"user": u.username, "role": r.name, "removed": True}


# ---- Example protected route using a permission ----
@router.get(
    "/check-read-permission",
    summary="Example: requires 'read' permission",
    description="Demonstration endpoint protected by the 'read' permission.",
)
def check_read_permission(_: User = dut_read_permission_dependency):
    return {"ok": True, "message": "You have 'read' permission."}
