"""
Router for RBAC (Role-Based Access Control) management.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi_cache.decorator import cache
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.dependencies.authz import get_current_user
from app.models.rbac import Permission, Role
from app.utils.admin_access import is_user_admin
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/rbac", tags=["User_Management"])


# ============================================================================
# Request & Response Models
# ============================================================================


class RoleSchema(BaseModel):
    """Role schema."""

    id: int
    name: str
    permissions: list[str]
    users_count: int

    model_config = ConfigDict(from_attributes=True)


class RBACStatsResponse(BaseModel):
    """RBAC statistics."""

    total_roles: int
    total_permissions: int
    users_with_roles: int
    active_sessions: int


class RoleListResponse(BaseModel):
    """Role list response."""

    roles: list[RoleSchema]
    stats: RBACStatsResponse


class PermissionSchema(BaseModel):
    """Permission schema."""

    id: int
    name: str
    description: str

    model_config = ConfigDict(from_attributes=True)


class PermissionListResponse(BaseModel):
    """Permission list response."""

    permissions: list[PermissionSchema]


class RoleDetailResponse(BaseModel):
    """Detailed role information."""

    id: int
    name: str
    description: str
    permissions: list[str]
    users: list[dict]
    users_count: int
    created_at: str | None = None
    updated_at: str | None = None

    model_config = ConfigDict(from_attributes=True)


class PermissionDetailResponse(BaseModel):
    """Detailed permission information."""

    id: int
    name: str
    description: str
    roles: list[str]
    usage_count: int

    model_config = ConfigDict(from_attributes=True)


class CreateRoleRequest(BaseModel):
    """Request model for creating a role."""

    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(default="", max_length=256)


class UpdateRoleRequest(BaseModel):
    """Request model for updating a role."""

    name: str | None = Field(None, min_length=1, max_length=64)
    description: str | None = Field(None, max_length=256)
    permissions: list[str] | None = None


class CreatePermissionRequest(BaseModel):
    """Request model for creating a permission."""

    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(default="", max_length=256)


class UpdatePermissionRequest(BaseModel):
    """Request model for updating a permission."""

    name: str | None = Field(None, min_length=1, max_length=64)
    description: str | None = Field(None, max_length=256)


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "/roles",
    response_model=RoleListResponse,
    summary="Get all roles",
    description="Retrieve all roles with statistics (Admin only)",
)
@cache(expire=120)  # Cache for 2 minutes (very stable metadata)
async def get_roles(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get all roles with statistics and user counts."""
    if not is_user_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators can access RBAC management")

    try:
        # Get all roles from database
        db_roles = db.query(Role).all()
        all_permissions = db.query(Permission).all()
        all_users = db.query(User).all()

        # Build role list with user counts
        roles = []
        for role in db_roles:
            perm_names = [p.name for p in role.permissions]
            users_count = len(role.users)

            roles.append(RoleSchema(id=role.id, name=role.name, description=role.description or "", permissions=perm_names, users_count=users_count))

        # Calculate statistics
        active_users = sum(1 for user in all_users if user.is_active)
        users_with_roles = sum(1 for user in all_users if user.roles)

        stats = RBACStatsResponse(total_roles=len(db_roles), total_permissions=len(all_permissions), users_with_roles=users_with_roles, active_sessions=active_users)

        return RoleListResponse(roles=roles, stats=stats)

    except Exception as e:
        logger.error(f"Error fetching roles: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch roles") from e


@router.post(
    "/roles",
    response_model=RoleSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Create new role",
    description="Create a new role (Admin only)",
)
async def create_role(
    role_data: CreateRoleRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new role."""
    if not is_user_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators can create roles")

    try:
        # Check if role already exists
        existing_role = db.query(Role).filter(Role.name == role_data.name).first()
        if existing_role:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Role '{role_data.name}' already exists")

        # Create new role
        new_role = Role(name=role_data.name, description=role_data.description)

        db.add(new_role)
        db.commit()
        db.refresh(new_role)

        return RoleSchema(id=new_role.id, name=new_role.name, description=new_role.description or "", permissions=[], users_count=0)

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating role: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create role") from e


@router.get(
    "/roles/{role_id}",
    response_model=RoleDetailResponse,
    summary="Get role details",
    description="Retrieve detailed information about a specific role (Admin only)",
)
@cache(expire=120)  # Cache for 2 minutes (very stable metadata)
async def get_role_details(
    role_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get detailed information about a specific role."""
    if not is_user_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators can access RBAC management")

    try:
        # Get role from database
        role = db.query(Role).filter(Role.id == role_id).first()
        if not role:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found")
        # Get permissions
        perm_names = [p.name for p in role.permissions]

        # Get users with this role
        role_users = []
        for user in role.users:
            role_users.append({"id": user.id, "username": user.username, "email": user.email, "is_active": user.is_active})

        return RoleDetailResponse(id=role.id, name=role.name, description=role.description or "", permissions=perm_names, users=role_users, users_count=len(role_users))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching role details: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch role details") from e


@router.put(
    "/roles/{role_id}",
    response_model=RoleSchema,
    summary="Update role",
    description="Update role information (Admin only)",
)
async def update_role(
    role_id: int,
    request: UpdateRoleRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update role information."""
    if not is_user_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators can update roles")

    try:
        # Get role from database
        role = db.query(Role).filter(Role.id == role_id).first()
        if not role:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found")

        # Update fields
        if request.name is not None:
            # Check if name is taken by another role
            existing = db.query(Role).filter(Role.name == request.name, Role.id != role_id).first()
            if existing:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Role name '{request.name}' already exists")
            role.name = request.name

        if request.description is not None:
            role.description = request.description

        if request.permissions is not None:
            # Get permission objects
            permissions = db.query(Permission).filter(Permission.name.in_(request.permissions)).all()
            role.permissions = permissions

        db.commit()
        db.refresh(role)

        return RoleSchema(id=role.id, name=role.name, description=role.description or "", permissions=[p.name for p in role.permissions], users_count=len(role.users))

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating role: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update role") from e


@router.delete(
    "/roles/{role_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete role",
    description="Delete a role (Admin only)",
)
async def delete_role(
    role_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a role."""
    if not is_user_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators can delete roles")

    try:
        role = db.query(Role).filter(Role.id == role_id).first()
        if not role:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found")

        # Don't allow deleting admin role
        if role.name == "admin":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete admin role")

        db.delete(role)
        db.commit()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting role: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete role") from e


@router.get(
    "/permissions",
    response_model=PermissionListResponse,
    summary="Get all permissions",
    description="Retrieve all available permissions (Admin only)",
)
@cache(expire=120)  # Cache for 2 minutes (very stable metadata)
async def get_permissions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get all available permissions."""
    if not is_user_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators can access RBAC management")

    try:
        # Get all permissions from database
        db_permissions = db.query(Permission).all()

        permissions = [PermissionSchema(id=perm.id, name=perm.name, description=perm.description or "") for perm in db_permissions]

        return PermissionListResponse(permissions=permissions)

    except Exception as e:
        logger.error(f"Error fetching permissions: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch permissions") from e


@router.post(
    "/permissions",
    response_model=PermissionSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Create new permission",
    description="Create a new permission (Admin only)",
)
async def create_permission(
    perm_data: CreatePermissionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new permission."""
    if not is_user_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators can create permissions")

    try:
        # Check if permission already exists
        existing_perm = db.query(Permission).filter(Permission.name == perm_data.name).first()
        if existing_perm:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Permission '{perm_data.name}' already exists")

        # Create new permission
        new_perm = Permission(name=perm_data.name, description=perm_data.description)

        db.add(new_perm)
        db.commit()
        db.refresh(new_perm)

        return PermissionSchema(id=new_perm.id, name=new_perm.name, description=new_perm.description or "")

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating permission: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create permission") from e


@router.get(
    "/permissions/{permission_id}",
    response_model=PermissionDetailResponse,
    summary="Get permission details",
    description="Retrieve detailed information about a specific permission (Admin only)",
)
@cache(expire=120)  # Cache for 2 minutes (very stable metadata)
async def get_permission_details(
    permission_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get detailed information about a specific permission."""
    if not is_user_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators can access RBAC management")

    try:
        # Get permission from database
        perm = db.query(Permission).filter(Permission.id == permission_id).first()
        if not perm:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Permission not found")

        # Get roles that have this permission
        role_names = [r.name for r in perm.roles]
        usage_count = len(role_names)

        return PermissionDetailResponse(id=perm.id, name=perm.name, description=perm.description or "", roles=role_names, usage_count=usage_count)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching permission details: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to fetch permission details") from e


@router.put(
    "/permissions/{permission_id}",
    response_model=PermissionSchema,
    summary="Update permission",
    description="Update permission information (Admin only)",
)
async def update_permission(
    permission_id: int,
    request: UpdatePermissionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update permission information."""
    if not is_user_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators can update permissions")

    try:
        # Get permission from database
        perm = db.query(Permission).filter(Permission.id == permission_id).first()
        if not perm:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Permission not found")

        # Update fields
        if request.name is not None:
            # Check if name is taken by another permission
            existing = db.query(Permission).filter(Permission.name == request.name, Permission.id != permission_id).first()
            if existing:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Permission name '{request.name}' already exists")
            perm.name = request.name

        if request.description is not None:
            perm.description = request.description

        db.commit()
        db.refresh(perm)

        return PermissionSchema(id=perm.id, name=perm.name, description=perm.description or "")

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating permission: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update permission") from e


@router.delete(
    "/permissions/{permission_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete permission",
    description="Delete a permission (Admin only)",
)
async def delete_permission(
    permission_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a permission."""
    if not is_user_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only administrators can delete permissions")

    try:
        # Get permission from database
        perm = db.query(Permission).filter(Permission.id == permission_id).first()
        if not perm:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Permission not found")

        db.delete(perm)
        db.commit()

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting permission: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete permission") from e
