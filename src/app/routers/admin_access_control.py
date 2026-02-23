"""
Router for Access Control Management (Super Admin / Developer).

Provides endpoints for managing user roles, permissions, and access settings.
Only accessible by users with superadmin or developer role.
"""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.dependencies.authz import get_current_user
from app.models.user import MENU_ACTIONS, MENU_RESOURCES, User, UserRole, get_default_menu_permissions
from app.utils.admin_access import is_developer_user, is_superadmin_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/access-control", tags=["Access_Control"])

# Module-level dependencies
current_user_dependency = Depends(get_current_user)
db_dependency = Depends(get_db)


# ============================================================================
# Request & Response Models
# ============================================================================


class AccessControlUserSchema(BaseModel):
    """User schema for access control management."""

    id: int
    username: str
    email: str | None
    worker_id: str | None
    is_admin: bool
    is_ptb_admin: bool
    is_superuser: bool
    is_staff: bool
    is_active: bool
    role: str
    menu_permissions: dict[str, list[str]] | None
    permission_updated_at: datetime | None
    last_login: datetime | None
    created_at: datetime


class AccessControlUserListResponse(BaseModel):
    """Response for listing users with access control settings."""

    users: list[AccessControlUserSchema]
    available_resources: list[str]
    available_actions: list[str]
    total: int


class UpdateAccessRequest(BaseModel):
    """Request to update a user's access control settings."""

    role: str | None = Field(None, description="New role: 'superadmin' or 'user'. Cannot set 'developer'.")
    menu_permissions: dict[str, list[str]] | None = Field(None, description="Per-resource CRUD permissions")
    is_active: bool | None = Field(None, description="Activate/deactivate user")
    is_ptb_admin: bool | None = Field(None, description="PTB admin flag")
    is_superuser: bool | None = Field(None, description="Superuser flag")
    is_staff: bool | None = Field(None, description="Staff flag")


class MenuResourcesResponse(BaseModel):
    """Response listing available menu resources and actions."""

    resources: list[str]
    actions: list[str]
    default_permissions: dict[str, list[str]]


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "/users",
    response_model=AccessControlUserListResponse,
    summary="List all users with access control settings",
    description="Get all users with their roles, permissions, and access settings. Requires superadmin or developer role.",
)
async def list_access_control_users(
    current_user: User = current_user_dependency,
    db: Session = db_dependency,
):
    """List all users with their access control settings."""
    if not is_superadmin_user(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superadmin or developer privileges required",
        )

    try:
        users = db.query(User).order_by(User.username).all()
        user_list = []
        for u in users:
            user_list.append(
                AccessControlUserSchema(
                    id=u.id,
                    username=u.username,
                    email=u.email,
                    worker_id=u.worker_id,
                    is_admin=u.is_admin,
                    is_ptb_admin=u.is_ptb_admin,
                    is_superuser=u.is_superuser,
                    is_staff=u.is_staff,
                    is_active=u.is_active,
                    role=u.role.value if u.role else "user",
                    menu_permissions=u.menu_permissions,
                    permission_updated_at=u.permission_updated_at,
                    last_login=u.last_login,
                    created_at=u.created_at,
                )
            )

        return AccessControlUserListResponse(
            users=user_list,
            available_resources=MENU_RESOURCES,
            available_actions=MENU_ACTIONS,
            total=len(user_list),
        )

    except Exception as e:
        logger.error(f"Error listing access control users: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch access control users",
        ) from e


@router.get(
    "/resources",
    response_model=MenuResourcesResponse,
    summary="Get available menu resources and actions",
    description="Returns the list of menu resources and CRUD actions available for permission assignment.",
)
async def get_menu_resources(
    current_user: User = current_user_dependency,
):
    """Get available menu resources and actions."""
    if not is_superadmin_user(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superadmin or developer privileges required",
        )

    return MenuResourcesResponse(
        resources=MENU_RESOURCES,
        actions=MENU_ACTIONS,
        default_permissions=get_default_menu_permissions(),
    )


@router.patch(
    "/users/{user_id}",
    response_model=AccessControlUserSchema,
    summary="Update user access control settings",
    description="Update a user's role, menu permissions, and flags. Requires superadmin or developer role.",
)
async def update_user_access(
    user_id: int,
    request: UpdateAccessRequest,
    current_user: User = current_user_dependency,
    db: Session = db_dependency,
):
    """Update a user's access control settings.

    Rules:
    - Developer users cannot be modified by anyone (they are hardcoded).
    - Only developers can grant/revoke the 'superadmin' role.
    - The 'developer' role cannot be assigned via this endpoint.
    - Users cannot modify their own access settings.
    """
    if not is_superadmin_user(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superadmin or developer privileges required",
        )

    try:
        target_user = db.query(User).filter(User.id == user_id).first()
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found",
            )

        # Prevent modifying developer users
        if is_developer_user(target_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Developer users cannot be modified",
            )

        # Prevent modifying own access settings
        if target_user.id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot modify your own access settings",
            )

        # Validate and apply role change
        if request.role is not None:
            if request.role == "developer":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The 'developer' role cannot be assigned via this endpoint",
                )

            if request.role == "superadmin" and not is_developer_user(current_user):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Only developers can grant the superadmin role",
                )

            if request.role not in ("superadmin", "user"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid role: '{request.role}'. Must be 'superadmin' or 'user'.",
                )

            old_role = target_user.role.value if target_user.role else "user"
            target_user.role = UserRole(request.role)
            logger.info(f"Role changed for {target_user.username}: {old_role} -> {request.role} by {current_user.username}")

        # Validate and apply menu permissions
        if request.menu_permissions is not None:
            # Validate resource names
            for resource in request.menu_permissions:
                if resource not in MENU_RESOURCES:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid resource: '{resource}'. Valid: {MENU_RESOURCES}",
                    )
                # Validate actions
                for action in request.menu_permissions[resource]:
                    if action not in MENU_ACTIONS:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid action: '{action}' for resource '{resource}'. Valid: {MENU_ACTIONS}",
                        )

            target_user.menu_permissions = request.menu_permissions
            target_user.permission_updated_at = datetime.now(UTC)
            logger.info(f"Menu permissions updated for {target_user.username} by {current_user.username}")

        # Apply boolean flags
        if request.is_active is not None:
            target_user.is_active = request.is_active
            # Revoke tokens if deactivated
            if not request.is_active:
                target_user.token_version = (target_user.token_version or 0) + 1
                logger.warning(f"User {target_user.username} deactivated by {current_user.username}")

        if request.is_ptb_admin is not None:
            target_user.is_ptb_admin = request.is_ptb_admin

        if request.is_superuser is not None:
            target_user.is_superuser = request.is_superuser

        if request.is_staff is not None:
            target_user.is_staff = request.is_staff

        target_user.updated_at = datetime.now(UTC)
        db.commit()
        db.refresh(target_user)

        logger.info(f"Access settings updated for {target_user.username} (ID: {target_user.id}) by {current_user.username}")

        return AccessControlUserSchema(
            id=target_user.id,
            username=target_user.username,
            email=target_user.email,
            worker_id=target_user.worker_id,
            is_admin=target_user.is_admin,
            is_ptb_admin=target_user.is_ptb_admin,
            is_superuser=target_user.is_superuser,
            is_staff=target_user.is_staff,
            is_active=target_user.is_active,
            role=target_user.role.value if target_user.role else "user",
            menu_permissions=target_user.menu_permissions,
            permission_updated_at=target_user.permission_updated_at,
            last_login=target_user.last_login,
            created_at=target_user.created_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating access for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user access settings",
        ) from e
