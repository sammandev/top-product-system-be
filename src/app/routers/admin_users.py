"""
Router for User Management (Admin).
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.dependencies.authz import get_current_user
from app.models.rbac import Role
from app.models.user import User
from app.utils.auth import hash_password

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/users", tags=["User_Management"])

# Module-level dependencies to avoid linting issues
current_user_dependency = Depends(get_current_user)
db_dependency = Depends(get_db)


# ============================================================================
# Request & Response Models
# ============================================================================


class UserCreateRequest(BaseModel):
    """User creation request."""

    username: str = Field(..., min_length=3, max_length=50)
    email: str | None = Field(None, max_length=255)
    password: str = Field(..., min_length=6)
    roles: list[str] = Field(default_factory=list)
    is_active: bool = True


class UserUpdateRequest(BaseModel):
    """User update request."""

    email: str | None = None
    roles: list[str] | None = None
    is_active: bool | None = None
    password: str | None = Field(None, min_length=6)


class UserSchema(BaseModel):
    """User schema."""

    id: int
    username: str
    email: str | None
    roles: list[str]
    is_active: bool
    last_login: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserStatsResponse(BaseModel):
    """User statistics."""

    total_users: int
    active_users: int
    online_users: int
    new_users: int


class UserListResponse(BaseModel):
    """User list response."""

    users: list[UserSchema]
    stats: UserStatsResponse


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "",
    response_model=UserListResponse,
    summary="Get all users",
    description="Retrieve all users with statistics (Admin only)",
)
# Cache disabled - users are frequently updated (activate/deactivate/role changes)
# Caching causes UI to show stale data after updates
async def get_users(
    current_user: User = current_user_dependency,
    db: Session = db_dependency,
):
    """
    Get all users with statistics.
    
    **Requires:** Admin authentication
    
    **Description:**
    Retrieves a complete list of all users in the system along with aggregate statistics
    including total users, active users, admin users, and recently active users.
    
    **Returns:**
    - **users**: List of user objects with id, username, email, is_admin, is_active, created_at
    - **stats**: Statistics object containing total_users, active_users, admin_users, recent_users
    
    **Example Response:**
    ```json
    {
        "users": [
            {"id": 1, "username": "admin", "email": "admin@example.com", "is_admin": true, ...}
        ],
        "stats": {"total_users": 10, "active_users": 8, "admin_users": 2, "recent_users": 3}
    }
    ```
    """
    # Check if user is admin
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can access user management"
        )

    try:
        # Get all users
        users = db.query(User).all()

        # Convert users to schema
        user_list = []
        for user in users:
            # Extract role names from Role objects
            roles_list = [role.name for role in user.roles] if user.roles else []
            user_list.append(UserSchema(
                id=user.id,
                username=user.username,
                email=user.email,
                roles=roles_list,
                is_active=user.is_active,
                last_login=user.last_login,
                created_at=user.created_at,
                updated_at=user.updated_at
            ))

        # Calculate statistics
        total_users = len(users)
        active_users = sum(1 for user in users if user.is_active)

        # Online users (logged in within last hour)
        one_hour_ago = datetime.now(UTC) - timedelta(hours=1)
        # Convert naive datetime to UTC-aware for comparison
        online_users = sum(
            1 for user in users
            if user.last_login and (
                user.last_login.replace(tzinfo=UTC) if user.last_login.tzinfo is None
                else user.last_login
            ) > one_hour_ago
        )

        # New users (created within last 30 days)
        thirty_days_ago = datetime.now(UTC) - timedelta(days=30)
        new_users = sum(
            1 for user in users
            if user.created_at and (
                user.created_at.replace(tzinfo=UTC) if user.created_at.tzinfo is None
                else user.created_at
            ) > thirty_days_ago
        )

        stats = UserStatsResponse(
            total_users=total_users,
            active_users=active_users,
            online_users=online_users,
            new_users=new_users
        )

        return UserListResponse(users=user_list, stats=stats)

    except Exception as e:
        logger.error(f"Error fetching users: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch users"
        ) from e


@router.post(
    "",
    response_model=UserSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Create new user",
    description="Create a new user account (Admin only)",
)
async def create_user(
    user_data: UserCreateRequest,
    current_user: User = current_user_dependency,
    db: Session = db_dependency,
):
    """
    Create a new user account.
    
    **Requires:** Admin authentication
    
    **Description:**
    Creates a new user account with the specified username, email, password, and admin status.
    Passwords are automatically hashed using bcrypt before storage.
    
    **Parameters:**
    - **username**: Unique username for the new user (alphanumeric and underscores)
    - **email**: Valid email address
    - **password**: Password (minimum 6 characters)
    - **is_admin**: Boolean flag to grant admin privileges (default: false)
    
    **Returns:**
    User object with id, username, email, is_admin, is_active, created_at
    
    **Raises:**
    - **400**: Username or email already exists
    - **422**: Invalid data format
    """
    # Check if user is admin
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can create users"
        )

    try:
        # Check if username already exists
        existing_user = db.query(User).filter(User.username == user_data.username).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Username '{user_data.username}' already exists"
            )

        # Create new user
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            password_hash=hash_password(user_data.password),
            is_active=user_data.is_active,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC)
        )

        # Assign roles
        if user_data.roles:
            roles = db.query(Role).filter(Role.name.in_(user_data.roles)).all()
            new_user.roles = roles

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        # Extract role names
        roles_list = [role.name for role in new_user.roles] if new_user.roles else []

        return UserSchema(
            id=new_user.id,
            username=new_user.username,
            email=new_user.email,
            roles=roles_list,
            is_active=new_user.is_active,
            last_login=new_user.last_login,
            created_at=new_user.created_at,
            updated_at=new_user.updated_at
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        ) from e


@router.put(
    "/{user_id}",
    response_model=UserSchema,
    summary="Update user",
    description="Update user information (Admin only)",
)
async def update_user(
    user_id: int,
    user_data: UserUpdateRequest,
    current_user: User = current_user_dependency,
    db: Session = db_dependency,
):
    """
    Update an existing user.

    Args:
        user_id: User ID to update
        user_data: User update data
        current_user: Authenticated admin user
        db: Database session

    Returns:
        UserSchema with updated user data
    """
    # Check if user is admin
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can update users"
        )

    try:
        # Get user
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )

        # Prevent admin from deactivating themselves
        if user_data.is_active is not None and user_data.is_active is False:
            if user.id == current_user.id:
                logger.warning(f"User {current_user.username} attempted to deactivate themselves")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot deactivate your own account"
                )

        # Store original state for logging
        original_is_active = user.is_active
        was_deactivated = False

        # Update fields
        if user_data.email is not None:
            logger.info(f"Updating email for user {user.username}: {user.email} -> {user_data.email}")
            user.email = user_data.email

        if user_data.is_active is not None:
            logger.info(f"Updating is_active for user {user.username}: {user.is_active} -> {user_data.is_active}")
            user.is_active = user_data.is_active
            # Check if user was deactivated
            if original_is_active and not user_data.is_active:
                was_deactivated = True
                logger.warning(f"User {user.username} is being deactivated by admin {current_user.username}")

        if user_data.password is not None:
            logger.info(f"Updating password for user {user.username}")
            user.password_hash = hash_password(user_data.password)

        # Update roles
        if user_data.roles is not None:
            old_roles = [r.name for r in user.roles] if user.roles else []
            logger.info(f"Updating roles for user {user.username}: {old_roles} -> {user_data.roles}")
            roles = db.query(Role).filter(Role.name.in_(user_data.roles)).all()
            user.roles = roles

        user.updated_at = datetime.now(UTC)

        # If user was deactivated, revoke all their tokens
        if was_deactivated:
            user.token_version += 1
            logger.warning(f"Revoked all tokens for deactivated user {user.username} (new version: {user.token_version})")

        db.commit()
        db.refresh(user)

        logger.info(f"User {user.username} (ID: {user.id}) updated successfully by admin {current_user.username}")

        # Extract role names
        roles_list = [role.name for role in user.roles] if user.roles else []

        return UserSchema(
            id=user.id,
            username=user.username,
            email=user.email,
            roles=roles_list,
            is_active=user.is_active,
            last_login=user.last_login,
            created_at=user.created_at,
            updated_at=user.updated_at
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        ) from e


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete user",
    description="Delete a user account (Admin only)",
)
async def delete_user(
    user_id: int,
    current_user: User = current_user_dependency,
    db: Session = db_dependency,
):
    """
    Delete a user.

    Args:
        user_id: User ID to delete
        current_user: Authenticated admin user
        db: Database session
    """
    # Check if user is admin
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can delete users"
        )

    try:
        # Get user
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )

        # Prevent deleting self
        if user.id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )

        db.delete(user)
        db.commit()

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting user: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        ) from e


@router.post(
    "/{user_id}/password",
    summary="Change user password",
    description="Change a user's password (Admin only)",
)
async def change_user_password(
    user_id: int,
    new_password: Annotated[str, Form(min_length=6, description="New password for the user")],
    current_user: User = current_user_dependency,
    db: Session = db_dependency,
):
    """
    Change user password.

    Args:
        user_id: User ID
        new_password: New password
        current_user: Authenticated admin user
        db: Database session

    Returns:
        Success response
    """
    # Check if user is admin
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can change user passwords"
        )

    try:
        # Get user
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )

        # Update password
        user.password_hash = hash_password(new_password)
        
        # Increment token version to invalidate all existing tokens
        user.token_version = (user.token_version or 0) + 1
        user.updated_at = datetime.now(UTC)

        db.commit()

        return {
            "user_id": user.id,
            "username": user.username,
            "changed": True,
            "revoked_tokens": True
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error changing password: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        ) from e


@router.post(
    "/{user_id}/revoke",
    summary="Revoke user tokens",
    description="Revoke all tokens for a user by incrementing token version (Admin only)",
)
async def revoke_user_tokens(
    user_id: int,
    current_user: User = current_user_dependency,
    db: Session = db_dependency,
):
    """
    Revoke all tokens for a user.

    Args:
        user_id: User ID
        current_user: Authenticated admin user
        db: Database session

    Returns:
        Success response with new token version
    """
    # Check if user is admin
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can revoke user tokens"
        )

    try:
        # Get user
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )

        # Increment token version
        user.token_version = (user.token_version or 0) + 1
        user.updated_at = datetime.now(UTC)

        db.commit()

        return {
            "user_id": user.id,
            "username": user.username,
            "revoked": True,
            "token_version": user.token_version
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error revoking tokens: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke tokens"
        ) from e
