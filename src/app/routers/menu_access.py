"""
Router for Menu Access Management.

Allows superadmin to control which menus are visible to which roles.
"""

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.dependencies.authz import get_current_user
from app.models.menu_access import MenuDefinition, MenuRoleAccess
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/menu-access", tags=["Menu_Access"])


# ============================================================================
# Helper Functions
# ============================================================================


def is_superadmin(user: User) -> bool:
    """Check if user is superadmin (Samuel_Halomoan or MW2400549)."""
    if not user:
        return False
    username_lower = (user.username or "").lower()
    worker_id_upper = (user.worker_id or "").upper()
    return username_lower == "samuel_halomoan" or worker_id_upper == "MW2400549"


# ============================================================================
# Request & Response Models
# ============================================================================


class MenuItemSchema(BaseModel):
    """Menu item schema."""
    id: int
    menu_key: str
    title: str
    path: str
    icon: str
    parent_key: str | None
    section: str
    sort_order: int
    is_active: bool
    description: str | None
    role_access: list[str]  # List of role names that can access

    model_config = ConfigDict(from_attributes=True)


class MenuListResponse(BaseModel):
    """Menu list response."""
    menus: list[MenuItemSchema]
    available_roles: list[str]


class MenuRoleAccessSchema(BaseModel):
    """Role access for a menu."""
    role_name: str
    can_view: bool


class UpdateMenuAccessRequest(BaseModel):
    """Request to update menu access for a role."""
    menu_key: str
    role_name: str
    can_view: bool


class BulkUpdateMenuAccessRequest(BaseModel):
    """Bulk update menu access."""
    updates: list[UpdateMenuAccessRequest]


class CreateMenuRequest(BaseModel):
    """Request to create a new menu definition."""
    menu_key: str = Field(..., min_length=1, max_length=100)
    title: str = Field(..., min_length=1, max_length=128)
    path: str = Field(..., min_length=1, max_length=256)
    icon: str = Field(default="mdi-circle-small", max_length=64)
    parent_key: str | None = Field(default=None, max_length=100)
    section: Literal["main", "tools", "system"] = "main"
    sort_order: int = Field(default=0, ge=0)
    description: str | None = None


class UserMenusResponse(BaseModel):
    """Response for user's accessible menus."""
    menus: list[dict]


# ============================================================================
# Default Menu Definitions
# ============================================================================


DEFAULT_MENUS = [
    # Main Section
    {"menu_key": "dashboard", "title": "Dashboard", "path": "/dashboard", "icon": "mdi-view-dashboard", "section": "main", "sort_order": 0},
    {"menu_key": "top-products", "title": "Top Products", "path": "", "icon": "mdi-trophy", "section": "main", "sort_order": 10},
    {"menu_key": "top-products-analysis", "title": "Analysis", "path": "/dut/top-products/analysis", "icon": "mdi-circle-small", "parent_key": "top-products", "section": "main", "sort_order": 11},
    {"menu_key": "top-products-data", "title": "Database", "path": "/dut/top-products/data", "icon": "mdi-circle-small", "parent_key": "top-products", "section": "main", "sort_order": 12},
    {"menu_key": "test-log-download", "title": "Test Log Download", "path": "/dut/test-log-download", "icon": "mdi-download-box", "section": "main", "sort_order": 20},
    {"menu_key": "data-explorer", "title": "Data Explorer", "path": "/dut/data-explorer", "icon": "mdi-database-search", "section": "main", "sort_order": 25},
    {"menu_key": "iplas-download", "title": "iPLAS Downloader", "path": "/iplas/download", "icon": "mdi-download", "section": "main", "sort_order": 31},
    
    # Tools Section
    {"menu_key": "file-upload", "title": "File Upload", "path": "", "icon": "mdi-file-upload", "section": "tools", "sort_order": 0},
    {"menu_key": "file-upload-upload", "title": "Upload File", "path": "/parsing", "icon": "mdi-circle-small", "parent_key": "file-upload", "section": "tools", "sort_order": 1},
    {"menu_key": "file-upload-download", "title": "Parse & Download", "path": "/parsing/download-format", "icon": "mdi-circle-small", "parent_key": "file-upload", "section": "tools", "sort_order": 2},
    {"menu_key": "compare-files", "title": "Compare Files", "path": "", "icon": "mdi-compare", "section": "tools", "sort_order": 10},
    {"menu_key": "compare-files-compare", "title": "Compare Files", "path": "/compare", "icon": "mdi-circle-small", "parent_key": "compare-files", "section": "tools", "sort_order": 11},
    {"menu_key": "compare-files-dvt-mc2", "title": "DVT-MC2 Compare", "path": "/compare/dvt-mc2", "icon": "mdi-circle-small", "parent_key": "compare-files", "section": "tools", "sort_order": 12},
    {"menu_key": "mastercontrol-analyze", "title": "MasterControl Analyze", "path": "/mastercontrol/analyze", "icon": "mdi-file-chart", "section": "tools", "sort_order": 20},
    {"menu_key": "dvt-to-mc2-converter", "title": "DVT to MC2 Converter", "path": "/conversion/dvt-to-mc2", "icon": "mdi-file-swap", "section": "tools", "sort_order": 30},
    
    # System Section (Admin only by default)
    {"menu_key": "access-control", "title": "Access Control", "path": "", "icon": "mdi-shield-lock", "section": "system", "sort_order": 0},
    {"menu_key": "access-control-users", "title": "User Management", "path": "/admin/users", "icon": "mdi-circle-small", "parent_key": "access-control", "section": "system", "sort_order": 1},
    {"menu_key": "access-control-rbac", "title": "Roles & Permissions", "path": "/admin/rbac", "icon": "mdi-circle-small", "parent_key": "access-control", "section": "system", "sort_order": 2},
    {"menu_key": "access-control-menus", "title": "Menu Access", "path": "/admin/menu-access", "icon": "mdi-circle-small", "parent_key": "access-control", "section": "system", "sort_order": 3},
    {"menu_key": "system-cleanup", "title": "System Cleanup", "path": "/admin/cleanup", "icon": "mdi-delete-sweep", "section": "system", "sort_order": 10},
    {"menu_key": "app-configuration", "title": "App Configuration", "path": "/admin/app-config", "icon": "mdi-cog", "section": "system", "sort_order": 20},
]


# Default role access: which roles can see which menus by default
DEFAULT_ROLE_ACCESS = {
    # Guest can see limited menus
    "guest": ["dashboard", "file-upload", "file-upload-upload", "file-upload-download", "compare-files", "compare-files-compare"],
    # Regular users can see most menus except system
    "user": [
        "dashboard", "top-products", "top-products-analysis", "top-products-data",
        "test-log-download", "data-explorer", "iplas-download",
        "file-upload", "file-upload-upload", "file-upload-download",
        "compare-files", "compare-files-compare", "compare-files-dvt-mc2",
        "mastercontrol-analyze", "dvt-to-mc2-converter"
    ],
    # Admin can see everything
    "admin": [m["menu_key"] for m in DEFAULT_MENUS],
}


# ============================================================================
# Endpoints
# ============================================================================


@router.post(
    "/initialize",
    status_code=status.HTTP_200_OK,
    summary="Initialize menu definitions",
    description="Initialize default menu definitions in the database (Superadmin only)",
)
async def initialize_menus(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Initialize or reset menu definitions to defaults."""
    if not is_superadmin(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only superadmin can initialize menu definitions"
        )

    try:
        # Create or update menu definitions
        for menu_data in DEFAULT_MENUS:
            existing = db.query(MenuDefinition).filter(
                MenuDefinition.menu_key == menu_data["menu_key"]
            ).first()
            
            if existing:
                # Update existing
                for key, value in menu_data.items():
                    setattr(existing, key, value)
            else:
                # Create new
                menu = MenuDefinition(**menu_data, is_active=True)
                db.add(menu)
        
        db.commit()
        
        # Initialize default role access
        for role_name, menu_keys in DEFAULT_ROLE_ACCESS.items():
            for menu_key in menu_keys:
                menu = db.query(MenuDefinition).filter(
                    MenuDefinition.menu_key == menu_key
                ).first()
                
                if menu:
                    existing_access = db.query(MenuRoleAccess).filter(
                        MenuRoleAccess.menu_id == menu.id,
                        MenuRoleAccess.role_name == role_name
                    ).first()
                    
                    if not existing_access:
                        access = MenuRoleAccess(
                            menu_id=menu.id,
                            role_name=role_name,
                            can_view=True
                        )
                        db.add(access)
        
        db.commit()
        
        return {"message": "Menu definitions initialized successfully", "count": len(DEFAULT_MENUS)}

    except Exception as e:
        db.rollback()
        logger.error(f"Error initializing menus: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize menu definitions"
        ) from e


@router.get(
    "/menus",
    response_model=MenuListResponse,
    summary="Get all menu definitions",
    description="Retrieve all menu definitions with role access (Superadmin only)",
)
async def get_all_menus(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get all menu definitions with their role access settings."""
    if not is_superadmin(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only superadmin can manage menu access"
        )

    try:
        menus = db.query(MenuDefinition).order_by(
            MenuDefinition.section, MenuDefinition.sort_order
        ).all()
        
        menu_list = []
        for menu in menus:
            role_names = [access.role_name for access in menu.role_access if access.can_view]
            menu_list.append(MenuItemSchema(
                id=menu.id,
                menu_key=menu.menu_key,
                title=menu.title,
                path=menu.path,
                icon=menu.icon,
                parent_key=menu.parent_key,
                section=menu.section,
                sort_order=menu.sort_order,
                is_active=menu.is_active,
                description=menu.description,
                role_access=role_names
            ))
        
        # Available roles
        available_roles = ["guest", "user", "admin"]
        
        return MenuListResponse(menus=menu_list, available_roles=available_roles)

    except Exception as e:
        logger.error(f"Error fetching menus: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch menu definitions"
        ) from e


@router.put(
    "/access",
    status_code=status.HTTP_200_OK,
    summary="Update menu access",
    description="Update menu access for a specific role (Superadmin only)",
)
async def update_menu_access(
    request: UpdateMenuAccessRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update menu access for a specific role."""
    if not is_superadmin(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only superadmin can update menu access"
        )

    try:
        menu = db.query(MenuDefinition).filter(
            MenuDefinition.menu_key == request.menu_key
        ).first()
        
        if not menu:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Menu '{request.menu_key}' not found"
            )
        
        # Find or create role access
        existing_access = db.query(MenuRoleAccess).filter(
            MenuRoleAccess.menu_id == menu.id,
            MenuRoleAccess.role_name == request.role_name
        ).first()
        
        if existing_access:
            if request.can_view:
                existing_access.can_view = True
            else:
                # Remove access entry if can_view is False
                db.delete(existing_access)
        elif request.can_view:
            # Create new access entry
            access = MenuRoleAccess(
                menu_id=menu.id,
                role_name=request.role_name,
                can_view=True
            )
            db.add(access)
        
        db.commit()
        
        return {"message": "Menu access updated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating menu access: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update menu access"
        ) from e


@router.put(
    "/access/bulk",
    status_code=status.HTTP_200_OK,
    summary="Bulk update menu access",
    description="Update multiple menu access entries at once (Superadmin only)",
)
async def bulk_update_menu_access(
    request: BulkUpdateMenuAccessRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Bulk update menu access."""
    if not is_superadmin(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only superadmin can update menu access"
        )

    try:
        updated_count = 0
        
        for update in request.updates:
            menu = db.query(MenuDefinition).filter(
                MenuDefinition.menu_key == update.menu_key
            ).first()
            
            if not menu:
                continue
            
            existing_access = db.query(MenuRoleAccess).filter(
                MenuRoleAccess.menu_id == menu.id,
                MenuRoleAccess.role_name == update.role_name
            ).first()
            
            if existing_access:
                if update.can_view:
                    existing_access.can_view = True
                else:
                    db.delete(existing_access)
                updated_count += 1
            elif update.can_view:
                access = MenuRoleAccess(
                    menu_id=menu.id,
                    role_name=update.role_name,
                    can_view=True
                )
                db.add(access)
                updated_count += 1
        
        db.commit()
        
        return {"message": f"Updated {updated_count} menu access entries"}

    except Exception as e:
        db.rollback()
        logger.error(f"Error bulk updating menu access: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to bulk update menu access"
        ) from e


@router.get(
    "/my-menus",
    response_model=UserMenusResponse,
    summary="Get current user's accessible menus",
    description="Get menus accessible to the current user based on their role",
)
async def get_user_menus(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get menus accessible to the current user."""
    try:
        # Determine user's role
        if is_superadmin(current_user):
            role = "admin"
        elif current_user.is_admin:
            role = "admin"
        elif hasattr(current_user, 'roles') and current_user.roles:
            # Get first role name, default to 'user'
            role = current_user.roles[0].name.lower() if current_user.roles else "user"
        else:
            role = "user"
        
        # Check for guest mode (passed via header or stored flag)
        # For now, we'll use the user role
        
        # Get all menus this role can access
        accessible_menus = db.query(MenuDefinition).join(
            MenuRoleAccess, MenuDefinition.id == MenuRoleAccess.menu_id
        ).filter(
            MenuRoleAccess.role_name == role,
            MenuRoleAccess.can_view == True,
            MenuDefinition.is_active == True
        ).order_by(
            MenuDefinition.section, MenuDefinition.sort_order
        ).all()
        
        # Also get admin menus if user is admin
        if role == "admin":
            accessible_menus = db.query(MenuDefinition).filter(
                MenuDefinition.is_active == True
            ).order_by(
                MenuDefinition.section, MenuDefinition.sort_order
            ).all()
        
        # Build menu structure
        menus = []
        for menu in accessible_menus:
            menus.append({
                "menu_key": menu.menu_key,
                "title": menu.title,
                "path": menu.path,
                "icon": menu.icon,
                "parent_key": menu.parent_key,
                "section": menu.section,
                "sort_order": menu.sort_order
            })
        
        return UserMenusResponse(menus=menus)

    except Exception as e:
        logger.error(f"Error fetching user menus: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user menus"
        ) from e


@router.get(
    "/guest-menus",
    response_model=UserMenusResponse,
    summary="Get guest accessible menus",
    description="Get menus accessible to guest users (no auth required for menu list)",
)
async def get_guest_menus(
    db: Session = Depends(get_db),
):
    """Get menus accessible to guest users."""
    try:
        # Get all menus that guests can access
        accessible_menus = db.query(MenuDefinition).join(
            MenuRoleAccess, MenuDefinition.id == MenuRoleAccess.menu_id
        ).filter(
            MenuRoleAccess.role_name == "guest",
            MenuRoleAccess.can_view == True,
            MenuDefinition.is_active == True
        ).order_by(
            MenuDefinition.section, MenuDefinition.sort_order
        ).all()
        
        menus = []
        for menu in accessible_menus:
            menus.append({
                "menu_key": menu.menu_key,
                "title": menu.title,
                "path": menu.path,
                "icon": menu.icon,
                "parent_key": menu.parent_key,
                "section": menu.section,
                "sort_order": menu.sort_order
            })
        
        return UserMenusResponse(menus=menus)

    except Exception as e:
        logger.error(f"Error fetching guest menus: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch guest menus"
        ) from e
