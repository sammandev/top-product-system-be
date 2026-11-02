"""
Menu Access Model for Role-Based Menu Visibility Control.

Stores which menu items are accessible to which roles.
"""

from sqlalchemy import Boolean, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class MenuDefinition(Base):
    """
    Defines all available menu items in the application.
    This acts as a master list of menu routes.
    """
    __tablename__ = "menu_definitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    menu_key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)  # e.g., 'dashboard', 'top-products-analysis'
    title: Mapped[str] = mapped_column(String(128), nullable=False)  # Display title
    path: Mapped[str] = mapped_column(String(256), nullable=False)  # Route path e.g., '/dashboard'
    icon: Mapped[str | None] = mapped_column(String(64), default="mdi-circle-small")
    parent_key: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)  # For nested menus
    section: Mapped[str] = mapped_column(String(32), nullable=False, default="main")  # 'main', 'tools', 'system'
    sort_order: Mapped[int | None] = mapped_column(Integer, default=0)  # For ordering menus
    is_active: Mapped[bool | None] = mapped_column(Boolean, default=True)  # Can disable entire menu items
    description: Mapped[str | None] = mapped_column(String(256), nullable=True)  # Optional description

    # Relationships
    role_access: Mapped[list["MenuRoleAccess"]] = relationship("MenuRoleAccess", back_populates="menu", cascade="all, delete-orphan")


class MenuRoleAccess(Base):
    """
    Junction table to control which roles can access which menus.
    """
    __tablename__ = "menu_role_access"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    menu_id: Mapped[int] = mapped_column(Integer, ForeignKey("menu_definitions.id", ondelete="CASCADE"), nullable=False)
    role_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)  # Role name (including 'guest', 'user', 'admin')
    can_view: Mapped[bool | None] = mapped_column(Boolean, default=True)  # Permission to view the menu
    
    # Relationships
    menu: Mapped[MenuDefinition] = relationship("MenuDefinition", back_populates="role_access")
    
    __table_args__ = (
        UniqueConstraint("menu_id", "role_name", name="uq_menu_role"),
    )
