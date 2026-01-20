"""
Menu Access Model for Role-Based Menu Visibility Control.

Stores which menu items are accessible to which roles.
"""

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db import Base


class MenuDefinition(Base):
    """
    Defines all available menu items in the application.
    This acts as a master list of menu routes.
    """
    __tablename__ = "menu_definitions"
    
    id = Column(Integer, primary_key=True)
    menu_key = Column(String(100), unique=True, nullable=False, index=True)  # e.g., 'dashboard', 'top-products-analysis'
    title = Column(String(128), nullable=False)  # Display title
    path = Column(String(256), nullable=False)  # Route path e.g., '/dashboard'
    icon = Column(String(64), default="mdi-circle-small")
    parent_key = Column(String(100), nullable=True, index=True)  # For nested menus
    section = Column(String(32), nullable=False, default="main")  # 'main', 'tools', 'system'
    sort_order = Column(Integer, default=0)  # For ordering menus
    is_active = Column(Boolean, default=True)  # Can disable entire menu items
    description = Column(Text, nullable=True)
    
    # Relationships
    role_access = relationship("MenuRoleAccess", back_populates="menu", cascade="all, delete-orphan")


class MenuRoleAccess(Base):
    """
    Junction table to control which roles can access which menus.
    """
    __tablename__ = "menu_role_access"
    
    id = Column(Integer, primary_key=True)
    menu_id = Column(Integer, ForeignKey("menu_definitions.id", ondelete="CASCADE"), nullable=False)
    role_name = Column(String(64), nullable=False, index=True)  # Role name (including 'guest', 'user', 'admin')
    can_view = Column(Boolean, default=True)  # Permission to view the menu
    
    # Relationships
    menu = relationship("MenuDefinition", back_populates="role_access")
    
    __table_args__ = (
        UniqueConstraint("menu_id", "role_name", name="uq_menu_role"),
    )
