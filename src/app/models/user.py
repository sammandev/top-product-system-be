from datetime import UTC, datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.orm import relationship

from app.db import Base
from app.models.rbac import user_roles


def _utc_now():
    """Helper function for SQLAlchemy default/onupdate."""
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), unique=True, index=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    email = Column(String(255), nullable=True)  # Optional email field
    worker_id = Column(String(64), nullable=True)  # External worker_id (from DUT API)
    is_admin = Column(Boolean, default=False)  # Local admin privileges (manually granted)
    is_ptb_admin = Column(Boolean, default=False)  # External PTB admin status (synced from external API)
    is_active = Column(Boolean, default=True)
    token_version = Column(Integer, default=1, nullable=False)
    last_login = Column(DateTime, nullable=True)  # Track last login time
    created_at = Column(DateTime, default=_utc_now, nullable=False)  # Track creation time
    updated_at = Column(DateTime, default=_utc_now, onupdate=_utc_now, nullable=False)  # Track update time
    roles = relationship("Role", secondary=user_roles, back_populates="users", lazy="selectin")
