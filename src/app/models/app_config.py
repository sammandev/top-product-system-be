from datetime import UTC, datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text

from app.db import Base


def _utc_now():
    return datetime.now(UTC)


class AppConfig(Base):
    __tablename__ = "app_config"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    version = Column(String(64), nullable=False)
    description = Column(String(500), nullable=True)
    tab_title = Column(String(200), nullable=True)
    favicon_path = Column(String(500), nullable=True)
    updated_at = Column(DateTime, default=_utc_now, onupdate=_utc_now, nullable=False)
    updated_by = Column(String(150), nullable=True)


class IplasToken(Base):
    """Stores iPLAS API tokens per site with encryption."""

    __tablename__ = "iplas_tokens"

    id = Column(Integer, primary_key=True, index=True)
    site = Column(String(10), nullable=False, index=True)  # PTB, PSZ, PXD, PVN, PTY
    base_url = Column(String(500), nullable=False)
    token_value = Column(Text, nullable=False)  # AES-encrypted
    label = Column(String(200), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=_utc_now, nullable=False)
    updated_at = Column(DateTime, default=_utc_now, onupdate=_utc_now, nullable=False)
    updated_by = Column(String(150), nullable=True)


class SfistspConfig(Base):
    """Stores SFISTSP API configuration with encryption for sensitive fields."""

    __tablename__ = "sfistsp_configs"

    id = Column(Integer, primary_key=True, index=True)
    base_url = Column(String(500), nullable=False)
    program_id = Column(String(200), nullable=False)
    program_password = Column(Text, nullable=False)  # AES-encrypted
    timeout = Column(Float, default=30.0, nullable=False)
    label = Column(String(200), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=_utc_now, nullable=False)
    updated_at = Column(DateTime, default=_utc_now, onupdate=_utc_now, nullable=False)
    updated_by = Column(String(150), nullable=True)


class GuestCredential(Base):
    """Stores guest login credentials with encryption."""

    __tablename__ = "guest_credentials"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(Text, nullable=False)  # AES-encrypted
    password = Column(Text, nullable=False)  # AES-encrypted
    label = Column(String(200), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=_utc_now, nullable=False)
    updated_at = Column(DateTime, default=_utc_now, onupdate=_utc_now, nullable=False)
    updated_by = Column(String(150), nullable=True)
