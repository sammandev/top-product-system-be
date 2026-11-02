from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _utc_now():
    return datetime.now(UTC)


class AppConfig(Base):
    __tablename__ = "app_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    tab_title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    favicon_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    primeui_license_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now, onupdate=_utc_now, nullable=False)
    updated_by: Mapped[str | None] = mapped_column(String(150), nullable=True)


class IplasToken(Base):
    """Stores iPLAS API tokens per site with encryption."""

    __tablename__ = "iplas_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    site: Mapped[str] = mapped_column(String(10), nullable=False, index=True)  # PTB, PSZ, PXD, PVN, PTY
    base_url: Mapped[str] = mapped_column(String(500), nullable=False)
    token_value: Mapped[str] = mapped_column(Text, nullable=False)  # AES-encrypted
    label: Mapped[str | None] = mapped_column(String(200), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now, onupdate=_utc_now, nullable=False)
    updated_by: Mapped[str | None] = mapped_column(String(150), nullable=True)


class SfistspConfig(Base):
    """Stores SFISTSP API configuration with encryption for sensitive fields."""

    __tablename__ = "sfistsp_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    base_url: Mapped[str] = mapped_column(String(500), nullable=False)
    program_id: Mapped[str] = mapped_column(String(200), nullable=False)
    program_password: Mapped[str] = mapped_column(Text, nullable=False)  # AES-encrypted
    timeout: Mapped[float] = mapped_column(Float, default=30.0, nullable=False)
    label: Mapped[str | None] = mapped_column(String(200), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now, onupdate=_utc_now, nullable=False)
    updated_by: Mapped[str | None] = mapped_column(String(150), nullable=True)


class GuestCredential(Base):
    """Stores guest login credentials with encryption."""

    __tablename__ = "guest_credentials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(Text, nullable=False)  # AES-encrypted
    password: Mapped[str] = mapped_column(Text, nullable=False)  # AES-encrypted
    label: Mapped[str | None] = mapped_column(String(200), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_now, onupdate=_utc_now, nullable=False)
    updated_by: Mapped[str | None] = mapped_column(String(150), nullable=True)
