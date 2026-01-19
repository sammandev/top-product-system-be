from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Integer, String

from app.db import Base


def _utc_now():
    return datetime.now(UTC)


class AppConfig(Base):
    __tablename__ = "app_config"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    version = Column(String(64), nullable=False)
    description = Column(String(500), nullable=True)
    updated_at = Column(DateTime, default=_utc_now, onupdate=_utc_now, nullable=False)
    updated_by = Column(String(150), nullable=True)
