"""Cached test items model for iPLAS test item name caching."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class CachedTestItem(Base):
    """
    Stores cached test item names from iPLAS.
    
    This table reduces load time for the station configuration dialog
    by caching unique test item names for each site/project/station combination.
    
    Cache strategy:
    - Key: site + project + station (date range NOT included since test items rarely change)
    - TTL: Managed at application level (default 7 days)
    - Refresh: Manual via "Refresh Test Items" button or when cache expires
    """
    
    __tablename__ = "cached_test_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    site: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    project: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    station: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    test_item_name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_value: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_bin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    has_ucl: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    has_lcl: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<CachedTestItem {self.site}/{self.project}/{self.station}: {self.test_item_name}>"
