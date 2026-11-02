from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class TopProduct(Base):
    __tablename__ = "top_products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dut_isn: Mapped[str] = mapped_column(String, nullable=False, index=True)
    dut_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    site_name: Mapped[str | None] = mapped_column(String, nullable=True)
    project_name: Mapped[str | None] = mapped_column(String, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String, nullable=True)
    station_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    device_name: Mapped[str | None] = mapped_column(String, nullable=True)
    test_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    test_duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    pass_count: Mapped[int | None] = mapped_column(Integer, default=0)
    fail_count: Mapped[int | None] = mapped_column(Integer, default=0)
    retest_count: Mapped[int | None] = mapped_column(Integer, default=0)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    measurements: Mapped[list[TopProductMeasurement]] = relationship(
        "TopProductMeasurement",
        back_populates="top_product",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class TopProductMeasurement(Base):
    __tablename__ = "top_product_measurements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    top_product_id: Mapped[int] = mapped_column(Integer, ForeignKey("top_products.id", ondelete="CASCADE"), nullable=False)
    test_item: Mapped[str] = mapped_column(String, nullable=False)
    usl: Mapped[float | None] = mapped_column(Float, nullable=True)
    lsl: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    deviation: Mapped[float | None] = mapped_column(Float, nullable=True)

    top_product: Mapped[TopProduct] = relationship("TopProduct", back_populates="measurements")
