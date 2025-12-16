from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import relationship

from app.db import Base


class TopProduct(Base):
    __tablename__ = "top_products"

    id = Column(Integer, primary_key=True, index=True)
    dut_isn = Column(String, nullable=False, index=True)
    dut_id = Column(Integer, nullable=True)
    site_name = Column(String, nullable=True)
    project_name = Column(String, nullable=True)
    model_name = Column(String, nullable=True)
    station_name = Column(String, nullable=False, index=True)
    device_name = Column(String, nullable=True)
    test_date = Column(DateTime(timezone=True), nullable=True)
    test_duration = Column(Float, nullable=True)
    pass_count = Column(Integer, default=0)
    fail_count = Column(Integer, default=0)
    retest_count = Column(Integer, default=0)
    score = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    measurements = relationship(
        "TopProductMeasurement",
        back_populates="top_product",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class TopProductMeasurement(Base):
    __tablename__ = "top_product_measurements"

    id = Column(Integer, primary_key=True, index=True)
    top_product_id = Column(Integer, ForeignKey("top_products.id", ondelete="CASCADE"), nullable=False)
    test_item = Column(String, nullable=False)
    usl = Column(Float, nullable=True)
    lsl = Column(Float, nullable=True)
    target_value = Column(Float, nullable=True)
    actual_value = Column(Float, nullable=True)
    deviation = Column(Float, nullable=True)

    top_product = relationship("TopProduct", back_populates="measurements")
