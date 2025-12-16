"""
Router for detailed activity logs and history.
"""

import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from fastapi_cache.decorator import cache
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.dependencies.authz import get_current_user
from app.models.top_product import TopProduct
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard/activity", tags=["Dashboard"])


# ============================================================================
# Response Models
# ============================================================================


class ActivityItem(BaseModel):
    """Detailed activity item."""

    id: int = Field(..., description="Activity ID")
    title: str = Field(..., description="Activity title")
    description: str = Field(..., description="Activity description")
    timestamp: datetime = Field(..., description="Activity timestamp (when analysis was created)")
    activity_type: str = Field(..., description="Activity type")
    dut_isn: str | None = Field(None, description="DUT ISN if applicable")
    station_name: str | None = Field(None, description="Station name if applicable")
    device_name: str | None = Field(None, description="Device name if applicable")
    score: float | None = Field(None, description="Score if applicable")
    pass_count: int | None = Field(None, description="Pass count if applicable")
    fail_count: int | None = Field(None, description="Fail count if applicable")
    test_date: datetime | None = Field(None, description="Test date (when the actual test occurred)")


class ActivityListResponse(BaseModel):
    """Activity list response."""

    total: int = Field(..., description="Total number of activities")
    activities: list[ActivityItem] = Field(..., description="List of activities")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    total_pages: int = Field(..., description="Total number of pages")


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "/list",
    response_model=ActivityListResponse,
    summary="Get paginated activity list",
    description="Retrieve detailed activity history with pagination",
)
@cache(expire=60)  # Cache for 1 minute (moderately stable data)
async def get_activity_list(
    page: int = Query(1, ge=1, description="Page number (starts at 1)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of items per page (max 100)"),
    days: int | None = Query(None, ge=1, le=365, description="Number of days to look back (default 30, max 365)"),
    start_date: datetime | None = Query(None, description="Custom start date (UTC)"),
    end_date: datetime | None = Query(None, description="Custom end date (UTC)"),
    search: str | None = Query(None, description="Search in title, description, DUT ISN"),
    sort_order: str = Query("newest", description="Sort order: newest or oldest"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get paginated activity list.

    Args:
        page: Page number (starts at 1)
        page_size: Number of items per page
        days: Number of days to look back
        current_user: Authenticated user
        db: Database session

    Returns:
        ActivityListResponse with paginated activities
    """
    try:
        # Calculate date range
        if start_date and end_date:
            # Custom date range
            filter_start = start_date
            filter_end = end_date
        elif days:
            # Days filter
            filter_end = datetime.now(UTC)
            filter_start = filter_end - timedelta(days=days)
        else:
            # Default to 30 days
            filter_end = datetime.now(UTC)
            filter_start = filter_end - timedelta(days=30)

        # Build query
        query = db.query(TopProduct).filter(
            TopProduct.created_at >= filter_start,
            TopProduct.created_at <= filter_end
        )

        # Apply search filter
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                (TopProduct.dut_isn.ilike(search_pattern)) |
                (TopProduct.station_name.ilike(search_pattern)) |
                (TopProduct.project_name.ilike(search_pattern))
            )

        # Count total activities
        total = query.count()

        # Apply sorting
        if sort_order == "oldest":
            query = query.order_by(TopProduct.created_at.asc())
        else:  # newest
            query = query.order_by(TopProduct.created_at.desc())

        # Calculate pagination
        offset = (page - 1) * page_size
        total_pages = (total + page_size - 1) // page_size

        # Get activities
        top_products = query.limit(page_size).offset(offset).all()

        # Format activities
        activities = []
        for product in top_products:
            # Determine activity title and description
            if product.pass_count > 0 and product.fail_count == 0:
                status = "PASS"
                description = f"DUT {product.dut_isn} tested at {product.station_name} - All tests passed (Score: {product.score:.2f})"
            elif product.fail_count > 0:
                status = "FAIL"
                description = f"DUT {product.dut_isn} tested at {product.station_name} - {product.fail_count} failures (Score: {product.score:.2f})"
            else:
                status = "ANALYZED"
                description = f"DUT {product.dut_isn} analyzed at {product.station_name} (Score: {product.score:.2f})"

            activity = ActivityItem(
                id=product.id,
                title=f"Top Product Analysis - {status}",
                description=description,
                timestamp=product.created_at,
                activity_type="top_product_analysis",
                dut_isn=product.dut_isn,
                station_name=product.station_name,
                device_name=product.device_name,
                score=product.score,
                pass_count=product.pass_count,
                fail_count=product.fail_count,
                test_date=product.test_date,
            )
            activities.append(activity)

        return ActivityListResponse(
            total=total,
            activities=activities,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    except Exception as e:
        logger.error(f"Error fetching activity list: {e}", exc_info=True)
        raise
