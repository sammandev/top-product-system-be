"""
Router for Top Product database management and viewing.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from fastapi_cache.decorator import cache
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.db import get_db
from app.dependencies.authz import get_current_user
from app.models.top_product import TopProduct, TopProductMeasurement
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/top-products", tags=["Top_Products_Database"])


# ============================================================================
# Response Models
# ============================================================================


class TopProductMeasurementSchema(BaseModel):
    """Top product measurement schema."""

    id: int
    test_item: str
    usl: float | None
    lsl: float | None
    target_value: float | None
    actual_value: float | None
    deviation: float | None

    model_config = ConfigDict(from_attributes=True)


class TopProductSchema(BaseModel):
    """Top product schema."""

    id: int
    dut_isn: str
    dut_id: int | None
    site_name: str | None
    project_name: str | None
    model_name: str | None
    station_name: str
    device_name: str | None
    test_date: datetime | None
    test_duration: float | None
    pass_count: int
    fail_count: int
    retest_count: int
    score: float
    created_at: datetime
    measurements_count: int | None = Field(None, description="Number of measurements")

    model_config = ConfigDict(from_attributes=True)


class TopProductDetailSchema(TopProductSchema):
    """Top product with measurements."""

    measurements: list[TopProductMeasurementSchema]


class TopProductListResponse(BaseModel):
    """Top product list response."""

    total: int = Field(..., description="Total number of top products")
    top_products: list[TopProductSchema] = Field(..., description="List of top products")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    total_pages: int = Field(..., description="Total number of pages")


class TopProductStatsResponse(BaseModel):
    """Top product statistics."""

    total_products: int  # Rename in frontend to "Total Analysis"
    total_unique_isns: int  # New: Unique DUT ISNs
    total_projects: int
    avg_score: float | None
    max_score: float | None
    min_score: float | None
    total_pass: int
    total_fail: int
    recent_products_24h: int
    recent_products_7d: int


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "/list",
    response_model=TopProductListResponse,
    summary="Get paginated top products list",
    description="Retrieve all top product analyses with pagination and filtering",
)
@cache(expire=60)  # Cache for 1 minute (moderately stable)
async def get_top_products_list(
    page: int = Query(1, ge=1, description="Page number (starts at 1)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of items per page (max 100)"),
    stations: list[str] | None = Query(None, description="Filter by station names (multiple allowed)"),
    projects: list[str] | None = Query(None, description="Filter by project names (multiple allowed)"),
    dut_isn: str | None = Query(None, description="Filter by DUT ISN"),
    min_score: float | None = Query(None, ge=0, le=10, description="Minimum score filter"),
    sort_by: str = Query("created_at", description="Sort field: created_at, score, dut_isn, station_name, project_name, test_date, pass_count, fail_count"),
    sort_desc: bool = Query(True, description="Sort in descending order"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get paginated list of top products.

    Args:
        page: Page number
        page_size: Items per page
        station: Filter by station name
        dut_isn: Filter by DUT ISN
        min_score: Minimum score filter
        sort_by: Sort field
        sort_desc: Sort descending
        current_user: Authenticated user
        db: Database session

    Returns:
        TopProductListResponse with paginated data
    """
    try:
        # Build query
        query = db.query(TopProduct)

        # Apply filters
        if stations:
            query = query.filter(TopProduct.station_name.in_(stations))
        if projects:
            query = query.filter(TopProduct.project_name.in_(projects))
        if dut_isn:
            query = query.filter(TopProduct.dut_isn.ilike(f"%{dut_isn}%"))
        if min_score is not None:
            query = query.filter(TopProduct.score >= min_score)

        # Count total
        total = query.count()

        # Apply sorting
        sort_column = getattr(TopProduct, sort_by, TopProduct.created_at)
        if sort_desc:
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(sort_column)

        # Apply pagination
        offset = (page - 1) * page_size
        total_pages = (total + page_size - 1) // page_size

        top_products = query.limit(page_size).offset(offset).all()

        # Add measurements count for each product
        products_with_count = []
        for product in top_products:
            measurement_count = db.query(func.count(TopProductMeasurement.id)).filter(
                TopProductMeasurement.top_product_id == product.id
            ).scalar()

            product_dict = {
                "id": product.id,
                "dut_isn": product.dut_isn,
                "dut_id": product.dut_id,
                "site_name": product.site_name,
                "project_name": product.project_name,
                "model_name": product.model_name,
                "station_name": product.station_name,
                "device_name": product.device_name,
                "test_date": product.test_date,
                "test_duration": product.test_duration,
                "pass_count": product.pass_count,
                "fail_count": product.fail_count,
                "retest_count": product.retest_count,
                "score": product.score,
                "created_at": product.created_at,
                "measurements_count": measurement_count,
            }
            products_with_count.append(TopProductSchema(**product_dict))

        return TopProductListResponse(
            total=total,
            top_products=products_with_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    except Exception as e:
        logger.error(f"Error fetching top products list: {e}", exc_info=True)
        raise


@router.get(
    "/{product_id}",
    response_model=TopProductDetailSchema,
    summary="Get top product details",
    description="Retrieve detailed information about a specific top product including all measurements",
)
@cache(expire=120)  # Cache for 2 minutes (stable historical data)
async def get_top_product_detail(
    product_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get detailed top product information.

    Args:
        product_id: Top product ID
        current_user: Authenticated user
        db: Database session

    Returns:
        TopProductDetailSchema with measurements
    """
    try:
        product = db.query(TopProduct).filter(TopProduct.id == product_id).first()
        
        if not product:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Top product not found")

        # Get measurements
        measurements = (
            db.query(TopProductMeasurement)
            .filter(TopProductMeasurement.top_product_id == product_id)
            .all()
        )

        measurement_count = len(measurements)

        product_dict = {
            "id": product.id,
            "dut_isn": product.dut_isn,
            "dut_id": product.dut_id,
            "site_name": product.site_name,
            "project_name": product.project_name,
            "model_name": product.model_name,
            "station_name": product.station_name,
            "device_name": product.device_name,
            "test_date": product.test_date,
            "test_duration": product.test_duration,
            "pass_count": product.pass_count,
            "fail_count": product.fail_count,
            "retest_count": product.retest_count,
            "score": product.score,
            "created_at": product.created_at,
            "measurements_count": measurement_count,
            "measurements": measurements,
        }

        return TopProductDetailSchema(**product_dict)

    except Exception as e:
        logger.error(f"Error fetching top product detail: {e}", exc_info=True)
        raise


@router.get(
    "/stats/summary",
    response_model=TopProductStatsResponse,
    summary="Get top products statistics",
    description="Retrieve aggregate statistics for top products",
)
@cache(expire=30)  # Cache for 30 seconds (frequently updated)
async def get_top_products_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get aggregate statistics about top products.

    Args:
        current_user: Authenticated user
        db: Database session

    Returns:
        TopProductStatsResponse with statistics
    """
    try:
        from datetime import timedelta

        now = datetime.now()
        twenty_four_hours_ago = now - timedelta(hours=24)
        seven_days_ago = now - timedelta(days=7)

        total_products = db.query(func.count(TopProduct.id)).scalar() or 0
        total_unique_isns = db.query(func.count(func.distinct(TopProduct.dut_isn))).scalar() or 0
        total_projects = db.query(func.count(func.distinct(TopProduct.project_name))).filter(TopProduct.project_name.isnot(None)).scalar() or 0

        avg_score = db.query(func.avg(TopProduct.score)).scalar()
        max_score = db.query(func.max(TopProduct.score)).scalar()
        min_score = db.query(func.min(TopProduct.score)).scalar()

        total_pass = db.query(func.sum(TopProduct.pass_count)).scalar() or 0
        total_fail = db.query(func.sum(TopProduct.fail_count)).scalar() or 0

        recent_24h = (
            db.query(func.count(TopProduct.id))
            .filter(TopProduct.created_at >= twenty_four_hours_ago)
            .scalar() or 0
        )

        recent_7d = (
            db.query(func.count(TopProduct.id))
            .filter(TopProduct.created_at >= seven_days_ago)
            .scalar() or 0
        )

        return TopProductStatsResponse(
            total_products=total_products,
            total_unique_isns=total_unique_isns,
            total_projects=total_projects,
            avg_score=float(avg_score) if avg_score else None,
            max_score=float(max_score) if max_score else None,
            min_score=float(min_score) if min_score else None,
            total_pass=total_pass,
            total_fail=total_fail,
            recent_products_24h=recent_24h,
            recent_products_7d=recent_7d,
        )

    except Exception as e:
        logger.error(f"Error fetching top products stats: {e}", exc_info=True)
        raise


class ProjectOption(BaseModel):
    """Project option for filter dropdown."""
    value: str
    label: str


class StationOption(BaseModel):
    """Station option for filter dropdown with project association."""
    value: str
    label: str
    project: str | None


@router.get(
    "/filters/projects",
    response_model=list[ProjectOption],
    summary="Get unique projects",
    description="Get list of unique projects for filter dropdown",
)
@cache(expire=300)  # Cache for 5 minutes (very stable metadata)
async def get_unique_projects(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get unique project names."""
    try:
        projects = (
            db.query(TopProduct.project_name)
            .filter(TopProduct.project_name.isnot(None))
            .distinct()
            .order_by(TopProduct.project_name)
            .all()
        )
        return [ProjectOption(value=p[0], label=p[0]) for p in projects]
    except Exception as e:
        logger.error(f"Error fetching projects: {e}", exc_info=True)
        raise


@router.get(
    "/filters/stations",
    response_model=list[StationOption],
    summary="Get unique stations",
    description="Get list of unique stations with their project associations",
)
@cache(expire=300)  # Cache for 5 minutes (very stable metadata)
async def get_unique_stations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get unique station names with project associations."""
    try:
        # Get unique station-project combinations
        stations = (
            db.query(
                TopProduct.station_name,
                TopProduct.project_name
            )
            .distinct()
            .order_by(TopProduct.station_name, TopProduct.project_name)
            .all()
        )
        
        # Group by station and collect projects
        station_map = {}
        for station_name, project_name in stations:
            if station_name not in station_map:
                station_map[station_name] = []
            if project_name:
                station_map[station_name].append(project_name)
        
        # Create options with project info in label
        options = []
        for station_name in sorted(station_map.keys()):
            projects = station_map[station_name]
            if projects:
                # Use the most common project or first one
                project = projects[0] if len(projects) == 1 else projects[0]
                label = f"{station_name} ({project})"
            else:
                project = None
                label = station_name
            
            options.append(StationOption(
                value=station_name,
                label=label,
                project=project
            ))
        
        return options
    except Exception as e:
        logger.error(f"Error fetching stations: {e}", exc_info=True)
        raise


@router.delete(
    "/{product_id}",
    summary="Delete a top product",
    description="Delete a top product and all its measurements. Admin only.",
)
async def delete_top_product(
    product_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete a top product by ID.
    
    Args:
        product_id: ID of the product to delete
        current_user: Authenticated user (must be admin)
        db: Database session
        
    Returns:
        Success message
    """
    try:
        # Check if user is admin
        if not current_user.is_admin:
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Only admins can delete top products")
        
        # Find the product
        product = db.query(TopProduct).filter(TopProduct.id == product_id).first()
        if not product:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Top product not found")
        
        # Delete measurements first (cascade should handle this, but being explicit)
        db.query(TopProductMeasurement).filter(TopProductMeasurement.top_product_id == product_id).delete()
        
        # Delete the product
        db.delete(product)
        db.commit()
        
        logger.info(f"Top product {product_id} deleted by user {current_user.username}")
        return {"message": "Top product deleted successfully", "id": product_id}
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting top product {product_id}: {e}", exc_info=True)
        raise

