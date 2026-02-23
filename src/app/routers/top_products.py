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
from app.dependencies.authz import get_current_user, is_user_admin
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
# Request Models for Creating Top Products
# ============================================================================


class TopProductMeasurementCreate(BaseModel):
    """Measurement data for creating a top product."""

    test_item: str = Field(..., description="Name of the test item")
    usl: float | None = Field(None, description="Upper Spec Limit")
    lsl: float | None = Field(None, description="Lower Spec Limit")
    target_value: float | None = Field(None, description="Target value")
    actual_value: float | None = Field(None, description="Actual measured value")
    deviation: float | None = Field(None, description="Deviation from target")


class TopProductCreate(BaseModel):
    """Request body for creating a top product."""

    dut_isn: str = Field(..., description="DUT ISN")
    dut_id: int | None = Field(None, description="DUT ID from database")
    site_name: str | None = Field(None, description="Site name")
    project_name: str | None = Field(None, description="Project name")
    model_name: str | None = Field(None, description="Model name")
    station_name: str = Field(..., description="Test station name")
    device_name: str | None = Field(None, description="Device name")
    test_date: datetime | None = Field(None, description="Test date and time")
    test_duration: float | None = Field(None, description="Test duration in seconds")
    pass_count: int = Field(0, description="Number of passed tests")
    fail_count: int = Field(0, description="Number of failed tests")
    retest_count: int = Field(0, description="Number of retests")
    score: float = Field(..., description="Overall score (0-10)")
    measurements: list[TopProductMeasurementCreate] = Field(default=[], description="Test measurements")


class TopProductBulkCreate(BaseModel):
    """Request body for bulk creating top products."""

    products: list[TopProductCreate] = Field(..., description="List of top products to create")


class TopProductCreateResponse(BaseModel):
    """Response for creating a top product."""

    success: bool = Field(..., description="Whether the operation was successful")
    id: int = Field(..., description="ID of the created top product")
    message: str = Field(..., description="Result message")


class TopProductBulkCreateResponse(BaseModel):
    """Response for bulk creating top products."""

    success: bool = Field(..., description="Whether the operation was successful")
    created_count: int = Field(..., description="Number of top products created")
    created_ids: list[int] = Field(..., description="IDs of created top products")
    message: str = Field(..., description="Result message")


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
            measurement_count = db.query(func.count(TopProductMeasurement.id)).filter(TopProductMeasurement.top_product_id == product.id).scalar()

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
        measurements = db.query(TopProductMeasurement).filter(TopProductMeasurement.top_product_id == product_id).all()

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

        recent_24h = db.query(func.count(TopProduct.id)).filter(TopProduct.created_at >= twenty_four_hours_ago).scalar() or 0

        recent_7d = db.query(func.count(TopProduct.id)).filter(TopProduct.created_at >= seven_days_ago).scalar() or 0

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
        projects = db.query(TopProduct.project_name).filter(TopProduct.project_name.isnot(None)).distinct().order_by(TopProduct.project_name).all()
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
        stations = db.query(TopProduct.station_name, TopProduct.project_name).distinct().order_by(TopProduct.station_name, TopProduct.project_name).all()

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

            options.append(StationOption(value=station_name, label=label, project=project))

        return options
    except Exception as e:
        logger.error(f"Error fetching stations: {e}", exc_info=True)
        raise


# ============================================================================
# Create Endpoints
# ============================================================================


@router.post(
    "/create",
    response_model=TopProductCreateResponse,
    summary="Create a new top product",
    description="Create a new top product entry with measurements from analyzed test log data",
)
async def create_top_product(
    product_data: TopProductCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create a new top product entry.

    Args:
        product_data: Top product data to create
        current_user: Authenticated user
        db: Database session

    Returns:
        TopProductCreateResponse with the created product ID
    """
    try:
        # Create the top product
        new_product = TopProduct(
            dut_isn=product_data.dut_isn,
            dut_id=product_data.dut_id,
            site_name=product_data.site_name,
            project_name=product_data.project_name,
            model_name=product_data.model_name,
            station_name=product_data.station_name,
            device_name=product_data.device_name,
            test_date=product_data.test_date,
            test_duration=product_data.test_duration,
            pass_count=product_data.pass_count,
            fail_count=product_data.fail_count,
            retest_count=product_data.retest_count,
            score=product_data.score,
        )
        db.add(new_product)
        db.flush()  # Get the ID without committing

        # Create measurements
        for measurement in product_data.measurements:
            new_measurement = TopProductMeasurement(
                top_product_id=new_product.id,
                test_item=measurement.test_item,
                usl=measurement.usl,
                lsl=measurement.lsl,
                target_value=measurement.target_value,
                actual_value=measurement.actual_value,
                deviation=measurement.deviation,
            )
            db.add(new_measurement)

        db.commit()

        logger.info(f"Top product created with ID {new_product.id} by user {current_user.username}")
        return TopProductCreateResponse(
            success=True,
            id=new_product.id,
            message=f"Top product created successfully for ISN {product_data.dut_isn}",
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Error creating top product: {e}", exc_info=True)
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=f"Failed to create top product: {str(e)}")


@router.post(
    "/create-bulk",
    response_model=TopProductBulkCreateResponse,
    summary="Create multiple top products",
    description="Create multiple top product entries in a single request",
)
async def create_top_products_bulk(
    bulk_data: TopProductBulkCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create multiple top product entries in bulk.

    Args:
        bulk_data: List of top products to create
        current_user: Authenticated user
        db: Database session

    Returns:
        TopProductBulkCreateResponse with created product IDs
    """
    try:
        created_ids = []

        for product_data in bulk_data.products:
            # Create the top product
            new_product = TopProduct(
                dut_isn=product_data.dut_isn,
                dut_id=product_data.dut_id,
                site_name=product_data.site_name,
                project_name=product_data.project_name,
                model_name=product_data.model_name,
                station_name=product_data.station_name,
                device_name=product_data.device_name,
                test_date=product_data.test_date,
                test_duration=product_data.test_duration,
                pass_count=product_data.pass_count,
                fail_count=product_data.fail_count,
                retest_count=product_data.retest_count,
                score=product_data.score,
            )
            db.add(new_product)
            db.flush()  # Get the ID without committing

            # Create measurements
            for measurement in product_data.measurements:
                new_measurement = TopProductMeasurement(
                    top_product_id=new_product.id,
                    test_item=measurement.test_item,
                    usl=measurement.usl,
                    lsl=measurement.lsl,
                    target_value=measurement.target_value,
                    actual_value=measurement.actual_value,
                    deviation=measurement.deviation,
                )
                db.add(new_measurement)

            created_ids.append(new_product.id)

        db.commit()

        logger.info(f"{len(created_ids)} top products created by user {current_user.username}")
        return TopProductBulkCreateResponse(
            success=True,
            created_count=len(created_ids),
            created_ids=created_ids,
            message=f"Successfully created {len(created_ids)} top product(s)",
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Error creating top products in bulk: {e}", exc_info=True)
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=f"Failed to create top products: {str(e)}")


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
        if not is_user_admin(current_user):
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


class BulkDeleteRequest(BaseModel):
    """Request body for bulk deleting top products."""

    ids: list[int] = Field(..., min_length=1, description="List of top product IDs to delete")


@router.delete(
    "/bulk-delete",
    summary="Bulk delete top products",
    description="Delete multiple top products and all their measurements. Superadmin/developer only.",
)
async def bulk_delete_top_products(
    request: BulkDeleteRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete multiple top products by IDs.

    Args:
        request: BulkDeleteRequest with list of product IDs
        current_user: Authenticated user (must be superadmin or developer)
        db: Database session

    Returns:
        Success message with count of deleted products
    """
    try:
        from fastapi import HTTPException

        # Check if user is superadmin or developer
        is_superadmin = getattr(current_user, "role", None) in ("superadmin", "developer")
        if not is_superadmin and not is_user_admin(current_user):
            raise HTTPException(
                status_code=403,
                detail="Only superadmin or developer users can bulk delete top products",
            )

        # Find existing products
        products = db.query(TopProduct).filter(TopProduct.id.in_(request.ids)).all()
        if not products:
            raise HTTPException(status_code=404, detail="No top products found with the given IDs")

        found_ids = [p.id for p in products]

        # Delete measurements first
        db.query(TopProductMeasurement).filter(
            TopProductMeasurement.top_product_id.in_(found_ids)
        ).delete(synchronize_session=False)

        # Delete the products
        db.query(TopProduct).filter(TopProduct.id.in_(found_ids)).delete(
            synchronize_session=False
        )
        db.commit()

        logger.info(
            f"Bulk deleted {len(found_ids)} top products by user {current_user.username}"
        )
        return {
            "message": f"Successfully deleted {len(found_ids)} top product(s)",
            "deleted_count": len(found_ids),
            "deleted_ids": found_ids,
        }

    except Exception as e:
        db.rollback()
        logger.error(f"Error bulk deleting top products: {e}", exc_info=True)
        raise
