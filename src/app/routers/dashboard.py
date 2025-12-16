"""
Router for dashboard statistics and metrics.
"""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi_cache.decorator import cache
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import get_db
from app.dependencies.authz import get_current_user
from app.models.top_product import TopProduct
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


# ============================================================================
# Response Models
# ============================================================================


class DashboardStatistics(BaseModel):
    """Dashboard statistics response."""

    total_top_products: int = Field(..., description="Total number of top product analyses stored")
    total_unique_duts: int = Field(..., description="Total number of unique DUTs analyzed")
    total_unique_projects: int = Field(..., description="Total number of unique projects")
    recent_analyses: int = Field(..., description="Number of analyses in the last 7 days")
    avg_score: float | None = Field(None, description="Average score across all top products")
    total_pass: int = Field(..., description="Total pass count across all analyses")
    total_fail: int = Field(..., description="Total fail count across all analyses")


class RecentActivity(BaseModel):
    """Recent activity item."""

    title: str = Field(..., description="Activity title")
    description: str = Field(..., description="Activity description")
    timestamp: datetime = Field(..., description="Activity timestamp")
    activity_type: str = Field(..., description="Activity type (upload, analysis, comparison)")
    dut_isn: str | None = Field(None, description="DUT ISN if applicable")
    station_name: str | None = Field(None, description="Station name if applicable")


class SystemStatus(BaseModel):
    """System status information."""

    cache_status: str = Field(..., description="Cache connection status (Online/Offline)")
    cache_enabled: bool = Field(..., description="Whether caching is enabled")
    cache_total_keys: int | None = Field(None, description="Total number of cached keys")
    cache_hits: int | None = Field(None, description="Cache hit count")
    cache_misses: int | None = Field(None, description="Cache miss count")
    cache_hit_rate: float | None = Field(None, description="Cache hit rate percentage")
    cache_memory_mb: float | None = Field(None, description="Cache memory usage in MB")
    database_status: str = Field(..., description="Database connection status")
    database_records: int = Field(..., description="Total records in top_products table")
    total_users: int = Field(..., description="Total number of registered users")
    active_users: int = Field(..., description="Number of active users")
    api_version: str = Field(..., description="Backend API version")
    uptime_hours: float | None = Field(None, description="System uptime in hours")


class DashboardResponse(BaseModel):
    """Complete dashboard data response."""

    statistics: DashboardStatistics
    recent_activities: list[RecentActivity]
    system_status: SystemStatus


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "/stats",
    response_model=DashboardResponse,
    summary="Get dashboard statistics and data",
    description="Retrieve comprehensive dashboard data including statistics, recent activities, and system status",
)
@cache(expire=30)  # Cache for 30 seconds (frequently changing as new analyses arrive)
async def get_dashboard_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Get dashboard statistics and recent activities.

    Args:
        current_user: Authenticated user
        db: Database session

    Returns:
        DashboardResponse with statistics, activities, and system status
    """
    try:
        # Calculate date for "recent" filter (7 days ago)
        seven_days_ago = datetime.now(UTC) - timedelta(days=7)
        thirty_days_ago = datetime.now(UTC) - timedelta(days=30)

        # Get statistics from TopProduct table
        total_top_products = db.query(func.count(TopProduct.id)).scalar() or 0
        total_unique_duts = db.query(func.count(func.distinct(TopProduct.dut_isn))).scalar() or 0
        total_unique_projects = db.query(func.count(func.distinct(TopProduct.project_name))).scalar() or 0

        # Recent analyses (last 7 days)
        recent_analyses = (
            db.query(func.count(TopProduct.id))
            .filter(TopProduct.created_at >= seven_days_ago)
            .scalar() or 0
        )

        # Average score
        avg_score_result = db.query(func.avg(TopProduct.score)).scalar()
        avg_score = float(avg_score_result) if avg_score_result else None

        # Total pass/fail counts
        total_pass = db.query(func.sum(TopProduct.pass_count)).scalar() or 0
        total_fail = db.query(func.sum(TopProduct.fail_count)).scalar() or 0

        statistics = DashboardStatistics(
            total_top_products=total_top_products,
            total_unique_duts=total_unique_duts,
            total_unique_projects=total_unique_projects,
            recent_analyses=recent_analyses,
            avg_score=avg_score,
            total_pass=total_pass,
            total_fail=total_fail,
        )

        # Get recent activities (last 6 top product analyses for dashboard preview)
        recent_top_products = (
            db.query(TopProduct)
            .filter(TopProduct.created_at >= thirty_days_ago)
            .order_by(TopProduct.created_at.desc())
            .limit(6)
            .all()
        )

        recent_activities = []
        for product in recent_top_products:
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

            activity = RecentActivity(
                title=f"Top Product Analysis - {status}",
                description=description,
                timestamp=product.created_at,
                activity_type="top_product_analysis",
                dut_isn=product.dut_isn,
                station_name=product.station_name,
            )
            recent_activities.append(activity)

        # Get system status
        # Check cache status
        cache_enabled = False
        cache_status = "Offline"
        cache_total_keys = None
        cache_hits = None
        cache_misses = None
        cache_hit_rate = None
        cache_memory_mb = None
        try:
            from app.utils.dut_cache import get_cache_stats

            cache_stats = get_cache_stats()
            cache_enabled = cache_stats.get("enabled", False)
            
            if cache_enabled and "error" not in cache_stats:
                cache_status = "Online"
                cache_total_keys = cache_stats.get("total_keys")
                cache_hits = cache_stats.get("hits")
                cache_misses = cache_stats.get("misses")
                cache_hit_rate = cache_stats.get("hit_rate")
                cache_memory_mb = cache_stats.get("memory_used_mb")
            else:
                cache_status = "Offline" if not cache_enabled else "Error"
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            cache_status = "Offline"

        # Get user statistics
        total_users = db.query(func.count(User.id)).scalar() or 0
        active_users = db.query(func.count(User.id)).filter(User.is_active == True).scalar() or 0  # noqa: E712

        # Get API version
        api_version = "1.0.0"  # You can read this from a config file or version module

        # Calculate uptime (placeholder - you'd track this with app start time)
        uptime_hours = None

        system_status = SystemStatus(
            cache_status=cache_status,
            cache_enabled=cache_enabled,
            cache_total_keys=cache_total_keys,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_hit_rate=cache_hit_rate,
            cache_memory_mb=cache_memory_mb,
            database_status="Online",
            database_records=total_top_products,
            total_users=total_users,
            active_users=active_users,
            api_version=api_version,
            uptime_hours=uptime_hours,
        )

        return DashboardResponse(
            statistics=statistics,
            recent_activities=recent_activities,
            system_status=system_status,
        )

    except Exception as e:
        logger.error(f"Error fetching dashboard stats: {e}", exc_info=True)
        raise


@router.get(
    "/uploads/stats",
    summary="Get upload directory statistics",
    description="Get statistics about uploaded files",
)
@cache(expire=60)  # Cache for 1 minute (changes less frequently)
async def get_upload_stats(
    current_user: User = Depends(get_current_user),
):
    """
    Get statistics about uploaded test log files.

    Args:
        current_user: Authenticated user

    Returns:
        Upload statistics
    """
    try:
        upload_dir = Path("data/uploads/test_logs")
        
        if not upload_dir.exists():
            return {
                "upload_dir_exists": False,
                "total_files": 0,
                "total_size_mb": 0.0,
                "recent_uploads": 0,
            }

        # Count files
        all_files = list(upload_dir.glob("**/*"))
        files_only = [f for f in all_files if f.is_file()]
        total_files = len(files_only)

        # Calculate total size
        total_size_bytes = sum(f.stat().st_size for f in files_only)
        total_size_mb = total_size_bytes / (1024 * 1024)

        # Count recent uploads (last 7 days)
        seven_days_ago = datetime.now(UTC) - timedelta(days=7)
        recent_uploads = sum(
            1 for f in files_only
            if datetime.fromtimestamp(f.stat().st_mtime, tz=UTC) >= seven_days_ago
        )

        return {
            "upload_dir_exists": True,
            "total_files": total_files,
            "total_size_mb": round(total_size_mb, 2),
            "recent_uploads": recent_uploads,
        }

    except Exception as e:
        logger.error(f"Error fetching upload stats: {e}", exc_info=True)
        return {
            "upload_dir_exists": False,
            "total_files": 0,
            "total_size_mb": 0.0,
            "recent_uploads": 0,
            "error": str(e),
        }
