# Health check endpoint configuration
# Add to your FastAPI main.py or create a dedicated health router

from datetime import datetime

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.db import get_db

router = APIRouter(tags=["Health"])


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        - status: overall health status
        - timestamp: current server time
        - database: database connection status
        - version: API version
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ast-tools-backend",
        "version": "1.0.0",
    }
    
    # Check database connection
    try:
        db = next(get_db())
        db.execute(text("SELECT 1"))
        health_status["database"] = "connected"
    except Exception as e:
        health_status["database"] = "disconnected"
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health_status
        )
    finally:
        db.close()
    
    return health_status


@router.get("/readiness", status_code=status.HTTP_200_OK)
async def readiness_check():
    """
    Readiness check for Kubernetes/Docker orchestration.
    Returns 200 if the service is ready to accept traffic.
    """
    return {"ready": True}


@router.get("/liveness", status_code=status.HTTP_200_OK)
async def liveness_check():
    """
    Liveness check for Kubernetes/Docker orchestration.
    Returns 200 if the service is alive.
    """
    return {"alive": True}
