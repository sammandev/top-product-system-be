"""
Database initialization helper.
"""

from app.db import Base, engine


def init_db() -> None:
    """
    Create database tables for all registered models.
    """
    Base.metadata.create_all(bind=engine)
