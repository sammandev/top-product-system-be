"""
Quick test script to verify dashboard endpoint works.
Run with: python scripts/test_dashboard.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app.db import SessionLocal
from app.models.top_product import TopProduct
from app.models.user import User
from sqlalchemy import func


async def test_dashboard_queries():
    """Test the database queries used in dashboard endpoint."""
    db = SessionLocal()
    
    try:
        print("=" * 60)
        print("Dashboard Database Queries Test")
        print("=" * 60)
        
        # Test statistics queries
        print("\n📊 Statistics:")
        total_top_products = db.query(func.count(TopProduct.id)).scalar() or 0
        print(f"  Total Top Products: {total_top_products}")
        
        total_unique_duts = db.query(func.count(func.distinct(TopProduct.dut_isn))).scalar() or 0
        print(f"  Unique DUTs: {total_unique_duts}")
        
        total_unique_stations = db.query(func.count(func.distinct(TopProduct.station_name))).scalar() or 0
        print(f"  Unique Stations: {total_unique_stations}")
        
        avg_score_result = db.query(func.avg(TopProduct.score)).scalar()
        avg_score = float(avg_score_result) if avg_score_result else None
        print(f"  Average Score: {avg_score:.2f}" if avg_score else "  Average Score: N/A")
        
        total_pass = db.query(func.sum(TopProduct.pass_count)).scalar() or 0
        print(f"  Total Pass: {total_pass}")
        
        total_fail = db.query(func.sum(TopProduct.fail_count)).scalar() or 0
        print(f"  Total Fail: {total_fail}")
        
        # Test recent activities query
        print("\n🕒 Recent Activities:")
        recent_top_products = (
            db.query(TopProduct)
            .order_by(TopProduct.created_at.desc())
            .limit(5)
            .all()
        )
        
        if recent_top_products:
            for i, product in enumerate(recent_top_products, 1):
                print(f"  {i}. DUT {product.dut_isn} @ {product.station_name} - Score: {product.score:.2f}")
        else:
            print("  No activities found")
        
        # Test user queries
        print("\n👥 User Statistics:")
        total_users = db.query(func.count(User.id)).scalar() or 0
        print(f"  Total Users: {total_users}")
        
        active_users = db.query(func.count(User.id)).filter(User.is_active == True).scalar() or 0  # noqa: E712
        print(f"  Active Users: {active_users}")
        
        print("\n✅ All queries executed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(test_dashboard_queries())
