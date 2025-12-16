#!/usr/bin/env python3
"""
Cache Performance Verification Script

Tests the Redis caching system performance and validates cache hits/misses.
"""

import asyncio
import time
from statistics import mean, stdev

import httpx

BASE_URL = "http://localhost:8000"
ENDPOINTS = [
    "/api/dut/sites",
    "/api/dut/models",
    "/api/dut/stations",
]


async def measure_latency(client: httpx.AsyncClient, url: str, runs: int = 3) -> dict:
    """Measure endpoint latency over multiple runs."""
    latencies = []

    for i in range(runs):
        start = time.perf_counter()
        try:
            response = await client.get(url)
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

            if response.status_code == 200:
                latencies.append(elapsed)
            else:
                print(f"  ⚠️  Run {i+1}: HTTP {response.status_code}")
        except Exception as e:
            print(f"  ❌ Run {i+1}: {e}")

    if not latencies:
        return {"error": "All requests failed"}

    return {
        "runs": len(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "avg_ms": mean(latencies),
        "std_ms": stdev(latencies) if len(latencies) > 1 else 0,
        "latencies": latencies,
    }


async def test_cache_performance():
    """Test cache performance with warm-up and multiple measurements."""
    print("=" * 70)
    print("🚀 Redis Cache Performance Verification")
    print("=" * 70)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        # Test 1: Check cache stats endpoint
        print("\n📊 Cache Statistics:")
        print("-" * 70)
        try:
            response = await client.get("/api/dut/cache/stats")
            if response.status_code == 200:
                stats = response.json()
                print(f"  ✅ Enabled: {stats.get('enabled', 'N/A')}")
                print(f"  ✅ Redis Connected: {stats.get('redis_connected', 'N/A')}")
                print(f"  ✅ Cache Keys: {stats.get('cache_keys', 'N/A')}")
                print(f"  ✅ Hit Rate: {stats.get('hit_rate', 'N/A')}")
            else:
                print(f"  ⚠️  Stats endpoint returned: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

        # Test 2: Clear cache for fresh start
        print("\n🗑️  Clearing Cache:")
        print("-" * 70)
        try:
            response = await client.delete("/api/dut/cache/invalidate")
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Deleted {result.get('deleted', 0)} cache keys")
            else:
                print(f"  ⚠️  Invalidate endpoint returned: {response.status_code}")
        except Exception as e:
            print(f"  ⚠️  Could not clear cache: {e}")

        # Test 3: Measure cache miss (first call)
        print("\n❄️  Cache MISS Performance (First Call):")
        print("-" * 70)
        miss_results = {}
        for endpoint in ENDPOINTS:
            print(f"\n  Testing: {endpoint}")
            result = await measure_latency(client, endpoint, runs=1)
            miss_results[endpoint] = result
            if "error" not in result:
                print(f"    ⏱️  Latency: {result['avg_ms']:.2f}ms")
            else:
                print(f"    ❌ {result['error']}")

        # Small delay to ensure cache is set
        await asyncio.sleep(0.5)

        # Test 4: Measure cache hit (second call)
        print("\n⚡ Cache HIT Performance (Subsequent Calls):")
        print("-" * 70)
        hit_results = {}
        for endpoint in ENDPOINTS:
            print(f"\n  Testing: {endpoint}")
            result = await measure_latency(client, endpoint, runs=5)
            hit_results[endpoint] = result
            if "error" not in result:
                print(f"    ⏱️  Average: {result['avg_ms']:.2f}ms")
                print(f"    📉 Min: {result['min_ms']:.2f}ms")
                print(f"    📈 Max: {result['max_ms']:.2f}ms")
                print(f"    📊 Std Dev: {result['std_ms']:.2f}ms")
            else:
                print(f"    ❌ {result['error']}")

        # Test 5: Calculate speedup
        print("\n🏆 Performance Improvement:")
        print("-" * 70)
        for endpoint in ENDPOINTS:
            if endpoint in miss_results and endpoint in hit_results:
                miss = miss_results[endpoint]
                hit = hit_results[endpoint]

                if "error" not in miss and "error" not in hit:
                    miss_latency = miss['avg_ms']
                    hit_latency = hit['avg_ms']
                    speedup = miss_latency / hit_latency
                    saved = miss_latency - hit_latency

                    print(f"\n  {endpoint}")
                    print(f"    🐌 Cache Miss: {miss_latency:.2f}ms")
                    print(f"    ⚡ Cache Hit:  {hit_latency:.2f}ms")
                    print(f"    🚀 Speedup:    {speedup:.1f}x faster")
                    print(f"    💾 Saved:      {saved:.2f}ms per request")

        # Test 6: Check final cache stats
        print("\n📊 Final Cache Statistics:")
        print("-" * 70)
        try:
            response = await client.get("/api/dut/cache/stats")
            if response.status_code == 200:
                stats = response.json()
                print(f"  ✅ Cache Keys: {stats.get('cache_keys', 'N/A')}")
                print(f"  ✅ Hit Rate: {stats.get('hit_rate', 'N/A')}")
                print(f"  ✅ Memory Usage: {stats.get('memory_usage', 'N/A')}")
            else:
                print(f"  ⚠️  Stats endpoint returned: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n" + "=" * 70)
    print("✅ Cache Performance Verification Complete!")
    print("=" * 70)


async def quick_test():
    """Quick sanity check for cache functionality."""
    print("🔍 Quick Cache Test")
    print("-" * 70)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        endpoint = "/api/dut/sites"

        # Clear cache
        await client.delete("/api/dut/cache/invalidate")
        print("  ✅ Cache cleared")

        # First call (miss)
        start = time.perf_counter()
        r1 = await client.get(endpoint)
        t1 = (time.perf_counter() - start) * 1000
        print(f"  🐌 First call (miss):  {t1:.2f}ms")

        # Second call (hit)
        start = time.perf_counter()
        r2 = await client.get(endpoint)
        t2 = (time.perf_counter() - start) * 1000
        print(f"  ⚡ Second call (hit):  {t2:.2f}ms")

        # Check if data is same
        if r1.json() == r2.json():
            print("  ✅ Data consistency: PASS")
        else:
            print("  ❌ Data consistency: FAIL")

        # Calculate improvement
        if t1 > 0 and t2 > 0:
            speedup = t1 / t2
            print(f"  🚀 Speedup: {speedup:.1f}x faster")

        print("-" * 70)


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "quick":
        asyncio.run(quick_test())
    else:
        asyncio.run(test_cache_performance())
