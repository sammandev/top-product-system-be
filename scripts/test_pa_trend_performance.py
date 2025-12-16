#!/usr/bin/env python3
"""
Performance test for /api/dut/pa/trend endpoint.

Compares request duration with external DUT API to identify bottlenecks.
"""

import asyncio
import time

import httpx

# Configuration
BACKEND_URL = "http://localhost:8001"
EXTERNAL_API_URL = "http://172.18.220.56:9001"

# Test payload
TEST_PAYLOAD = {
    "start_time": "2025-11-17T00:27:37Z",
    "end_time": "2025-11-17T06:27:37Z",
    "model": "",
    "station_id": 145,
    "test_items": [
        "WiFi_PA1_SROM_OLD_6015_11AX_MCS9_B20",
        "WiFi_PA1_SROM_NEW_6015_11AX_MCS9_B20",
        "WiFi_PA2_SROM_OLD_6015_11AX_MCS9_B20",
        "WiFi_PA2_SROM_NEW_6015_11AX_MCS9_B20",
        "WiFi_PA3_SROM_OLD_6015_11AX_MCS9_B20",
        "WiFi_PA3_SROM_NEW_6015_11AX_MCS9_B20",
        "WiFi_PA4_SROM_OLD_6015_11AX_MCS9_B20",
        "WiFi_PA4_SROM_NEW_6015_11AX_MCS9_B20",
    ],
}


async def measure_endpoint_latency(url: str, token: str, runs: int = 5) -> dict:
    """Measure endpoint latency over multiple runs."""
    latencies = []

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(runs):
            start = time.perf_counter()
            try:
                response = await client.post(url, json=TEST_PAYLOAD, headers=headers)
                elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

                if response.status_code == 200:
                    latencies.append(elapsed)
                    print(f"  Run {i+1}: {elapsed:.2f}ms (status: {response.status_code})")
                else:
                    print(f"  ⚠️  Run {i+1}: HTTP {response.status_code}")
                    print(f"       Response: {response.text[:200]}")
            except Exception as e:
                print(f"  ❌ Run {i+1}: {e}")

    if not latencies:
        return {"error": "All requests failed"}

    from statistics import mean, stdev
    return {
        "runs": len(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "avg_ms": mean(latencies),
        "std_ms": stdev(latencies) if len(latencies) > 1 else 0,
        "latencies": latencies,
    }


async def test_performance():
    """Test performance comparison between backend and external API."""
    print("=" * 70)
    print("🔬 PA Trend Endpoint Performance Test")
    print("=" * 70)

    # Get tokens
    backend_token = input("\nEnter Backend JWT token: ").strip()
    external_token = input("Enter External API JWT token: ").strip()

    # Test external API (baseline)
    print("\n📊 External DUT API Performance (Baseline):")
    print("-" * 70)
    external_url = f"{EXTERNAL_API_URL}/api/api/testitems/PA/trend"
    external_results = await measure_endpoint_latency(external_url, external_token, runs=5)

    if "error" not in external_results:
        print("\n  📈 Results:")
        print(f"     Average: {external_results['avg_ms']:.2f}ms")
        print(f"     Min: {external_results['min_ms']:.2f}ms")
        print(f"     Max: {external_results['max_ms']:.2f}ms")
        print(f"     Std Dev: {external_results['std_ms']:.2f}ms")

    # Test backend API
    print("\n📊 Backend API Performance:")
    print("-" * 70)
    backend_url = f"{BACKEND_URL}/api/dut/pa/trend"
    backend_results = await measure_endpoint_latency(backend_url, backend_token, runs=5)

    if "error" not in backend_results:
        print("\n  📈 Results:")
        print(f"     Average: {backend_results['avg_ms']:.2f}ms")
        print(f"     Min: {backend_results['min_ms']:.2f}ms")
        print(f"     Max: {backend_results['max_ms']:.2f}ms")
        print(f"     Std Dev: {backend_results['std_ms']:.2f}ms")

    # Compare
    print("\n🏁 Performance Comparison:")
    print("-" * 70)
    if "error" not in external_results and "error" not in backend_results:
        external_avg = external_results['avg_ms']
        backend_avg = backend_results['avg_ms']
        overhead = backend_avg - external_avg
        overhead_pct = (overhead / external_avg) * 100

        print(f"  External API (baseline): {external_avg:.2f}ms")
        print(f"  Backend API: {backend_avg:.2f}ms")
        print(f"  Overhead: {overhead:.2f}ms ({overhead_pct:.1f}%)")

        if overhead_pct > 50:
            print(f"  ⚠️  WARNING: Backend is {overhead_pct:.1f}% slower than baseline!")
        elif overhead_pct > 20:
            print(f"  ⚡ Backend overhead is acceptable ({overhead_pct:.1f}%)")
        else:
            print(f"  ✅ Backend performance is excellent ({overhead_pct:.1f}% overhead)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(test_performance())
