import asyncio
import json
from contextlib import aclosing, asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from redis import Redis

from app.routers import iplas_proxy
from app.routers.iplas_proxy import (
    StationBucketMetadata,
    StationBucketState,
    _fetch_chunked_from_iplas,
    _iter_bounded_in_order,
    _run_singleflight,
    _select_iplas_cache_ttl,
    _write_station_bucket,
)
from app.schemas.iplas_schemas import IplasCsvTestItemRequest, IplasDeviceListRequest

NOW = datetime(2026, 11, 5, 12, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _ttl_settings(monkeypatch):
    monkeypatch.setattr(iplas_proxy, "IPLAS_HISTORICAL_AGE_SECONDS", 6 * 3600)
    monkeypatch.setattr(iplas_proxy, "IPLAS_CACHE_TTL_HISTORICAL", 3600)
    monkeypatch.setattr(iplas_proxy, "IPLAS_CACHE_TTL_HISTORICAL_EMPTY", 300)


# --- A2: time-window-aware TTL ---------------------------------------------


def test_recent_window_keeps_recent_ttl():
    assert _select_iplas_cache_ttl(NOW - timedelta(hours=1), 180, now=NOW) == 180
    assert _select_iplas_cache_ttl(NOW - timedelta(hours=1), 15, is_empty=True, now=NOW) == 15


def test_historical_window_uses_long_ttl():
    query_end = NOW - timedelta(hours=7)

    assert _select_iplas_cache_ttl(query_end, 180, now=NOW) == 3600
    assert _select_iplas_cache_ttl(query_end, 15, is_empty=True, now=NOW) == 300


def test_historical_threshold_boundary_and_unknown_end():
    assert _select_iplas_cache_ttl(NOW - timedelta(hours=6), 180, now=NOW) == 3600
    assert _select_iplas_cache_ttl(NOW - timedelta(hours=6) + timedelta(seconds=1), 180, now=NOW) == 180
    assert _select_iplas_cache_ttl(None, 180, now=NOW) == 180


def test_historical_ttl_never_shortens_and_naive_end_is_utc():
    assert _select_iplas_cache_ttl(NOW - timedelta(days=2), 86400, now=NOW) == 86400
    naive_end = (NOW - timedelta(hours=7)).replace(tzinfo=None)
    assert _select_iplas_cache_ttl(naive_end, 180, now=NOW) == 3600


class _RecordingPipeline:
    def __init__(self) -> None:
        self.ttls: list[int] = []

    def set(self, _key, _value, ex):
        self.ttls.append(ex)

    def hset(self, _key, mapping):
        return None

    def expire(self, _key, ttl):
        self.ttls.append(ttl)

    def execute(self):
        return None


def _bucket_metadata(state: StationBucketState, bucket_end: datetime, record_count: int) -> StationBucketMetadata:
    return StationBucketMetadata(
        site="PTB",
        project="HH5K",
        station="Wireless",
        device_scope="ALL",
        test_status="ALL",
        bucket_start=bucket_end - timedelta(hours=1),
        bucket_end=bucket_end,
        state=state,
        validated_until=None,
        latest_record_time=None,
        record_count=record_count,
        fetched_at=NOW,
    )


@pytest.mark.parametrize(
    ("state", "age", "record_count", "expected"),
    [
        ("partial", timedelta(hours=1), 3, iplas_proxy.IPLAS_STATION_BUCKET_PARTIAL_TTL),
        ("partial", timedelta(hours=8), 3, 3600),
        ("partial", timedelta(hours=8), 0, 300),
        ("hot", timedelta(minutes=5), 3, iplas_proxy.IPLAS_STATION_BUCKET_HOT_TTL),
        ("complete", timedelta(days=3), 3, iplas_proxy.IPLAS_STATION_BUCKET_COMPLETE_TTL),
    ],
)
def test_station_bucket_ttl_extends_only_historical_partial(state, age, record_count, expected):
    pipe = _RecordingPipeline()
    redis = Mock(spec=Redis)
    redis.pipeline.return_value = pipe
    _write_station_bucket(
        redis,
        scope_hash="scope",
        scope_label="label",
        metadata=_bucket_metadata(state, NOW - age, record_count),
        records=[],
    )

    assert set(pipe.ttls) == {expected}


# --- A3: in-flight request dedupe -------------------------------------------


@pytest.mark.asyncio
async def test_singleflight_shares_one_upstream_call():
    calls = 0
    release = asyncio.Event()

    async def fetch():
        nonlocal calls
        calls += 1
        await release.wait()
        return {"value": calls}

    waiters = [asyncio.create_task(_run_singleflight("test", "key", fetch)) for _ in range(5)]
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(*waiters)

    assert calls == 1
    assert all(result is results[0] for result in results)
    assert not iplas_proxy._in_flight_requests

    # Once settled, the next miss starts a fresh upstream call.
    await _run_singleflight("test", "key", fetch)
    assert calls == 2


@pytest.mark.asyncio
async def test_singleflight_kinds_do_not_share_results():
    calls: list[str] = []

    async def fetch(kind: str):
        calls.append(kind)
        await asyncio.sleep(0)
        return kind

    first, second = await asyncio.gather(
        _run_singleflight("stations", "key", lambda: fetch("stations")),
        _run_singleflight("project-stations", "key", lambda: fetch("project-stations")),
    )

    assert (first, second) == ("stations", "project-stations")
    assert sorted(calls) == ["project-stations", "stations"]


@pytest.mark.asyncio
async def test_singleflight_error_reaches_every_waiter_and_clears_entry():
    calls = 0
    release = asyncio.Event()

    async def fetch():
        nonlocal calls
        calls += 1
        await release.wait()
        raise HTTPException(status_code=502, detail="upstream down")

    waiters = [asyncio.create_task(_run_singleflight("test", "error", fetch)) for _ in range(3)]
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.wait_for(asyncio.gather(*waiters, return_exceptions=True), timeout=1)

    assert calls == 1
    assert all(isinstance(result, HTTPException) and result.status_code == 502 for result in results)
    assert not iplas_proxy._in_flight_requests


@pytest.mark.asyncio
async def test_singleflight_cancelling_one_waiter_keeps_shared_fetch_alive():
    started = asyncio.Event()
    release = asyncio.Event()
    cancelled = False

    async def fetch():
        nonlocal cancelled
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled = True
            raise
        return "ok"

    first = asyncio.create_task(_run_singleflight("test", "cancel", fetch))
    second = asyncio.create_task(_run_singleflight("test", "cancel", fetch))
    await started.wait()

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()

    assert await second == "ok"
    assert cancelled is False

    release.clear()
    started.clear()
    only = asyncio.create_task(_run_singleflight("test", "cancel", fetch))
    await started.wait()
    only.cancel()
    with pytest.raises(asyncio.CancelledError):
        await only
    for _ in range(5):
        await asyncio.sleep(0)

    assert cancelled is True
    assert not iplas_proxy._in_flight_requests


@pytest.mark.asyncio
async def test_concurrent_identical_device_requests_share_one_upstream_call(monkeypatch):
    upstream_calls = 0
    release = asyncio.Event()

    class StubResponse:
        status_code = 200

        @staticmethod
        def json():
            return ["DEV-1"]

    class StubClient:
        async def get(self, *_args, **_kwargs):
            nonlocal upstream_calls
            upstream_calls += 1
            await release.wait()
            return StubResponse()

    @asynccontextmanager
    async def pooled_client():
        yield StubClient()

    async def no_redis():
        return None

    monkeypatch.setattr(iplas_proxy, "_get_redis_client_async", no_redis)
    monkeypatch.setattr(iplas_proxy, "_pooled_client", pooled_client)
    monkeypatch.setattr(
        iplas_proxy,
        "_get_site_config",
        lambda _site, _token: {"v2_url": "http://iplas.test", "token": "configured"},
    )
    end_time = datetime.now(UTC)
    request = IplasDeviceListRequest(
        site="PTB",
        project="HH5K",
        station="Wireless_Test_2_5G",
        start_time=end_time - timedelta(hours=1),
        end_time=end_time,
    )

    tasks = [asyncio.create_task(iplas_proxy.get_devices(request)) for _ in range(4)]
    await asyncio.sleep(0.01)
    release.set()
    responses = await asyncio.gather(*tasks)

    assert upstream_calls == 1
    assert all(response.data == ["DEV-1"] for response in responses)


# --- A4: bounded, order-preserving chunk fetches ------------------------------


@pytest.mark.asyncio
async def test_bounded_iterator_preserves_order_and_caps_concurrency():
    active = 0
    max_active = 0

    async def work(index: int) -> int:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        # Later items finish first to prove ordering is enforced.
        await asyncio.sleep(0.002 * (10 - index))
        active -= 1
        return index

    factories = [lambda index=index: work(index) for index in range(10)]
    results = [result async for result in _iter_bounded_in_order(factories, 3)]

    assert results == list(range(10))
    assert max_active == 3


@pytest.mark.asyncio
async def test_bounded_iterator_cancels_pending_on_early_close():
    cancelled: list[int] = []
    completed: list[int] = []

    async def work(index: int) -> int:
        if index > 0:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(index)
                raise
        completed.append(index)
        return index

    factories = [lambda index=index: work(index) for index in range(5)]
    async with aclosing(_iter_bounded_in_order(factories, 3)) as results:
        async for first in results:
            assert first == 0
            break

    # Started chunks are cancelled; the not-yet-started one never runs.
    assert sorted(cancelled) == [1, 2]
    assert completed == [0]


def _install_chunk_fetch(monkeypatch, *, fail: dict[int, int] | None = None, delay=lambda index: 0.002 * (10 - index)):
    stats = {"active": 0, "max_active": 0, "calls": 0}
    cancelled: list[int] = []
    fail = fail or {}

    async def fake_fetch(site, project, station, device_id, begin_time, end_time, test_status, user_token):
        index = stats["calls"]
        stats["calls"] += 1
        stats["active"] += 1
        stats["max_active"] = max(stats["max_active"], stats["active"])
        try:
            # Later chunks finish first by default to prove output order is enforced.
            await asyncio.sleep(delay(index))
            if index in fail:
                raise HTTPException(status_code=fail[index], detail=f"chunk {index} failed")
            return [{"ISN": f"chunk-{index}", "Test Start Time": begin_time.strftime("%Y/%m/%d %H:%M:%S")}]
        except asyncio.CancelledError:
            cancelled.append(index)
            raise
        finally:
            stats["active"] -= 1

    monkeypatch.setattr(iplas_proxy, "_fetch_from_iplas", fake_fetch)
    monkeypatch.setattr(iplas_proxy, "MAX_DAYS_PER_CHUNK", 1)
    return stats, cancelled


@pytest.mark.asyncio
async def test_chunked_fetch_runs_concurrently_in_chunk_order(monkeypatch):
    stats, _ = _install_chunk_fetch(monkeypatch)
    begin = datetime(2026, 1, 1, tzinfo=UTC)

    records, truncated, chunk_count, _total, _hybrid = await _fetch_chunked_from_iplas(
        "PTB", "HH5K", "Wireless", "DEV-1", begin, begin + timedelta(days=8), "ALL", chunk_concurrency=3
    )

    assert [record["ISN"] for record in records] == [f"chunk-{index}" for index in range(8)]
    assert chunk_count == 8
    assert truncated is False
    assert stats["max_active"] == 3


@pytest.mark.asyncio
async def test_chunked_fetch_skips_failed_chunk_and_aborts_on_auth_error(monkeypatch):
    _install_chunk_fetch(monkeypatch, fail={2: 500})
    begin = datetime(2026, 1, 1, tzinfo=UTC)

    records, *_ = await _fetch_chunked_from_iplas("PTB", "HH5K", "Wireless", "DEV-1", begin, begin + timedelta(days=5), "ALL", chunk_concurrency=3)
    assert [record["ISN"] for record in records] == ["chunk-0", "chunk-1", "chunk-3", "chunk-4"]

    _, cancelled = _install_chunk_fetch(monkeypatch, fail={0: 401}, delay=lambda index: 0 if index == 0 else 10)
    with pytest.raises(HTTPException) as exc_info:
        await asyncio.wait_for(
            _fetch_chunked_from_iplas("PTB", "HH5K", "Wireless", "DEV-1", begin, begin + timedelta(days=5), "ALL", chunk_concurrency=3),
            timeout=2,
        )
    assert exc_info.value.status_code == 401
    # In-flight chunks are cancelled rather than leaked.
    assert sorted(cancelled) == [1, 2]


@pytest.mark.asyncio
async def test_stream_csv_test_items_yields_chunks_in_order_with_cap(monkeypatch):
    stats, _ = _install_chunk_fetch(monkeypatch)
    monkeypatch.setattr(iplas_proxy, "IPLAS_CSV_CHUNK_CONCURRENCY", 2)

    async def no_redis():
        return None

    monkeypatch.setattr(iplas_proxy, "_get_redis_client_async", no_redis)
    begin = datetime(2026, 1, 1, tzinfo=UTC)
    request = IplasCsvTestItemRequest(
        site="PTB",
        project="HH5K",
        station="Wireless",
        device_id="DEV-1",
        begin_time=begin,
        end_time=begin + timedelta(days=6),
    )

    response = await iplas_proxy.stream_csv_test_items(request)
    lines = [json.loads(bytes(line) if isinstance(line, memoryview) else line) async for line in response.body_iterator]

    assert lines[0]["_metadata"] is True
    assert lines[0]["total_records"] == 6
    assert [line["ISN"] for line in lines[1:]] == [f"chunk-{index}" for index in range(6)]
    assert stats["max_active"] == 2


@pytest.mark.asyncio
async def test_csv_cache_write_uses_historical_ttl(monkeypatch):
    _install_chunk_fetch(monkeypatch)
    observed: list[int] = []

    redis = Mock(spec=Redis)
    redis.setex.side_effect = lambda _key, ttl, _value: observed.append(ttl)

    begin = datetime.now(UTC) - timedelta(days=3)
    await iplas_proxy._fetch_and_cache_csv_test_items(redis, "csv-key", "PTB", "HH5K", "Wireless", "DEV-1", begin, begin + timedelta(days=2), "ALL", None)

    assert observed == [3600]
