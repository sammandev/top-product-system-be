import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

from app.services import dut_token_service


def test_expired_token_refresh_is_coalesced_per_user(monkeypatch):
    refresh_calls = 0

    class StubResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"access": "new-access"}

    def refresh(_url, *, json, timeout):
        nonlocal refresh_calls
        refresh_calls += 1
        assert json == {"refresh": "refresh-token"}
        assert timeout == 30
        time.sleep(0.03)
        return StubResponse()

    monkeypatch.setattr(dut_token_service, "_redis_available", False)
    monkeypatch.setattr(dut_token_service, "_FILE_CACHE_ENABLED", False)
    monkeypatch.setattr(dut_token_service.requests, "post", refresh)
    with dut_token_service._local_lock:
        dut_token_service._local_store["tester"] = {
            "access": "old-access",
            "refresh": "refresh-token",
            "expiry": (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
        }

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(dut_token_service.ensure_valid_access, ["tester"] * 5))

    assert results == ["new-access"] * 5
    assert refresh_calls == 1
    with dut_token_service._local_lock:
        dut_token_service._local_store.pop("tester", None)
    with dut_token_service._refresh_locks_guard:
        dut_token_service._refresh_locks.pop("tester", None)
