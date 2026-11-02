import asyncio

import pytest

from app.external_services import sfistsp_api_client as mod


@pytest.mark.asyncio
async def test_batch_dedupes_caches_and_coalesces(monkeypatch):
    mod._lookup_cache.clear()
    calls: list[str] = []

    async def fake_uncached(self, isn):
        calls.append(isn)
        await asyncio.sleep(0.01)
        return mod.SfistspIsnReference(isn=isn, success=isn != "BAD")

    monkeypatch.setattr(mod.SfistspApiClient, "_lookup_isn_uncached", fake_uncached)
    client = mod.SfistspApiClient(mod.SfistspConfig())

    results = await client.lookup_isns_batch(["A", "B", "A", "BAD", "BAD"])
    assert [r.isn for r in results] == ["A", "B", "A", "BAD", "BAD"]
    assert sorted(calls) == ["A", "B", "BAD"]

    await asyncio.gather(client.lookup_isn("A"), client.lookup_isn("BAD"), client.lookup_isn("BAD"))
    assert sorted(calls) == ["A", "B", "BAD", "BAD"]  # success cached; failure retried once (coalesced)
    mod._lookup_cache.clear()

