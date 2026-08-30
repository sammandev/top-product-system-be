import httpx
import pytest

from app.external_services.sfistsp_api_client import (
    SfistspApiClient,
    SfistspConfig,
    close_shared_clients,
    create_sfistsp_client,
)


@pytest.mark.asyncio
async def test_sfistsp_contexts_reuse_shared_client():
    await close_shared_clients()

    async with create_sfistsp_client(base_url="http://sfistsp.test") as first:
        first_client = first._client

    async with create_sfistsp_client(base_url="http://sfistsp.test") as second:
        second_client = second._client

    assert first_client is second_client
    assert first_client is not None
    assert not first_client.is_closed

    await close_shared_clients()
    assert first_client.is_closed


@pytest.mark.asyncio
async def test_sfistsp_transport_error_does_not_expose_credentials():
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    config = SfistspConfig(
        base_url="http://sfistsp.test",
        program_id="program",
        program_password="secret-password",
        timeout=1,
    )
    client = SfistspApiClient(config)
    client._client = httpx.AsyncClient(base_url=config.base_url, transport=httpx.MockTransport(fail))

    result = await client.lookup_isn("ISN123")

    assert result.success is False
    assert result.error_message == "SFISTSP service unavailable"
    assert "secret-password" not in result.error_message
    await client._client.aclose()
