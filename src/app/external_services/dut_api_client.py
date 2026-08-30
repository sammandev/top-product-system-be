import asyncio
import logging
import os
import threading
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_shared_clients: dict[str, httpx.AsyncClient] = {}
_shared_clients_lock = threading.Lock()
DUT_MAX_PARALLEL_REQUESTS = int(os.getenv("DUT_CATALOG_MAX_PARALLEL_REQUESTS", "8"))
DUT_REQUEST_TIMEOUT = float(os.getenv("DUT_API_TIMEOUT", "30"))
DUT_CONNECT_TIMEOUT = float(os.getenv("DUT_API_CONNECT_TIMEOUT", "5"))
DUT_POOL_TIMEOUT = float(os.getenv("DUT_API_POOL_TIMEOUT", "5"))
DUT_MAX_CONNECTIONS = int(os.getenv("DUT_API_MAX_CONNECTIONS", "100"))
DUT_MAX_KEEPALIVE_CONNECTIONS = int(os.getenv("DUT_API_MAX_KEEPALIVE_CONNECTIONS", "20"))


def _build_timeout() -> httpx.Timeout:
    return httpx.Timeout(
        DUT_REQUEST_TIMEOUT,
        connect=DUT_CONNECT_TIMEOUT,
        pool=DUT_POOL_TIMEOUT,
    )


def _get_shared_client(base_url: str) -> httpx.AsyncClient:
    with _shared_clients_lock:
        client = _shared_clients.get(base_url)
        if client is None:
            client = httpx.AsyncClient(
                base_url=base_url,
                timeout=_build_timeout(),
                limits=httpx.Limits(
                    max_keepalive_connections=DUT_MAX_KEEPALIVE_CONNECTIONS,
                    max_connections=DUT_MAX_CONNECTIONS,
                    keepalive_expiry=30.0,
                ),
            )
            _shared_clients[base_url] = client
        return client


async def close_shared_clients() -> None:
    with _shared_clients_lock:
        clients = list(_shared_clients.values())
        _shared_clients.clear()
    for client in clients:
        await client.aclose()


class DUTAPIClient:
    """Async client for interacting with the DUT Information API."""

    def __init__(self, base_url: str = "http://172.18.220.56:9001"):
        self.base_url = base_url.rstrip("/")
        self.access_token: str | None = None
        self.refresh_token: str | None = None
        self.token_expiry: datetime | None = None

        self._client = _get_shared_client(self.base_url)
        self._timeout = _build_timeout()
        self._token_lock = asyncio.Lock()

    def _token_is_valid(self) -> bool:
        if not self.access_token:
            return False

        expiry = self.token_expiry
        if expiry is None:
            return True
        if expiry.tzinfo is None:
            return datetime.now() < expiry
        return datetime.now(UTC) < expiry

    async def authenticate(self, username: str, password: str) -> dict[str, Any]:
        """Authenticate against the upstream API and cache the returned tokens."""

        payload = {"username": username, "password": password}
        async with self._token_lock:
            response = await self._perform_request(
                "POST",
                "/api/user/token/",
                headers={"Content-Type": "application/json"},
                json=payload,
                include_auth=False,
            )
            data = response.json()
            self.access_token = data.get("access")
            self.refresh_token = data.get("refresh")
            # UPDATED: tokens expire in 24h; refresh slightly early to avoid race conditions.
            self.token_expiry = datetime.now(UTC) + timedelta(hours=23)
            logger.info("DUTAPI authentication succeeded for %s", data.get("username", username))
            return data

    async def refresh_access_token(self) -> str:
        """Refresh the cached access token using the refresh token."""

        if not self.refresh_token:
            raise ValueError("No refresh token available. Call authenticate() first.")

        async with self._token_lock:
            if self._token_is_valid():
                return self.access_token or ""

            payload = {"refresh": self.refresh_token}
            response = await self._perform_request(
                "POST",
                "/api/user/token/refresh/",
                headers={"Content-Type": "application/json"},
                json=payload,
                include_auth=False,
            )
            data = response.json()
            self.access_token = data.get("access")
            self.token_expiry = datetime.now(UTC) + timedelta(hours=23)
            logger.info("DUTAPI access token refreshed")
            return self.access_token or ""

    async def _get_headers(self) -> dict[str, str]:
        if not self._token_is_valid():
            await self.refresh_access_token()

        if not self.access_token:
            raise ValueError("Not authenticated. Call authenticate() first.")

        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    async def _perform_request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: Any = None,
        timeout: float | None = None,
        include_auth: bool = True,
    ) -> httpx.Response:
        """Execute an HTTP request while recording latency metrics."""

        resolved_headers: dict[str, str] = {}
        if include_auth:
            resolved_headers.update(await self._get_headers())
        if headers:
            resolved_headers.update(headers)

        started = perf_counter()
        try:
            response = await self._client.request(
                method,
                path,
                headers=resolved_headers or None,
                params=params,
                json=json,
                timeout=timeout or self._timeout,
                follow_redirects=True,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            elapsed_ms = (perf_counter() - started) * 1000
            logger.warning(
                "DUTAPI request failed: method=%s path=%s status=%s elapsed_ms=%.1f",
                method,
                path,
                exc.response.status_code,
                elapsed_ms,
            )
            raise
        except httpx.HTTPError as exc:
            elapsed_ms = (perf_counter() - started) * 1000
            logger.warning(
                "DUTAPI request error: method=%s path=%s elapsed_ms=%.1f error=%s",
                method,
                path,
                elapsed_ms,
                exc,
            )
            raise

        elapsed_ms = (perf_counter() - started) * 1000
        payload_size = len(response.content) if response.content is not None else 0
        logger.info(
            "DUTAPI request: method=%s path=%s status=%s elapsed_ms=%.1f size_bytes=%s",
            method,
            path,
            response.status_code,
            elapsed_ms,
            payload_size,
        )
        return response

    async def get_sites(self) -> list[dict[str, Any]]:
        response = await self._perform_request("GET", "/api/site")
        return response.json()

    async def get_models_by_site(self, site_id: int) -> list[dict[str, Any]]:
        response = await self._perform_request("GET", f"/api/models/list/{site_id}/")
        return response.json()

    async def get_stations_by_model(self, model_id: int) -> list[dict[str, Any]]:
        response = await self._perform_request("GET", f"/api/api/station/{model_id}")
        return response.json()

    async def get_dut_records(self, dut_id: str) -> dict[str, Any]:
        params = {"DUT": dut_id}
        response = await self._perform_request("GET", "/api/api/dut/records", params=params)
        return response.json()

    async def get_station_records(self, station_id: int, dut_id: int) -> dict[str, Any]:
        response = await self._perform_request("GET", f"/api/dut/records/{station_id}/{dut_id}")
        return response.json()

    async def get_latest_station_records(self, station_id: int, dut_id: int) -> dict[str, Any]:
        """Compat helper: upstream has no /latest endpoint; reuse full records."""
        return await self.get_station_records(station_id, dut_id)

    async def get_svn_info(self) -> list[dict[str, Any]]:
        response = await self._perform_request("GET", "/api/api/svn/info")
        return response.json()

    async def get_complete_dut_info(self, dut_id: str, site_name: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "dut_id": dut_id,
            "dut_records": None,
            "sites": [],
            "models": [],
            "stations": [],
        }

        records_result, sites_result = await asyncio.gather(
            self.get_dut_records(dut_id),
            self.get_sites(),
            return_exceptions=True,
        )
        if isinstance(records_result, Exception):
            logger.error("Error getting DUT records for %s: %s", dut_id, records_result)
        else:
            result["dut_records"] = records_result

        if isinstance(sites_result, Exception):
            logger.error("Error getting sites: %s", sites_result)
            return result

        sites = sites_result
        if site_name:
            sites = [site for site in sites if site.get("name") == site_name]
        result["sites"] = sites
        semaphore = asyncio.Semaphore(max(1, DUT_MAX_PARALLEL_REQUESTS))

        async def get_models(site: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            site_id = site.get("id")
            if not site_id:
                return site, []
            try:
                async with semaphore:
                    return site, await self.get_models_by_site(site_id)
            except Exception as exc:
                logger.warning("Error getting models for site %s: %s", site_id, exc)
                return site, []

        models_by_site = await asyncio.gather(*(get_models(site) for site in sites))
        model_contexts = [
            (site, model)
            for site, models in models_by_site
            for model in models
            if model.get("id")
        ]
        result["models"] = [model for _, model in model_contexts]

        async def get_stations(site: dict[str, Any], model: dict[str, Any]) -> list[dict[str, Any]]:
            model_id = model["id"]
            try:
                async with semaphore:
                    stations = await self.get_stations_by_model(model_id)
            except Exception as exc:
                logger.warning("Error getting stations for model %s: %s", model_id, exc)
                return []

            for station in stations:
                station["site_name"] = site.get("name")
                station["site_id"] = site.get("id")
                station["model_name"] = model.get("name")
                station["model_id"] = model_id
            return stations

        stations_by_model = await asyncio.gather(
            *(get_stations(site, model) for site, model in model_contexts)
        )
        result["stations"] = [station for stations in stations_by_model for station in stations]

        logger.info(
            "Retrieved complete DUT info: %s models, %s stations",
            len(result["models"]),
            len(result["stations"]),
        )
        return result

    async def get_devices_by_station(self, station_id: int) -> list[dict[str, Any]]:
        try:
            response = await self._perform_request("GET", f"/api/device/{station_id}")
            return response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            fallback = await self._perform_request("GET", f"/api/device/{station_id}/")
            return fallback.json()

    async def get_devices_by_period(
        self,
        station_id: int,
        start_time: datetime,
        end_time: datetime,
        test_result: str = "ALL",
    ) -> list[dict[str, Any]]:
        params = {
            "start_time": start_time.isoformat().replace("+00:00", "Z"),
            "end_time": end_time.isoformat().replace("+00:00", "Z"),
        }
        response = await self._perform_request(
            "GET",
            f"/api/device/period/{station_id}/{test_result}",
            params=params,
        )
        return response.json()

    async def get_test_results_by_device(self, payload: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        body: dict[str, Any]
        if isinstance(payload, list):
            body = payload[0] if payload else {}
        else:
            body = payload
        if isinstance(body, dict) and "device_id" in body:
            body = {**body, "device_id": str(body.get("device_id"))}
        response = await self._perform_request("POST", "/api/testresultiplas/device", json=body)
        return response.json()

    async def get_test_items_by_station(self, station_id: int) -> list[dict[str, Any]]:
        response = await self._perform_request("GET", f"/api/testitems/{station_id}")
        return response.json()

    async def get_latest_test_items_by_range(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Fetch the latest station test items using the external DUT time-range endpoint."""
        response = await self._perform_request("POST", "/api/testitems/latest", json=payload)
        return response.json()

    async def get_model_summary(self, payload: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
        body: dict[str, Any]
        if isinstance(payload, list):
            body = payload[0] if payload else {}
        else:
            body = payload
        if isinstance(body, dict) and "model_id" in body and isinstance(body["model_id"], str) and body["model_id"].isdigit():
            body = {**body, "model_id": int(body["model_id"])}
        response = await self._perform_request("POST", "/api/testresultiplas/summary2", json=body)
        return response.json()

    async def get_station_nonvalue_records(self, station_id: int, dut_id: int) -> dict[str, Any]:
        response = await self._perform_request(
            "GET",
            f"/api/dut/records/nonvalue/{station_id}/{dut_id}",
        )
        return response.json()

    async def get_latest_nonvalue_record(self, station_id: int, dut_id: int) -> dict[str, Any]:
        """Get latest non-value test records for a specific station and DUT.

        Upstream DUT API has no dedicated 'latest' nonvalue endpoint; reuse the full record set and trim in caller.
        """
        return await self.get_station_nonvalue_records(station_id, dut_id)

    async def get_station_nonvalue_bin_records(self, station_id: int, dut_id: int) -> dict[str, Any]:
        response = await self._perform_request(
            "GET",
            f"/api/dut/records/nonvalueBin/{station_id}/{dut_id}",
        )
        return response.json()

    async def get_latest_nonvalue_bin_record(self, station_id: int, dut_id: int) -> dict[str, Any]:
        """Get latest non-value BIN test records for a specific station and DUT.

        Upstream DUT API has no dedicated 'latest' nonvalue-bin endpoint; reuse the full record set and trim in caller.
        """
        return await self.get_station_nonvalue_bin_records(station_id, dut_id)

    async def get_pa_test_items_trend(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Get PA test items trend data (mean and mid values for SROM_OLD and SROM_NEW patterns).

        Calls POST /api/api/testitems/PA/trend endpoint.

        Expected payload:
        {
            "start_time": "2025-11-15T08:22:21.00Z",
            "end_time": "2025-11-17T08:22:21.00Z",
            "model": "",
            "station_id": 145,
            "test_items": ["WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80", "WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80"]
        }

        Returns:
        {
            "WiFi_PA1_SROM_OLD_5985_11AX_MCS9_B80": {"mid": 11219.0, "mean": 11227},
            "WiFi_PA1_SROM_NEW_5985_11AX_MCS9_B80": {"mid": 11313.0, "mean": 11308}
        }
        """
        response = await self._perform_request("POST", "/api/api/testitems/PA/trend", json=payload)
        return response.json()

    async def get_user_account_info(self) -> dict[str, Any]:
        """Get authenticated user's account information from external DUT API.

        Calls GET /api/user/account/info endpoint.

        Returns user data including:
        {
            "id": 123,
            "username": "user_name",
            "email": "user@example.com",
            "is_ptb_admin": true,
            "first_name": "John",
            "last_name": "Doe",
            ...
        }
        """
        response = await self._perform_request("GET", "/api/user/account/info")
        return response.json()
