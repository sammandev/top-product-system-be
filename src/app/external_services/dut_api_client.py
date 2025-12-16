import asyncio
import logging
import threading
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_shared_clients: dict[str, httpx.AsyncClient] = {}
_shared_clients_lock = threading.Lock()


def _get_shared_client(base_url: str) -> httpx.AsyncClient:
    with _shared_clients_lock:
        client = _shared_clients.get(base_url)
        if client is None:
            client = httpx.AsyncClient(
                base_url=base_url,
                timeout=httpx.Timeout(30.0, connect=30.0),  # Increased connect timeout
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
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
        self._timeout = httpx.Timeout(30.0, connect=30.0)  # Increased connect timeout
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

        try:
            result["dut_records"] = await self.get_dut_records(dut_id)
        except Exception as exc:
            logger.error("Error getting DUT records for %s: %s", dut_id, exc)

        try:
            sites = await self.get_sites()
            if site_name:
                sites = [site for site in sites if site.get("name") == site_name]
            result["sites"] = sites
        except Exception as exc:
            logger.error("Error getting sites: %s", exc)
            return result

        for site in result["sites"]:
            site_id = site.get("id")
            if not site_id:
                continue

            try:
                models = await self.get_models_by_site(site_id)
                result["models"].extend(models)
            except Exception as exc:
                logger.warning("Error getting models for site %s: %s", site_id, exc)
                continue

            for model in models:
                model_id = model.get("id")
                if not model_id:
                    continue

                try:
                    stations = await self.get_stations_by_model(model_id)
                except Exception as exc:
                    logger.warning("Error getting stations for model %s: %s", model_id, exc)
                    continue

                for station in stations:
                    station["site_name"] = site.get("name")
                    station["site_id"] = site_id
                    station["model_name"] = model.get("name")
                    station["model_id"] = model_id

                result["stations"].extend(stations)

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
