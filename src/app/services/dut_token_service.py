import json
import logging
import os
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock

import redis
import requests

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DUT_BASE_URL = os.getenv("DUT_API_BASE_URL", "http://172.18.220.56:9001")
REDIS_DUT_TTL = int(os.getenv("REDIS_DUT_TTL", "90000"))  # ~25h

_local_store: dict[str, dict] = {}
_local_lock = Lock()
_cleanup_started = False
_cleanup_lock = Lock()
_FILE_CACHE_ENABLED = os.getenv("DUT_TOKEN_FILE_CACHE", "1").lower() not in {"0", "false", "no"}
_CACHE_FILE = Path(os.getenv("DUT_TOKEN_CACHE_FILE", ".cache/dut_tokens.json"))


def _safe_parse_datetime(value: str) -> datetime:
    """
    Parse ISO timestamps that may be naive (no tzinfo) or offset-aware.

    Always return a timezone-aware datetime in UTC for reliable comparisons.
    """
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _cleanup_expired_tokens():
    while True:
        now = datetime.now(UTC)
        with _local_lock:
            expired = [u for u, t in _local_store.items() if "expiry" in t and _safe_parse_datetime(t["expiry"]) < now]
            for u in expired:
                _local_store.pop(u, None)
        time.sleep(600)  # every 10 minutes


def _start_token_cleanup_worker():
    global _cleanup_started
    if _cleanup_started:
        return
    with _cleanup_lock:
        if _cleanup_started:
            return
        threading.Thread(target=_cleanup_expired_tokens, daemon=True).start()
        _cleanup_started = True


_start_token_cleanup_worker()


def _load_local_cache_from_disk():
    if not _FILE_CACHE_ENABLED:
        return
    try:
        if not _CACHE_FILE.exists():
            return
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        with _local_lock:
            _local_store.update({str(k): v for k, v in data.items() if isinstance(v, dict)})
        logger.info("Loaded DUT token cache from %s", _CACHE_FILE)
    except Exception as exc:
        logger.warning("Failed to load DUT token cache from %s: %s", _CACHE_FILE, exc)


def _persist_local_cache():
    if not _FILE_CACHE_ENABLED:
        return
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _local_lock:
            snapshot = dict(_local_store)
        _CACHE_FILE.write_text(json.dumps(snapshot), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to persist DUT token cache to %s: %s", _CACHE_FILE, exc)


_load_local_cache_from_disk()

try:
    _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    _redis_client.ping()
    _redis_available = True
except Exception as exc:
    # Log the resolved REDIS_URL and the reason to aid debugging
    logger.warning("Redis unavailable at %s; using in-memory DUT token cache. \nError: %s", REDIS_URL, exc)
    _redis_client = None
    _redis_available = False


def _dut_key(username: str) -> str:
    return f"dut:tokens:{username}"


def store_tokens(username: str, access: str, refresh: str, hours: int = 23):
    """Persist DUT tokens in Redis when available, otherwise in-memory."""
    expiry = (datetime.now(UTC) + timedelta(hours=hours)).isoformat()
    payload = {"access": access, "refresh": refresh, "expiry": expiry}

    if _redis_available and _redis_client is not None:
        try:
            _redis_client.setex(_dut_key(username), REDIS_DUT_TTL, json.dumps(payload))
            logger.info("Stored DUT tokens for user=%s (redis)", username)
            return
        except redis.exceptions.RedisError as exc:
            logger.warning("Redis setex failed (%s); falling back to in-memory cache.", exc)

    with _local_lock:
        _local_store[username] = payload
    _persist_local_cache()
    logger.info("Stored DUT tokens for user=%s (memory)", username)


def get_tokens(username: str) -> dict | None:
    if _redis_available and _redis_client is not None:
        try:
            data = _redis_client.get(_dut_key(username))
            if data:
                return json.loads(data)
        except redis.exceptions.RedisError as exc:
            logger.warning("Redis get failed (%s); attempting in-memory lookup.", exc)

    with _local_lock:
        payload = _local_store.get(username)
        return payload.copy() if payload else None


def clear_tokens(username: str):
    removed = False
    if _redis_available and _redis_client is not None:
        try:
            removed = bool(_redis_client.delete(_dut_key(username)))
        except redis.exceptions.RedisError as exc:
            logger.warning("Redis delete failed (%s); clearing in-memory cache.", exc)

    with _local_lock:
        removed_mem = _local_store.pop(username, None) is not None
    _persist_local_cache()

    if removed or removed_mem:
        logger.info("Cleared DUT tokens for user=%s", username)


def ensure_valid_access(username: str) -> str | None:
    """
    Retrieve valid DUT access token.
    Auto-refresh using stored refresh token if expired.
    Returns the valid access token, or None if re-login required.
    """
    tokens = get_tokens(username)
    if not tokens:
        logger.warning("No DUT tokens found for user=%s", username)
        return None

    try:
        expiry = _safe_parse_datetime(tokens["expiry"])
    except Exception:
        expiry = datetime.now(UTC) - timedelta(seconds=1)

    if datetime.now(UTC) < expiry:
        return tokens["access"]

    try:
        refresh_url = f"{DUT_BASE_URL.rstrip('/')}/api/user/token/refresh/"
        response = requests.post(refresh_url, json={"refresh": tokens["refresh"]}, timeout=30)
        response.raise_for_status()
        data = response.json()
        new_access = data.get("access")
        if not new_access:
            raise ValueError("No access token in refresh response")

        store_tokens(username, new_access, tokens["refresh"])
        logger.info("Refreshed DUT token for user=%s", username)
        return new_access
    except Exception as exc:
        logger.error("DUT token refresh failed for user=%s: %s", username, exc)
        clear_tokens(username)
    return None
