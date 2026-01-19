from __future__ import annotations

from app.models.user import User

ADMIN_USERNAMES = {
    "samuel_halomoan",
}

ADMIN_WORKER_IDS = {
    "MW2400549",
}


def _normalize_username(username: str | None) -> str:
    return (username or "").strip().lower()


def _normalize_worker_id(worker_id: str | None) -> str:
    return (worker_id or "").strip().upper()


def is_user_admin(user: User) -> bool:
    if user.is_admin or user.is_ptb_admin:
        return True

    if _normalize_username(user.username) in ADMIN_USERNAMES:
        return True

    if _normalize_worker_id(getattr(user, "worker_id", None)) in ADMIN_WORKER_IDS:
        return True

    return False
