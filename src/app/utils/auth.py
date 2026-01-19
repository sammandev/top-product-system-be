import logging
import os
import uuid
from datetime import UTC, datetime, timedelta

import jwt
from sqlalchemy import func
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher
from sqlalchemy.orm import Session

from app.models.user import User
from app.utils.admin_access import is_user_admin

# === Config ===
JWT_SECRET = os.environ.get("JWT_SECRET", "change_this_secret")  # random 32-bytes | run in terminal: openssl rand -hex 32
JWT_ALG = os.environ.get("JWT_ALG", "HS256")
JWT_TTL_SECONDS = int(os.environ.get("JWT_TTL", "86400"))  # access TTL
JWT_REFRESH_TTL_SECONDS = int(os.environ.get("JWT_REFRESH_TTL", "86400"))  # refresh TTL

# Argon2 tuning (adjust via env)
ARGON2_TIME_COST = int(os.environ.get("ARGON2_TIME_COST", "2"))
ARGON2_MEMORY_COST = int(os.environ.get("ARGON2_MEMORY_COST", str(32_768)))  # KiB
ARGON2_PARALLELISM = int(os.environ.get("ARGON2_PARALLELISM", "2"))

password_hasher = PasswordHash(
    (
        Argon2Hasher(
            time_cost=ARGON2_TIME_COST,
            memory_cost=ARGON2_MEMORY_COST,
            parallelism=ARGON2_PARALLELISM,
        ),
    )
)


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return password_hasher.verify(plain, hashed)
    except Exception:
        return False


def _base_payload(user: User) -> dict:
    now = datetime.now(UTC)
    return {
        "sub": user.username,
        "admin": is_user_admin(user),
        "ver": user.token_version,  # versioned JWT for stateless revocation
        "iat": int(now.timestamp()),
        "jti": uuid.uuid4().hex,
    }


def create_access_token(user: User, ttl: int | None = None) -> str:
    p = _base_payload(user)
    exp = datetime.now(UTC) + timedelta(seconds=(ttl or JWT_TTL_SECONDS))
    p.update({"type": "access", "exp": int(exp.timestamp())})
    return jwt.encode(p, JWT_SECRET, algorithm=JWT_ALG)


def create_refresh_token(user: User, ttl: int | None = None) -> str:
    p = _base_payload(user)
    exp = datetime.now(UTC) + timedelta(seconds=(ttl or JWT_REFRESH_TTL_SECONDS))
    # IMPORTANT: mark type=refresh so it cannot be used for Bearer auth
    p.update({"type": "refresh", "exp": int(exp.timestamp())})
    return jwt.encode(p, JWT_SECRET, algorithm=JWT_ALG)


def decode_jwt(token: str) -> dict | None:
    logger = logging.getLogger(__name__)
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError:
        # Token expired
        logger.debug("JWT decode failed: token expired")
        return None
    except jwt.InvalidTokenError as exc:
        # Signature invalid or other token errors
        logger.debug("JWT decode failed: invalid token (%s)", exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Unexpected error decoding JWT: %s", exc)
        return None


def decode_jwt_token(token: str, expected_type: str | None = "access") -> tuple[str | None, bool]:
    """
    Decode a JWT and return (username, is_admin) while enforcing the token type when requested.

    Args:
        token: Raw JWT string.
        expected_type: When provided, the decoded payload must include the matching ``type`` field.

    Returns:
        Tuple of (username, is_admin_flag). Returns (None, False) when validation fails.
    """
    payload = decode_jwt(token)
    if not payload:
        return None, False
    if expected_type is not None and payload.get("type") != expected_type:
        return None, False
    return payload.get("sub"), bool(payload.get("admin"))


def normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def get_user(db: Session, username: str) -> User | None:
    normalized = normalize_username(username)
    if not normalized:
        return None
    return db.query(User).filter(func.lower(User.username) == normalized).one_or_none()


def create_user(db: Session, username: str, password: str, is_admin: bool = False) -> User:
    normalized = normalize_username(username)
    user = get_user(db, normalized)
    if user:
        user.password_hash = hash_password(password)
        user.is_admin = is_admin
    else:
        user = User(username=normalized, password_hash=hash_password(password), is_admin=is_admin)
        db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    user = get_user(db, username)
    if not user or not user.is_active:
        return None
    if verify_password(password, user.password_hash):
        return user
    return None


def revoke_user_tokens(db: Session, user: User):
    """Increment token_version to invalidate all existing tokens (access & refresh)."""
    user.token_version += 1
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
