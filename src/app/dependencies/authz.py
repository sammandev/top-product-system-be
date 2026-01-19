from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.db import get_db
from app.models.user import User as DBUser
from app.utils.auth import decode_jwt, get_user
from app.utils.admin_access import is_user_admin

bearer = HTTPBearer(auto_error=False)


# Module-level dependency object to avoid calling Depends() in function defaults
bearer_dep = Depends(bearer)
db_dep = Depends(get_db)


def get_current_user(
    cred: HTTPAuthorizationCredentials = bearer_dep,
    db: Session = db_dep,
) -> DBUser:
    if cred is None or cred.scheme.lower() != "bearer":
        raise HTTPException(401, "missing or invalid authorization header", headers={"WWW-Authenticate": "Bearer"})

    # Some clients (or users pasting into Swagger UI) may include the literal
    # "Bearer <token>" string inside the credentials value. HTTPBearer normally
    # separates the scheme and token, but we defensively strip any leading
    # 'Bearer ' prefix from the credential text to avoid double-bearer issues.
    token = cred.credentials
    if isinstance(token, str) and token.lower().startswith("bearer "):
        token = token.split(" ", 1)[1]

    payload = decode_jwt(token)
    if not payload or payload.get("type") != "access":
        raise HTTPException(401, "invalid or expired token", headers={"WWW-Authenticate": "Bearer"})

    user = get_user(db, payload.get("sub"))
    if not user:
        raise HTTPException(401, "user not found or inactive")
    
    if not user.is_active:
        raise HTTPException(401, "user not found or inactive")

    if payload.get("ver") != user.token_version:
        raise HTTPException(401, "token no longer valid (revoked)")

    return user


# Module-level dependency object to avoid calling Depends() in function defaults
current_user_dependency = Depends(get_current_user)


def require_admin(user: DBUser = current_user_dependency) -> DBUser:
    if not is_user_admin(user):
        raise HTTPException(403, "admin privileges required")
    return user


def require_permission(perm: str):
    def checker(user: DBUser = current_user_dependency) -> DBUser:
        if is_user_admin(user):
            return user
        has = any(perm == p.name for r in user.roles for p in r.permissions)
        if not has:
            raise HTTPException(403, f"missing permission: {perm}")
        return user

    return checker
