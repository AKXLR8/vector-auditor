"""Auth — JWT, password hashing, GitHub OAuth, MFA stubs."""
import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

logger = logging.getLogger("rga_auditor.auth")

JWT_ALG = "HS256"
_1_HOUR = 60


def get_jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY")
    if not secret:
        env = os.getenv("ENVIRONMENT", "development").lower()
        if env == "production":
            raise RuntimeError("JWT_SECRET_KEY is required in production")
        secret = "dev-only-fallback-secret-change-me"
    return secret


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False


def create_access_token(
    user_id: str,
    roles: Optional[list[str]] = None,
    extra: Optional[dict] = None,
    expires_minutes: Optional[int] = None,
) -> tuple[str, int, str]:
    minutes = expires_minutes or int(os.getenv("JWT_EXPIRE_MINUTES", str(_1_HOUR)))
    expire = datetime.utcnow() + timedelta(minutes=minutes)
    jti = secrets.token_urlsafe(16)
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": jti,
        "roles": roles or ["user"],
    }
    if extra:
        payload.update(extra)
    token = jwt.encode(payload, get_jwt_secret(), algorithm=JWT_ALG)
    return token, minutes * 60, jti


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, get_jwt_secret(), algorithms=[JWT_ALG])
    except JWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"invalid token: {e}")


bearer_scheme = HTTPBearer(auto_error=False)


def _roles_from_payload(payload: dict) -> list[str]:
    roles = payload.get("roles")
    if isinstance(roles, list) and roles:
        return [str(r) for r in roles]
    legacy = payload.get("role")
    if legacy:
        return [str(legacy)]
    return ["user"]


async def current_user(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    if creds is None:
        raise HTTPException(status_code=401, detail="missing bearer token")
    payload = decode_token(creds.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="token missing subject")

    from ..database.repository import get_user_by_id, is_token_blacklisted
    from ..database.session import get_session_factory

    jti = payload.get("jti", "")
    sf = get_session_factory()
    if sf is not None:
        async with sf() as s:
            if await is_token_blacklisted(s, jti):
                raise HTTPException(status_code=401, detail="token revoked")
            db_user = await get_user_by_id(s, user_id)
        if db_user is None:
            raise HTTPException(status_code=401, detail="user not found")
        roles = db_user.get("roles") or _roles_from_payload(payload)
    else:
        # In-memory fallback: also check the in-memory blacklist store
        if await is_token_blacklisted(None, jti):
            raise HTTPException(status_code=401, detail="token revoked")
        roles = _roles_from_payload(payload)

    request.state.user_id = user_id
    request.state.jti = payload.get("jti", "")
    request.state.roles = roles
    return {"id": user_id, "role": roles[0] if roles else "user", "roles": roles}


def require_role(*roles: str):
    allowed = set(roles)

    async def dep(user: dict = Depends(current_user)) -> dict:
        user_roles = set(user.get("roles") or [user.get("role", "user")])
        if not (user_roles & allowed):
            raise HTTPException(status_code=403, detail=f"requires role: {', '.join(sorted(allowed))}")
        return user

    return dep
