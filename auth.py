# auth.py
import os
from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi import HTTPException, Header
from typing import Optional

SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret-key-change-in-prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8  # 8小时


def create_token(tenant_id: str, user_id: str) -> str:
    payload = {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Token 无效或已过期")


def get_current_tenant(authorization: Optional[str] = Header(None)) -> dict:
    """
    FastAPI 依赖注入：从请求头解析租户信息
    没有 token 时默认 tenant_id = "default"（开发模式）
    """
    if not authorization or not authorization.startswith("Bearer "):
        return {"tenant_id": "default", "user_id": "anonymous"}
    token = authorization.replace("Bearer ", "")
    return decode_token(token)