"""Shared FastAPI dependencies and type aliases used across routers."""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from .. import models
from ..config import get_settings
from ..database import get_db
from ..security import InvalidTokenError, get_token_subject

DBSession = Annotated[AsyncSession, Depends(get_db)]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/users/login")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)], db: DBSession
) -> models.User:
    """Resolve the authenticated user from a bearer token."""

    settings = get_settings()
    try:
        subject = get_token_subject(
            token=token,
            expected_type="access",
            secret_key=settings.auth_secret_key,
            algorithm=settings.auth_algorithm,
        )
    except InvalidTokenError as exc:  # pragma: no cover - runtime validation
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from exc

    try:
        user_id = int(subject)
    except ValueError as exc:  # pragma: no cover - defensive
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token subject",
        ) from exc

    user = await db.get(models.User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


CurrentUser = Annotated[models.User, Depends(get_current_user)]

__all__ = ["DBSession", "CurrentUser", "get_current_user"]
