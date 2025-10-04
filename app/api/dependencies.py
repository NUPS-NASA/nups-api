"""Shared FastAPI dependencies and type aliases used across routers."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db

DBSession = Annotated[AsyncSession, Depends(get_db)]

__all__ = ["DBSession"]
