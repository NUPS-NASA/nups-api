"""Password hashing, verification, and token helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import base64
import hashlib
import hmac
import secrets
from typing import Any

from jose import JWTError, jwt

_PBKDF2_ITERATIONS = 100_000
_SALT_SIZE = 16
_ACCESS_TOKEN_TYPE = "access"
_REFRESH_TOKEN_TYPE = "refresh"


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _b64decode(encoded: str) -> bytes:
    return base64.urlsafe_b64decode(encoded.encode("ascii"))


def hash_password(password: str) -> str:
    """Return a salted PBKDF2 hash for the supplied password."""

    salt = secrets.token_bytes(_SALT_SIZE)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"{_b64encode(salt)}:{_b64encode(derived)}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Check a plaintext password against the stored salted hash."""

    try:
        salt_b64, derived_b64 = stored_hash.split(":", maxsplit=1)
    except ValueError:
        return False
    salt = _b64decode(salt_b64)
    expected = _b64decode(derived_b64)
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return hmac.compare_digest(actual, expected)


class InvalidTokenError(Exception):
    """Raised when a token cannot be decoded or validated."""


def _create_token(
    subject: str,
    *,
    secret_key: str,
    algorithm: str,
    expires_delta: timedelta,
    token_type: str,
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + expires_delta,
        "type": token_type,
    }
    return jwt.encode(payload, secret_key, algorithm=algorithm)


def create_access_token(
    user_id: int,
    *,
    secret_key: str,
    algorithm: str,
    expires_minutes: int,
) -> str:
    """Generate a signed JWT access token for the provided user identifier."""

    return _create_token(
        subject=str(user_id),
        secret_key=secret_key,
        algorithm=algorithm,
        expires_delta=timedelta(minutes=expires_minutes),
        token_type=_ACCESS_TOKEN_TYPE,
    )


def create_refresh_token(
    user_id: int,
    *,
    secret_key: str,
    algorithm: str,
    expires_minutes: int,
) -> str:
    """Generate a signed JWT refresh token for the provided user identifier."""

    return _create_token(
        subject=str(user_id),
        secret_key=secret_key,
        algorithm=algorithm,
        expires_delta=timedelta(minutes=expires_minutes),
        token_type=_REFRESH_TOKEN_TYPE,
    )


def decode_token(*, token: str, secret_key: str, algorithm: str) -> dict[str, Any]:
    """Decode a JWT and return its payload, raising on failure."""

    try:
        return jwt.decode(token, secret_key, algorithms=[algorithm])
    except JWTError as exc:  # pragma: no cover - jose error handling
        raise InvalidTokenError("Token decoding failed") from exc


def get_token_subject(
    *, token: str, expected_type: str, secret_key: str, algorithm: str
) -> str:
    """Return the subject from a token after validating its declared type."""

    payload = decode_token(token=token, secret_key=secret_key, algorithm=algorithm)
    token_type = payload.get("type")
    if token_type != expected_type:
        raise InvalidTokenError("Token type mismatch")
    subject = payload.get("sub")
    if subject is None:
        raise InvalidTokenError("Token subject missing")
    return str(subject)
