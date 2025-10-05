from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or defaults."""

    app_name: str = "nups-api"
    app_port: int = 4000
    database_url: str = "sqlite+aiosqlite:///./nups.db"
    storage_tmp_dir: str = Field(
        default="./storage/tmp",
        description="Directory used for staging uploaded artefacts before persistence.",
    )
    storage_data_dir: str = Field(
        default="./storage/data",
        description="Directory where committed data artefacts are stored.",
    )
    auth_secret_key: str = Field(
        default="change-me",
        description="Secret key used for signing authentication tokens.",
    )
    auth_algorithm: str = Field(
        default="HS256",
        description="Signing algorithm for JWT tokens.",
    )
    auth_access_token_exp_minutes: int = Field(
        default=60,
        description="Access token lifetime in minutes.",
    )
    auth_refresh_token_exp_minutes: int = Field(
        default=60 * 24 * 14,
        description="Refresh token lifetime in minutes.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""

    return Settings()
