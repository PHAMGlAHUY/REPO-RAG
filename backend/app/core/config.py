from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "my_rag"
    database_url: str = Field(
        default=f"sqlite:///{(BACKEND_DIR / 'app.db').as_posix()}",
        validation_alias=AliasChoices("database_url", "db_url", "DATABASE_URL", "DB_URL"),
    )


settings = Settings()