from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional


class DatabaseSettings(BaseSettings):
    POSTGRES_HOST: str
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        # Extra validation/masking of sensitive fields in logs/errors
        json_schema_extra={
            "sensitive_fields": {"POSTGRES_PASSWORD"}
        }
    )

    def get_connection_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"


class ServiceSettings(BaseSettings):
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    EMBEDDING_SERVICE_URL: Optional[str] = None
    RAG_SERVICE_URL: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True
    )


@lru_cache()
def get_db_settings() -> DatabaseSettings:
    return DatabaseSettings()


@lru_cache()
def get_service_settings() -> ServiceSettings:
    return ServiceSettings()