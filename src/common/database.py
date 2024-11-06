from contextlib import contextmanager
import psycopg2
from psycopg2.extras import DictCursor
from .config import get_db_settings
from .logging_config import setup_logging

logger = setup_logging("database")

@contextmanager
def get_db_connection():
    """
    Context manager for database connections that handles sensitive credentials safely
    """
    settings = get_db_settings()
    conn = None
    try:
        logger.debug("Establishing database connection...",
                    extra={"host": settings.POSTGRES_HOST, "db": settings.POSTGRES_DB})
        conn = psycopg2.connect(
            settings.get_connection_url(),
            cursor_factory=DictCursor
        )
        yield conn
    except Exception as e:
        logger.error("Database connection failed", exc_info=True,
                    extra={"host": settings.POSTGRES_HOST, "db": settings.POSTGRES_DB})
        raise
    finally:
        if conn is not None:
            conn.close()