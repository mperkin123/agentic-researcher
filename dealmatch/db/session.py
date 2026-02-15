import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is required")
    # allow either sqlalchemy or raw postgres URLs
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _pool_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


engine = create_engine(
    get_database_url(),
    pool_pre_ping=True,
    pool_size=_pool_int("DB_POOL_SIZE", 10),
    max_overflow=_pool_int("DB_MAX_OVERFLOW", 20),
    pool_timeout=_pool_int("DB_POOL_TIMEOUT", 30),
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
