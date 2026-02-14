"""Minimal schema bootstrap.

We use Alembic long-term, but for early Render deploys we also ensure tables exist.
This avoids crashes when the database is fresh.
"""

from dealmatch.core import models  # noqa: F401
from dealmatch.db.base import Base
from dealmatch.db.session import engine


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("init_db: OK")
