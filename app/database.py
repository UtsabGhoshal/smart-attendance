"""
Database engine and session management using SQLAlchemy.
Connection pooling is configured for low-resource environments.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import settings

# ── Engine ────────────────────────────────────────────────────
# pool_size=3 and max_overflow=2 keep memory usage low on 4GB RAM
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=3,
    max_overflow=2,
    pool_pre_ping=True,  # Verify connections before use
    echo=settings.DEBUG,  # Log SQL in debug mode
)

# ── Session Factory ───────────────────────────────────────────
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# ── Base class for ORM models ────────────────────────────────
Base = declarative_base()


def get_db():
    """
    FastAPI dependency that yields a database session.
    Ensures the session is closed after each request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
