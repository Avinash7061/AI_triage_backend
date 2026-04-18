"""
Database configuration — SQLAlchemy engine and session for MySQL.
Reads DATABASE_URL from environment (Railway auto-provides this).
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# Railway provides MYSQL_URL or DATABASE_URL
# Format: mysql://user:pass@host:port/dbname
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    os.environ.get("MYSQL_URL", "sqlite:///./mediflow_dev.db")
)

# SQLAlchemy needs specific dialect adjustments
if DATABASE_URL.startswith("mysql://"):
    DATABASE_URL = DATABASE_URL.replace("mysql://", "mysql+pymysql://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency — yields a DB session and closes it after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    print(f"  ✅ Database tables created/verified")
    print(f"  📊 URL: {DATABASE_URL[:DATABASE_URL.find('@')] if '@' in DATABASE_URL else DATABASE_URL}")
