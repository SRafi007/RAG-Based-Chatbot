# app/config/db.py

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Create SQLAlchemy engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,  # Enable connection health checks
    echo=False,  # Set to True for SQL query logging in development
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()


def get_db() -> Session:
    """
    Dependency function to get database session.
    Yields a database session and ensures it's closed after use.

    Usage in FastAPI:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database - create all tables.
    This should be called on application startup.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise
