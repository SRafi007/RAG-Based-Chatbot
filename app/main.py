# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import settings
from app.utils.logger import get_logger
from app.api import chat_api
from app.config.db import Base, engine
from app.models import conversation  # Import models to register with Base




logger = get_logger(__name__)

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    # Create DB tables (in dev; in production, use Alembic migrations)
    Base.metadata.create_all(bind=engine)

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="RAG-based Chatbot with Multi-Agent System"
    )

    # ✅ CORS Setup
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # change to specific domains in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ✅ Routers
    app.include_router(chat_api.router, prefix="/api/chat", tags=["Chat"])

    # ✅ Startup and Shutdown Events
    @app.on_event("startup")
    async def startup_event():
        logger.info(f"🚀 {settings.app_name} starting in {settings.app_env} mode on port {settings.app_port}")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("🛑 Application shutdown complete.")

    return app


app = create_app()

