"""
FastAPI application entry point.

Initializes the app, includes routers, creates database tables,
loads the FaceEngine model, and rebuilds the FAISS index on startup.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import engine, Base, SessionLocal
from app.models import User, FacialEmbedding, AttendanceLog  # noqa: F401 — registers models
from app.services.face_engine import face_engine
from app.services.faiss_index import faiss_manager
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup & shutdown logic.
    - Creates all database tables on startup (if they don't exist).
    - Ensures the data directory exists for FAISS index files.
    - Loads the InsightFace model (one-time cost of ~5-8 seconds).
    - Rebuilds the FAISS index from PostgreSQL data.
    """
    # ── Startup ───────────────────────────────────────────
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Create tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready")

    # Ensure data directory for FAISS indexes
    settings.ensure_data_dir()
    logger.info(f"Data directory: {settings.DATA_DIR}")

    # Load InsightFace model
    face_engine.initialize()
    logger.info("Face engine ready")

    # Rebuild FAISS index from database
    db = SessionLocal()
    try:
        faiss_manager.rebuild_from_db(db)
    finally:
        db.close()
    logger.info(f"FAISS index ready ({faiss_manager.total_embeddings} embeddings)")

    yield

    # ── Shutdown ──────────────────────────────────────────
    logger.info("Shutting down Smart Attendance System")
    engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered facial recognition attendance system for college use.",
    lifespan=lifespan,
)


# ── Middleware ────────────────────────────────────────────────

# CORS — allow frontend to connect later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    if duration > 1.0:  # Only log slow requests
        logger.warning(f"{request.method} {request.url.path} took {duration:.2f}s")
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check logs for details."},
    )


# ── Register Routers ──────────────────────────────────────────

from app.routes.users import router as users_router  # noqa: E402
from app.routes.enroll import router as enroll_router  # noqa: E402
from app.routes.attendance import router as attendance_router  # noqa: E402
from app.routes.reports import router as reports_router  # noqa: E402
from app.routes.admin import router as admin_router  # noqa: E402

app.include_router(users_router)
app.include_router(enroll_router)
app.include_router(attendance_router)
app.include_router(reports_router)
app.include_router(admin_router)


# ── Health Check ──────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "database": "connected",
        "face_model": settings.FACE_MODEL,
        "face_engine_ready": face_engine.is_initialized,
        "faiss_index_ready": faiss_manager.is_initialized,
        "faiss_total_embeddings": faiss_manager.total_embeddings,
    }
