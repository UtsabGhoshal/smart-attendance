"""
Admin routes — Protected utilities for system management.

All admin routes require the ADMIN_API_KEY header for authentication.
"""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Header, status
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import User, FacialEmbedding, AttendanceLog
from app.schemas import MessageResponse
from app.services.faiss_index import faiss_manager
from app.services.face_engine import face_engine

router = APIRouter(prefix="/api/admin", tags=["Admin"])


def verify_admin_key(x_admin_key: str = Header(..., description="Admin API key")):
    """Dependency that checks the admin API key from request header."""
    if x_admin_key != settings.ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin API key.",
        )
    return True


@router.get(
    "/system-info",
    summary="System health & status",
    dependencies=[Depends(verify_admin_key)],
)
def system_info(db: Session = Depends(get_db)):
    """Get detailed system information (requires admin key)."""
    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()  # noqa: E712
    total_embeddings_db = db.query(FacialEmbedding).count()
    total_logs = db.query(AttendanceLog).count()
    today_logs = (
        db.query(AttendanceLog)
        .filter(AttendanceLog.attendance_date == date.today())
        .count()
    )

    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "face_model": settings.FACE_MODEL,
        "face_engine_ready": face_engine.is_initialized,
        "faiss_index_ready": faiss_manager.is_initialized,
        "faiss_embeddings": faiss_manager.total_embeddings,
        "db_embeddings": total_embeddings_db,
        "db_synced": faiss_manager.total_embeddings == total_embeddings_db,
        "total_users": total_users,
        "active_users": active_users,
        "total_attendance_logs": total_logs,
        "today_attendance": today_logs,
        "cooldown_minutes": settings.ATTENDANCE_COOLDOWN_MINUTES,
        "similarity_threshold": settings.SIMILARITY_THRESHOLD,
    }


@router.post(
    "/rebuild-index",
    response_model=MessageResponse,
    summary="Rebuild FAISS index from database",
    dependencies=[Depends(verify_admin_key)],
)
def rebuild_index(db: Session = Depends(get_db)):
    """Force rebuild the FAISS index from PostgreSQL embeddings."""
    faiss_manager.rebuild_from_db(db)
    return MessageResponse(
        success=True,
        message=f"FAISS index rebuilt with {faiss_manager.total_embeddings} embeddings.",
    )


@router.delete(
    "/clear-today",
    response_model=MessageResponse,
    summary="Clear today's attendance (for testing)",
    dependencies=[Depends(verify_admin_key)],
)
def clear_today_attendance(db: Session = Depends(get_db)):
    """Delete all attendance logs for today. Useful during testing."""
    deleted = (
        db.query(AttendanceLog)
        .filter(AttendanceLog.attendance_date == date.today())
        .delete()
    )
    db.commit()
    return MessageResponse(
        success=True,
        message=f"Cleared {deleted} attendance records for today.",
    )


@router.post(
    "/re-enroll/{user_id}",
    response_model=MessageResponse,
    summary="Delete enrollment for re-enrollment",
    dependencies=[Depends(verify_admin_key)],
)
def re_enroll_user(user_id: int, db: Session = Depends(get_db)):
    """
    Delete all face embeddings for a user so they can re-enroll.
    Does NOT delete the user account.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found.",
        )

    deleted = (
        db.query(FacialEmbedding)
        .filter(FacialEmbedding.user_id == user_id)
        .delete()
    )
    db.commit()

    faiss_manager.remove_user(user_id)
    faiss_manager.save_to_disk()

    return MessageResponse(
        success=True,
        message=f"Deleted {deleted} embeddings for {user.name}. Ready for re-enrollment.",
    )


@router.delete(
    "/user/{user_id}/full-delete",
    response_model=MessageResponse,
    summary="Permanently delete a user and all data",
    dependencies=[Depends(verify_admin_key)],
)
def full_delete_user(user_id: int, db: Session = Depends(get_db)):
    """Permanently delete a user, their embeddings, and attendance logs."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found.",
        )

    name = user.name

    # Cascade deletes embeddings and logs (defined in model relationships)
    db.delete(user)
    db.commit()

    faiss_manager.remove_user(user_id)
    faiss_manager.save_to_disk()

    return MessageResponse(
        success=True,
        message=f"Permanently deleted user '{name}' and all associated data.",
    )
