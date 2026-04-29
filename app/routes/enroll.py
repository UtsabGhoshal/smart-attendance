"""
Enrollment routes — Capture face samples and manage enrollment status.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import EnrollmentStatus, MessageResponse
from app.services.enrollment import EnrollmentService

router = APIRouter(prefix="/api/enroll", tags=["Enrollment"])


@router.post(
    "/{user_id}",
    response_model=EnrollmentStatus,
    summary="Enroll a user's face",
)
def enroll_user(user_id: int, db: Session = Depends(get_db)):
    """
    Start face enrollment for a user.

    Opens the webcam, captures 5 face samples with angle guidance,
    extracts 512D embeddings, and stores them in PostgreSQL + FAISS.

    The server will print guidance messages to the console:
    1. Look STRAIGHT at the camera
    2. Turn your head slightly to the LEFT
    3. Turn your head slightly to the RIGHT
    4. Tilt your head slightly UP
    5. Tilt your head slightly DOWN

    Each sample has a ~2 second pause for you to adjust your position.
    """
    service = EnrollmentService(db)

    try:
        result = service.enroll_user(user_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return EnrollmentStatus(**result)


@router.get(
    "/{user_id}/status",
    response_model=EnrollmentStatus,
    summary="Check enrollment status",
)
def enrollment_status(user_id: int, db: Session = Depends(get_db)):
    """Check how many face samples have been captured for a user."""
    service = EnrollmentService(db)
    result = service.get_enrollment_status(user_id)

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result["error"],
        )

    return EnrollmentStatus(
        user_id=result["user_id"],
        user_name=result["user_name"],
        samples_captured=result["samples_captured"],
        samples_required=result["samples_required"],
        is_complete=result["is_complete"],
        message=(
            "Enrollment complete"
            if result["is_complete"]
            else f"{result['samples_required'] - result['samples_captured']} more samples needed"
        ),
    )


@router.delete(
    "/{user_id}",
    response_model=MessageResponse,
    summary="Delete enrollment (allow re-enrollment)",
)
def delete_enrollment(user_id: int, db: Session = Depends(get_db)):
    """
    Delete all face embeddings for a user.

    This allows the user to re-enroll with fresh face samples.
    The user account itself is NOT deleted.
    """
    service = EnrollmentService(db)

    try:
        result = service.delete_enrollment(user_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return MessageResponse(success=result["success"], message=result["message"])
