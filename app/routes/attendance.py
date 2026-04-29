"""
Attendance routes — One-shot recognition + attendance history.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import AttendanceMarkResponse, AttendanceLogResponse, AttendanceLogsListResponse
from app.services.recognition import RecognitionService
from app.services.attendance import AttendanceService
from app.utils.camera import Camera

router = APIRouter(prefix="/api/attendance", tags=["Attendance"])


@router.post(
    "/recognize",
    response_model=AttendanceMarkResponse,
    summary="One-shot: recognize face and log attendance",
)
def recognize_and_log(db: Session = Depends(get_db)):
    """
    Opens webcam, captures a frame, detects and recognizes the face,
    and logs attendance if the person is enrolled.

    This is a one-shot operation — the webcam opens and closes automatically.
    Unknown faces are shown as 'Unknown' and NOT logged.
    """
    recognition = RecognitionService()
    attendance = AttendanceService(db)

    # Capture a few frames (let camera warm up) and use the last one
    with Camera() as cam:
        frame = None
        for _ in range(10):  # Warm up camera
            frame = cam.capture_frame()

    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not capture frame from webcam.",
        )

    result = recognition.recognize_single(frame)

    if result is None:
        return AttendanceMarkResponse(
            success=False,
            message="No face detected in the frame. Try again.",
        )

    if not result.recognized:
        return AttendanceMarkResponse(
            success=False,
            confidence=result.confidence,
            message=f"Unknown person (confidence: {result.confidence:.2f}, threshold: {0.4}). Not logged.",
        )

    # Log attendance
    log_result = attendance.log_attendance(result.user_id, result.confidence)

    if log_result["logged"]:
        return AttendanceMarkResponse(
            success=True,
            user_id=log_result["user_id"],
            user_name=log_result["user_name"],
            roll_number=log_result["roll_number"],
            confidence=result.confidence,
            check_in_time=log_result["check_in_time"],
            message=f"Attendance logged for {log_result['user_name']}.",
        )
    else:
        return AttendanceMarkResponse(
            success=False,
            user_id=log_result.get("user_id"),
            user_name=log_result.get("user_name"),
            confidence=result.confidence,
            message=log_result["reason"],
        )


@router.get(
    "/today",
    response_model=AttendanceLogsListResponse,
    summary="Get today's attendance",
)
def get_today_attendance(db: Session = Depends(get_db)):
    """Get all attendance records for today."""
    service = AttendanceService(db)
    logs = service.get_today_logs()
    return AttendanceLogsListResponse(
        logs=[AttendanceLogResponse(**log) for log in logs],
        total=len(logs),
    )


@router.get(
    "/user/{user_id}",
    response_model=AttendanceLogsListResponse,
    summary="Get user's attendance history",
)
def get_user_attendance(
    user_id: int, limit: int = 30, db: Session = Depends(get_db)
):
    """Get attendance history for a specific user."""
    service = AttendanceService(db)
    logs = service.get_user_history(user_id, limit)

    if not logs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No attendance records found for user {user_id}.",
        )

    return AttendanceLogsListResponse(
        logs=[AttendanceLogResponse(**log) for log in logs],
        total=len(logs),
    )


@router.get(
    "/stats",
    summary="Get today's attendance stats",
)
def get_attendance_stats(db: Session = Depends(get_db)):
    """Get summary statistics for today's attendance."""
    service = AttendanceService(db)
    return service.get_today_stats()
