"""
Attendance Service — Logs attendance with cooldown protection.

Prevents duplicate entries within the configured cooldown period (default: 30 min).
"""

from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session

from app.config import settings
from app.models import User, AttendanceLog


class AttendanceService:
    """Handles attendance logging with cooldown logic."""

    def __init__(self, db: Session):
        self.db = db

    def log_attendance(
        self, user_id: int, confidence_score: float
    ) -> dict:
        """
        Log a check-in for a user, respecting the cooldown period.

        Args:
            user_id: Recognized user's ID.
            confidence_score: FAISS similarity score (0-1).

        Returns:
            Dict with result info.
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            return {"logged": False, "reason": "User not found"}

        if not user.is_active:
            return {"logged": False, "reason": "User is deactivated"}

        # Check cooldown — was this user logged recently?
        cooldown_cutoff = datetime.utcnow() - timedelta(
            minutes=settings.ATTENDANCE_COOLDOWN_MINUTES
        )

        recent_log = (
            self.db.query(AttendanceLog)
            .filter(
                AttendanceLog.user_id == user_id,
                AttendanceLog.check_in_time >= cooldown_cutoff,
            )
            .order_by(AttendanceLog.check_in_time.desc())
            .first()
        )

        if recent_log:
            minutes_ago = (datetime.utcnow() - recent_log.check_in_time).seconds // 60
            return {
                "logged": False,
                "reason": f"Already checked in {minutes_ago} min ago (cooldown: {settings.ATTENDANCE_COOLDOWN_MINUTES} min)",
                "user_id": user.id,
                "user_name": user.name,
                "last_check_in": recent_log.check_in_time,
            }

        # Log new attendance
        now = datetime.utcnow()
        log_entry = AttendanceLog(
            user_id=user_id,
            attendance_date=date.today(),
            check_in_time=now,
            confidence_score=confidence_score,
        )

        try:
            self.db.add(log_entry)
            self.db.commit()
            self.db.refresh(log_entry)
        except Exception:
            self.db.rollback()
            # Likely unique constraint violation (same user, same date)
            # Update the existing record instead
            existing = (
                self.db.query(AttendanceLog)
                .filter(
                    AttendanceLog.user_id == user_id,
                    AttendanceLog.attendance_date == date.today(),
                )
                .first()
            )
            if existing:
                return {
                    "logged": False,
                    "reason": "Already logged today",
                    "user_id": user.id,
                    "user_name": user.name,
                }
            raise

        print(f"[ATTENDANCE] {user.name} checked in (confidence: {confidence_score:.2f})")

        return {
            "logged": True,
            "user_id": user.id,
            "user_name": user.name,
            "roll_number": user.roll_number,
            "check_in_time": now,
            "confidence_score": confidence_score,
        }

    def get_today_logs(self) -> list[dict]:
        """Get all attendance logs for today."""
        logs = (
            self.db.query(AttendanceLog)
            .filter(AttendanceLog.attendance_date == date.today())
            .order_by(AttendanceLog.check_in_time.desc())
            .all()
        )

        results = []
        for log in logs:
            user = self.db.query(User).filter(User.id == log.user_id).first()
            results.append({
                "id": log.id,
                "user_id": log.user_id,
                "user_name": user.name if user else "Unknown",
                "roll_number": user.roll_number if user else "N/A",
                "attendance_date": log.attendance_date,
                "check_in_time": log.check_in_time,
                "confidence_score": log.confidence_score,
            })

        return results

    def get_user_history(self, user_id: int, limit: int = 30) -> list[dict]:
        """Get attendance history for a specific user."""
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            return []

        logs = (
            self.db.query(AttendanceLog)
            .filter(AttendanceLog.user_id == user_id)
            .order_by(AttendanceLog.check_in_time.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "id": log.id,
                "user_id": log.user_id,
                "user_name": user.name,
                "roll_number": user.roll_number,
                "attendance_date": log.attendance_date,
                "check_in_time": log.check_in_time,
                "confidence_score": log.confidence_score,
            }
            for log in logs
        ]

    def get_today_stats(self) -> dict:
        """Get summary stats for today."""
        today = date.today()

        total_checkins = (
            self.db.query(AttendanceLog)
            .filter(AttendanceLog.attendance_date == today)
            .count()
        )

        total_enrolled = (
            self.db.query(User)
            .filter(User.is_active == True)  # noqa: E712
            .count()
        )

        return {
            "date": today,
            "total_present": total_checkins,
            "total_enrolled": total_enrolled,
            "attendance_rate": (
                f"{(total_checkins / total_enrolled * 100):.1f}%"
                if total_enrolled > 0 else "0%"
            ),
        }
