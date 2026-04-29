"""
Reports routes — Daily/weekly reports and CSV export.
"""

import csv
import io
from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, AttendanceLog

router = APIRouter(prefix="/api/reports", tags=["Reports"])


@router.get("/daily", summary="Daily attendance report")
def daily_report(
    report_date: date = Query(default=None, description="Date (YYYY-MM-DD). Defaults to today."),
    db: Session = Depends(get_db),
):
    """Get detailed attendance report for a specific date."""
    if report_date is None:
        report_date = date.today()

    # Get all logs for the date
    logs = (
        db.query(AttendanceLog)
        .filter(AttendanceLog.attendance_date == report_date)
        .order_by(AttendanceLog.check_in_time.asc())
        .all()
    )

    # Get all active users
    total_enrolled = db.query(User).filter(User.is_active == True).count()  # noqa: E712
    present_ids = {log.user_id for log in logs}

    # Build present list
    present = []
    for log in logs:
        user = db.query(User).filter(User.id == log.user_id).first()
        present.append({
            "user_id": log.user_id,
            "name": user.name if user else "Unknown",
            "roll_number": user.roll_number if user else "N/A",
            "department": user.department if user else "N/A",
            "check_in_time": log.check_in_time.strftime("%H:%M:%S"),
            "confidence": round(log.confidence_score, 3),
        })

    # Build absent list
    absent_users = (
        db.query(User)
        .filter(User.is_active == True, ~User.id.in_(present_ids))  # noqa: E712
        .all()
    )
    absent = [
        {
            "user_id": u.id,
            "name": u.name,
            "roll_number": u.roll_number,
            "department": u.department,
        }
        for u in absent_users
    ]

    return {
        "date": str(report_date),
        "total_enrolled": total_enrolled,
        "total_present": len(present),
        "total_absent": len(absent),
        "attendance_rate": f"{(len(present) / total_enrolled * 100):.1f}%" if total_enrolled else "0%",
        "present": present,
        "absent": absent,
    }


@router.get("/weekly", summary="Weekly attendance summary")
def weekly_report(db: Session = Depends(get_db)):
    """Get attendance summary for the last 7 days."""
    today = date.today()
    days = []

    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        count = (
            db.query(AttendanceLog)
            .filter(AttendanceLog.attendance_date == d)
            .count()
        )
        total = db.query(User).filter(User.is_active == True).count()  # noqa: E712
        days.append({
            "date": str(d),
            "day": d.strftime("%A"),
            "present": count,
            "total_enrolled": total,
            "rate": f"{(count / total * 100):.1f}%" if total else "0%",
        })

    return {
        "period": f"{today - timedelta(days=6)} to {today}",
        "days": days,
        "average_rate": f"{sum(d['present'] for d in days) / max(sum(d['total_enrolled'] for d in days), 1) * 100:.1f}%",
    }


@router.get("/export", summary="Export attendance as CSV")
def export_csv(
    report_date: date = Query(default=None, description="Date (YYYY-MM-DD). Defaults to today."),
    db: Session = Depends(get_db),
):
    """Download attendance data as a CSV file."""
    if report_date is None:
        report_date = date.today()

    logs = (
        db.query(AttendanceLog)
        .filter(AttendanceLog.attendance_date == report_date)
        .order_by(AttendanceLog.check_in_time.asc())
        .all()
    )

    # Build CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["S.No", "Name", "Roll Number", "Department", "Check-in Time", "Confidence"])

    for i, log in enumerate(logs, 1):
        user = db.query(User).filter(User.id == log.user_id).first()
        writer.writerow([
            i,
            user.name if user else "Unknown",
            user.roll_number if user else "N/A",
            user.department if user else "N/A",
            log.check_in_time.strftime("%H:%M:%S"),
            f"{log.confidence_score:.3f}",
        ])

    # Also add absent users
    present_ids = {log.user_id for log in logs}
    absent = db.query(User).filter(User.is_active == True, ~User.id.in_(present_ids)).all()  # noqa: E712
    for j, u in enumerate(absent, len(logs) + 1):
        writer.writerow([j, u.name, u.roll_number, u.department, "ABSENT", ""])

    output.seek(0)
    filename = f"attendance_{report_date}.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
