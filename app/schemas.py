"""
Pydantic schemas for request validation and response serialization.
"""

from datetime import datetime, date
from typing import Optional, List
from pydantic import BaseModel, Field


# ── User Schemas ──────────────────────────────────────────────

class UserCreate(BaseModel):
    """Schema for creating a new user."""
    name: str = Field(..., min_length=1, max_length=100, examples=["Utsab Sharma"])
    roll_number: str = Field(..., min_length=1, max_length=50, examples=["CSE2023001"])
    department: str = Field(..., min_length=1, max_length=100, examples=["Computer Science"])


class UserResponse(BaseModel):
    """Schema for user response data."""
    id: int
    name: str
    roll_number: str
    department: str
    created_at: datetime
    is_active: bool

    model_config = {"from_attributes": True}


class UserListResponse(BaseModel):
    """Schema for listing multiple users."""
    users: List[UserResponse]
    total: int


# ── Enrollment Schemas ────────────────────────────────────────

class EnrollmentStatus(BaseModel):
    """Schema for enrollment progress feedback."""
    user_id: int
    user_name: str
    samples_captured: int
    samples_required: int
    is_complete: bool
    message: str


# ── Attendance Schemas ────────────────────────────────────────

class AttendanceMarkResponse(BaseModel):
    """Schema for attendance marking result."""
    success: bool
    user_id: Optional[int] = None
    user_name: Optional[str] = None
    roll_number: Optional[str] = None
    confidence: Optional[float] = None
    check_in_time: Optional[datetime] = None
    message: str


class AttendanceLogResponse(BaseModel):
    """Schema for a single attendance log entry."""
    id: int
    user_id: int
    user_name: str
    roll_number: str
    attendance_date: date
    check_in_time: datetime
    confidence_score: float

    model_config = {"from_attributes": True}


class AttendanceLogsListResponse(BaseModel):
    """Schema for listing attendance logs."""
    logs: List[AttendanceLogResponse]
    total: int


# ── General Schemas ───────────────────────────────────────────

class MessageResponse(BaseModel):
    """Generic message response."""
    success: bool
    message: str
