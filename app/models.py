"""
SQLAlchemy ORM models for the Smart Attendance System.

Tables:
  - users: Student/staff info
  - facial_embeddings: Embedding metadata (actual vectors live in FAISS)
  - attendance_logs: Check-in records with confidence scores
"""

from datetime import datetime, date
from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    Float,
    Date,
    DateTime,
    LargeBinary,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    """Registered user (student or staff)."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    roll_number = Column(String(50), unique=True, nullable=False, index=True)
    department = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    embeddings = relationship(
        "FacialEmbedding", back_populates="user", cascade="all, delete-orphan"
    )
    attendance_logs = relationship(
        "AttendanceLog", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, name='{self.name}', roll='{self.roll_number}')>"


class FacialEmbedding(Base):
    """
    Face embedding record.

    The actual 512D vector is stored as raw bytes (numpy .tobytes()) in the
    `embedding` column for persistence. At runtime, FAISS handles all
    similarity search — this table is the source of truth for rebuilding
    the FAISS index on startup.
    """

    __tablename__ = "facial_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # 512 * 4 bytes = 2048 bytes
    sample_number = Column(Integer, nullable=False)  # 1 to 5
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="embeddings")

    def __repr__(self) -> str:
        return f"<FacialEmbedding(id={self.id}, user_id={self.user_id}, sample={self.sample_number})>"


class AttendanceLog(Base):
    """
    Attendance check-in record.

    A unique constraint on (user_id, attendance_date) prevents duplicate
    daily entries. Cooldown logic (1 hour) is handled in the service layer.
    """

    __tablename__ = "attendance_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    attendance_date = Column(Date, default=date.today, nullable=False)
    check_in_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    confidence_score = Column(Float, nullable=False)

    # Relationship
    user = relationship("User", back_populates="attendance_logs")

    def __repr__(self) -> str:
        return f"<AttendanceLog(id={self.id}, user_id={self.user_id}, date={self.attendance_date})>"
