"""
User CRUD routes — Register, list, get, and deactivate users.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.schemas import UserCreate, UserResponse, UserListResponse, MessageResponse

router = APIRouter(prefix="/api/users", tags=["Users"])


@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
def create_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new student/staff member.

    - **name**: Full name (e.g., "Utsab Sharma")
    - **roll_number**: Unique identifier (e.g., "CSE2023001")
    - **department**: Department name (e.g., "Computer Science")
    """
    # Check for duplicate roll number
    existing = db.query(User).filter(User.roll_number == user_data.roll_number).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with roll number '{user_data.roll_number}' already exists.",
        )

    user = User(
        name=user_data.name,
        roll_number=user_data.roll_number,
        department=user_data.department,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    print(f"[USERS] Created user: {user.name} (Roll: {user.roll_number})")
    return user


@router.get(
    "",
    response_model=UserListResponse,
    summary="List all users",
)
def list_users(
    skip: int = 0,
    limit: int = 50,
    active_only: bool = True,
    db: Session = Depends(get_db),
):
    """
    Get a list of all registered users.

    - **skip**: Number of records to skip (pagination)
    - **limit**: Maximum records to return
    - **active_only**: If true, only return active users
    """
    query = db.query(User)

    if active_only:
        query = query.filter(User.is_active == True)  # noqa: E712

    total = query.count()
    users = query.offset(skip).limit(limit).all()

    return UserListResponse(users=users, total=total)


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user details",
)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get details of a specific user by ID."""
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found.",
        )

    return user


@router.delete(
    "/{user_id}",
    response_model=MessageResponse,
    summary="Deactivate a user",
)
def deactivate_user(user_id: int, db: Session = Depends(get_db)):
    """
    Soft-delete a user by setting is_active=False.

    This does NOT delete their data — they can be reactivated later.
    """
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User '{user.name}' is already deactivated.",
        )

    user.is_active = False
    db.commit()

    print(f"[USERS] Deactivated user: {user.name} (ID: {user.id})")

    return MessageResponse(
        success=True,
        message=f"User '{user.name}' has been deactivated.",
    )
