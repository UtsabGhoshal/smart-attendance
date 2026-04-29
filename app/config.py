"""
Application configuration loaded from environment variables.
Uses python-dotenv to read from .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    """Central configuration class for the Smart Attendance System."""

    # ── Database ──────────────────────────────────────────────
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "attendance")
    DB_USER: str = os.getenv("DB_USER", "utsab")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "admin")

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # ── Application ───────────────────────────────────────────
    APP_NAME: str = os.getenv("APP_NAME", "Smart Attendance System")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    # ── Face Recognition ──────────────────────────────────────
    FACE_MODEL: str = os.getenv("FACE_MODEL", "buffalo_sc")
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.4"))
    ENROLLMENT_SAMPLES: int = int(os.getenv("ENROLLMENT_SAMPLES", "5"))

    # ── FAISS ─────────────────────────────────────────────────
    DATA_DIR: Path = BASE_DIR / "data"
    FAISS_INDEX_PATH: Path = BASE_DIR / os.getenv("FAISS_INDEX_PATH", "data/faiss_index.bin")
    FAISS_MAPPING_PATH: Path = BASE_DIR / os.getenv("FAISS_MAPPING_PATH", "data/faiss_mapping.json")

    # ── Attendance ────────────────────────────────────────────
    ATTENDANCE_COOLDOWN_MINUTES: int = int(os.getenv("ATTENDANCE_COOLDOWN_MINUTES", "30"))
    ADMIN_API_KEY: str = os.getenv("ADMIN_API_KEY", "admin-secret-key-2026")

    def ensure_data_dir(self) -> None:
        """Create the data directory if it doesn't exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)


# Singleton settings instance used across the app
settings = Settings()
