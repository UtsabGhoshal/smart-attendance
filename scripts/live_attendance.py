"""
Live Attendance Mode — Continuous face recognition via webcam.

Runs a live webcam feed that:
  - Detects all faces in real-time
  - Recognizes enrolled users via FAISS
  - Logs attendance automatically (with 30-min cooldown)
  - Shows "Unknown" for unrecognized faces (no logging)
  - Displays today's attendance count

Usage:
    python scripts/live_attendance.py

Press Q or ESC to quit.
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from app.config import settings
from app.database import SessionLocal
from app.services.face_engine import face_engine
from app.services.faiss_index import faiss_manager
from app.services.recognition import RecognitionService
from app.services.attendance import AttendanceService

# ── UI Constants ──────────────────────────────────────────────
WINDOW_NAME = "Smart Attendance - Live Mode"
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
GRAY = (150, 150, 150)


def draw_face_result(frame, bbox, name, confidence, is_known, logged_msg=None):
    """Draw face bounding box with name and confidence."""
    x1, y1, x2, y2 = bbox

    if is_known:
        color = GREEN
        label = f"{name} ({confidence:.0%})"
    else:
        color = RED
        label = f"Unknown ({confidence:.0%})"

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Name label background
    text_size = cv2.getTextSize(label, FONT, 0.6, 2)[0]
    cv2.rectangle(
        frame,
        (x1, y1 - text_size[1] - 10),
        (x1 + text_size[0] + 6, y1),
        color, -1,
    )
    cv2.putText(frame, label, (x1 + 3, y1 - 5), FONT, 0.6, WHITE, 2)

    # Logged message below box
    if logged_msg:
        cv2.putText(frame, logged_msg, (x1, y2 + 20), FONT, 0.4, CYAN, 1)


def draw_header(frame, total_present, total_enrolled):
    """Draw the top status bar."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 45), BLACK, -1)

    cv2.putText(frame, "SMART ATTENDANCE SYSTEM", (10, 18), FONT, 0.5, WHITE, 1)
    cv2.putText(frame, "LIVE MODE", (10, 38), FONT, 0.5, GREEN, 2)

    stats = f"Present: {total_present}/{total_enrolled}"
    ts = cv2.getTextSize(stats, FONT, 0.55, 2)[0]
    cv2.putText(frame, stats, (w - ts[0] - 10, 30), FONT, 0.55, YELLOW, 2)


def draw_footer(frame):
    """Draw bottom hints."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 30), (w, h), BLACK, -1)
    cv2.putText(frame, "Press Q to quit | Cooldown: 30 min", (10, h - 10), FONT, 0.4, GRAY, 1)


def main():
    print("=" * 55)
    print("  Smart Attendance System — Live Recognition Mode")
    print("=" * 55)

    # Initialize services
    print("\n[1/3] Loading face engine...")
    face_engine.initialize()
    print("[OK] Face engine ready.")

    print("\n[2/3] Loading FAISS index from database...")
    db = SessionLocal()
    faiss_manager.rebuild_from_db(db)
    print(f"[OK] FAISS index: {faiss_manager.total_embeddings} embeddings")

    if faiss_manager.total_embeddings == 0:
        print("\n[WARNING] No enrolled users! Enroll users first via the API.")
        print("  POST /api/users → POST /api/enroll/{{user_id}}")

    recognition = RecognitionService()
    attendance = AttendanceService(db)

    print("\n[3/3] Opening webcam...")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam!")
        return

    print("[OK] Webcam ready. Starting live recognition...\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 100, 100)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

    # Brief warmup
    for _ in range(5):
        cap.read()
        cv2.waitKey(30)

    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0)

    # Track recent log messages per user (for display)
    recent_messages = {}  # user_id → (message, timestamp)
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            frame_count += 1

            # Run recognition every 3rd frame (save CPU on i3)
            results = []
            if frame_count % 3 == 0:
                results = recognition.recognize_from_frame(frame)

                for r in results:
                    if r.recognized and r.user_id is not None:
                        # Try to log attendance
                        log_result = attendance.log_attendance(r.user_id, r.confidence)

                        if log_result["logged"]:
                            msg = f"Checked in at {log_result['check_in_time'].strftime('%H:%M')}"
                            recent_messages[r.user_id] = (msg, time.time())
                        elif "reason" in log_result:
                            recent_messages[r.user_id] = (log_result["reason"][:50], time.time())

            # Draw faces (use last results if we skipped this frame)
            if results:
                # Fetch user names
                from app.models import User

                for r in results:
                    if r.face_bbox:
                        name = "Unknown"
                        if r.recognized and r.user_id:
                            user = db.query(User).filter(User.id == r.user_id).first()
                            name = user.name if user else f"User {r.user_id}"

                        # Get recent log message
                        logged_msg = None
                        if r.user_id and r.user_id in recent_messages:
                            msg, ts = recent_messages[r.user_id]
                            if time.time() - ts < 10:  # Show for 10 seconds
                                logged_msg = msg

                        draw_face_result(
                            display, r.face_bbox, name, r.confidence,
                            r.recognized, logged_msg,
                        )

            # Stats
            stats = attendance.get_today_stats()
            draw_header(display, stats["total_present"], stats["total_enrolled"])
            draw_footer(display)

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break

            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        db.close()
        print("\n[OK] Live attendance stopped.")
        print(f"Today's stats: {stats['total_present']}/{stats['total_enrolled']} present")


if __name__ == "__main__":
    main()
