"""
Enrollment Service — Live webcam preview with ML head pose detection.

Uses InsightFace facial landmarks + OpenCV solvePnP to estimate
real-time head orientation (yaw, pitch, roll). Auto-captures only
when the user's head is in the correct position.
"""

import time
import math
import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.models import User, FacialEmbedding
from app.services.face_engine import face_engine
from app.services.faiss_index import faiss_manager
from app.utils.camera import Camera


# ── 3D Generic Face Model Points ─────────────────────────────
# Corresponds to the 5 landmarks InsightFace returns:
#   [0] left_eye, [1] right_eye, [2] nose, [3] left_mouth, [4] right_mouth
MODEL_POINTS_3D = np.array([
    (-30.0, -30.0, -30.0),   # Left eye
    (30.0, -30.0, -30.0),    # Right eye
    (0.0, 0.0, 0.0),         # Nose tip
    (-20.0, 30.0, -25.0),    # Left mouth corner
    (20.0, 30.0, -25.0),     # Right mouth corner
], dtype=np.float64)


# ── Pose Requirements for Each Capture Step ───────────────────
# Each step defines the required yaw/pitch range (in degrees)
CAPTURE_STEPS = [
    {
        "instruction": "Look STRAIGHT at the camera",
        "short": "STRAIGHT",
        "yaw_range": (-12, 12),
        "pitch_range": (-12, 12),
    },
    {
        "instruction": "Turn your head to the LEFT",
        "short": "LEFT",
        "yaw_range": (-45, -12),
        "pitch_range": (-20, 20),
    },
    {
        "instruction": "Turn your head to the RIGHT",
        "short": "RIGHT",
        "yaw_range": (12, 45),
        "pitch_range": (-20, 20),
    },
    {
        "instruction": "Tilt your head UP",
        "short": "UP",
        "yaw_range": (-20, 20),
        "pitch_range": (-45, -10),
    },
    {
        "instruction": "Tilt your head DOWN",
        "short": "DOWN",
        "yaw_range": (-20, 20),
        "pitch_range": (10, 45),
    },
]

# ── UI Constants ──────────────────────────────────────────────
WINDOW_NAME = "Smart Attendance - Face Enrollment"
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
DARK_GREEN = (0, 180, 0)
GRAY = (80, 80, 80)

STABILITY_DURATION = 1.5   # Seconds face must be in correct pose
STABILITY_THRESHOLD = 15.0  # Max pixel movement to count as "stable"


def estimate_head_pose(landmarks_2d: np.ndarray, frame_shape: tuple) -> tuple:
    """
    Estimate head pose (yaw, pitch, roll) from 5 facial landmarks
    using the PnP (Perspective-n-Point) algorithm.

    Args:
        landmarks_2d: 5x2 array of facial landmark pixel coordinates.
        frame_shape: (height, width, channels) of the frame.

    Returns:
        (yaw, pitch, roll) in degrees.
        yaw: negative=left, positive=right
        pitch: negative=up, positive=down
        roll: head tilt
    """
    h, w = frame_shape[:2]

    # Approximate camera intrinsics from frame dimensions
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    points_2d = landmarks_2d.astype(np.float64)

    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS_3D, points_2d, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_SQPNP,
    )

    if not success:
        return 0.0, 0.0, 0.0

    # Convert rotation vector to rotation matrix, then to Euler angles
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)

    # Decompose rotation matrix to Euler angles
    sy = math.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(rotation_mat[2, 1], rotation_mat[2, 2])
        yaw = math.atan2(-rotation_mat[2, 0], sy)
        roll = math.atan2(rotation_mat[1, 0], rotation_mat[0, 0])
    else:
        pitch = math.atan2(-rotation_mat[1, 2], rotation_mat[1, 1])
        yaw = math.atan2(-rotation_mat[2, 0], sy)
        roll = 0

    # Convert to degrees
    yaw_deg = math.degrees(yaw)
    pitch_deg = math.degrees(pitch)
    roll_deg = math.degrees(roll)

    return yaw_deg, pitch_deg, roll_deg


def check_pose_match(yaw: float, pitch: float, step: dict) -> tuple:
    """
    Check if the current head pose matches the required pose for a step.

    Returns:
        (is_match, feedback_message)
    """
    yaw_min, yaw_max = step["yaw_range"]
    pitch_min, pitch_max = step["pitch_range"]

    yaw_ok = yaw_min <= yaw <= yaw_max
    pitch_ok = pitch_min <= pitch <= pitch_max

    if yaw_ok and pitch_ok:
        return True, "CORRECT! Hold still..."

    # Generate helpful feedback
    if step["short"] == "STRAIGHT":
        if yaw < yaw_min:
            return False, "Turn more to the RIGHT"
        if yaw > yaw_max:
            return False, "Turn more to the LEFT"
        if pitch < pitch_min:
            return False, "Tilt your head DOWN a bit"
        if pitch > pitch_max:
            return False, "Tilt your head UP a bit"
    elif step["short"] == "LEFT":
        if yaw > yaw_min:
            return False, f"Turn more LEFT (yaw: {yaw:.0f}, need < {yaw_min})"
    elif step["short"] == "RIGHT":
        if yaw < yaw_max:
            return False, f"Turn more RIGHT (yaw: {yaw:.0f}, need > {yaw_max})"
    elif step["short"] == "UP":
        if pitch > pitch_min:
            return False, f"Tilt more UP (pitch: {pitch:.0f})"
    elif step["short"] == "DOWN":
        if pitch < pitch_max:
            return False, f"Tilt more DOWN (pitch: {pitch:.0f})"

    return False, "Adjust your head position"


class EnrollmentService:
    """Handles face enrollment with ML head pose verification."""

    def __init__(self, db: Session):
        self.db = db

    def get_enrollment_status(self, user_id: int) -> dict:
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            return {"error": "User not found", "user_id": user_id}
        count = (
            self.db.query(FacialEmbedding)
            .filter(FacialEmbedding.user_id == user_id)
            .count()
        )
        return {
            "user_id": user.id,
            "user_name": user.name,
            "samples_captured": count,
            "samples_required": settings.ENROLLMENT_SAMPLES,
            "is_complete": count >= settings.ENROLLMENT_SAMPLES,
        }

    def enroll_user(self, user_id: int) -> dict:
        """Full enrollment with live preview + ML head pose verification."""
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"User with ID {user_id} not found")
        if not user.is_active:
            raise ValueError(f"User '{user.name}' is deactivated")

        existing = (
            self.db.query(FacialEmbedding)
            .filter(FacialEmbedding.user_id == user_id)
            .count()
        )
        if existing >= settings.ENROLLMENT_SAMPLES:
            raise ValueError(
                f"User '{user.name}' is already enrolled with "
                f"{existing} samples. Delete existing enrollment first."
            )

        print(f"\n[ENROLL] Starting enrollment for: {user.name} (ID: {user_id})")
        print(f"[ENROLL] ML head pose detection active")
        print(f"[ENROLL] Press Q or ESC to cancel.\n")

        result = self._live_capture_loop(user)

        if result is None:
            raise ValueError("Enrollment cancelled by user.")
        if len(result) < settings.ENROLLMENT_SAMPLES:
            raise ValueError(f"Only {len(result)} samples captured.")

        # Store in DB + FAISS
        print(f"[ENROLL] Storing {len(result)} embeddings...")
        for sample_num, embedding in result:
            self.db.add(FacialEmbedding(
                user_id=user_id,
                embedding=embedding.tobytes(),
                sample_number=sample_num,
            ))
            faiss_manager.add_embedding(embedding, user_id)

        self.db.commit()
        faiss_manager.save_to_disk()

        print(f"[ENROLL] Enrollment complete for {user.name}!")
        print(f"[ENROLL] Total embeddings in FAISS: {faiss_manager.total_embeddings}\n")

        return {
            "success": True,
            "user_id": user.id,
            "user_name": user.name,
            "samples_captured": len(result),
            "samples_required": settings.ENROLLMENT_SAMPLES,
            "is_complete": True,
            "message": f"Successfully enrolled {user.name} with {len(result)} face samples.",
        }

    def _live_capture_loop(self, user: User) -> list | None:
        """Live webcam loop with ML head pose verification."""
        captured = []
        current_step = 0
        total_steps = settings.ENROLLMENT_SAMPLES

        # Stability tracking
        pose_correct_since = None
        last_face_center = None
        flash_until = 0

        with Camera() as cam:
            # Create window and force to foreground
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 640, 480)
            cv2.moveWindow(WINDOW_NAME, 100, 100)
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

            # Warmup
            for _ in range(15):
                f = cam.capture_frame()
                wf = f.copy()
                h, w = wf.shape[:2]
                cv2.rectangle(wf, (0, 0), (w, h), BLACK, -1)
                t = "Initializing ML head pose detection..."
                ts = cv2.getTextSize(t, FONT, 0.7, 2)[0]
                cv2.putText(wf, t, ((w-ts[0])//2, h//2), FONT, 0.7, WHITE, 2)
                cv2.imshow(WINDOW_NAME, wf)
                cv2.waitKey(33)

            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0)

            while current_step < total_steps:
                frame = cam.capture_frame()
                display = frame.copy()
                step = CAPTURE_STEPS[current_step]
                now = time.time()

                # ── Detect face ───────────────────────────
                faces = face_engine.detect_faces(frame, confidence_threshold=0.5)
                face = faces[0] if faces else None

                yaw, pitch, roll = 0.0, 0.0, 0.0
                pose_ok = False
                feedback = "No face detected"

                if face is not None:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    face_center = ((x1+x2)/2, (y1+y2)/2)

                    # ── ML Head Pose Estimation ───────────
                    if face.kps is not None and len(face.kps) == 5:
                        yaw, pitch, roll = estimate_head_pose(face.kps, frame.shape)
                        pose_ok, feedback = check_pose_match(yaw, pitch, step)

                    # ── Stability check ───────────────────
                    is_stable = False
                    if pose_ok:
                        if last_face_center is not None:
                            dx = abs(face_center[0] - last_face_center[0])
                            dy = abs(face_center[1] - last_face_center[1])
                            mov = (dx**2 + dy**2) ** 0.5
                            if mov < STABILITY_THRESHOLD:
                                if pose_correct_since is None:
                                    pose_correct_since = now
                                elif now - pose_correct_since >= STABILITY_DURATION:
                                    is_stable = True
                            else:
                                pose_correct_since = now
                        else:
                            pose_correct_since = now
                    else:
                        pose_correct_since = None

                    last_face_center = face_center

                    # ── Draw face box ─────────────────────
                    box_color = GREEN if pose_ok else RED
                    cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)

                    # Pose indicator icon
                    icon = "OK" if pose_ok else "X"
                    icon_color = GREEN if pose_ok else RED
                    cv2.putText(display, icon, (x2+5, y1+20), FONT, 0.7, icon_color, 2)

                    # Show yaw/pitch angles next to face
                    cv2.putText(display, f"Yaw: {yaw:.1f}", (x2+5, y1+45), FONT, 0.4, WHITE, 1)
                    cv2.putText(display, f"Pitch: {pitch:.1f}", (x2+5, y1+65), FONT, 0.4, WHITE, 1)

                    # ── Stability progress bar ────────────
                    if pose_correct_since is not None and not is_stable:
                        elapsed = now - pose_correct_since
                        progress = min(elapsed / STABILITY_DURATION, 1.0)
                        bw = x2 - x1
                        by = y2 + 15
                        cv2.rectangle(display, (x1, by), (x1+bw, by+10), GRAY, -1)
                        cv2.rectangle(display, (x1, by), (x1+int(bw*progress), by+10), GREEN, -1)
                        cv2.putText(display, "Hold still...", (x1, by+25), FONT, 0.4, GREEN, 1)

                    # ── Auto-capture ──────────────────────
                    if is_stable and now > flash_until:
                        embedding = face_engine.extract_embedding(frame, face)
                        current_step += 1
                        captured.append((current_step, embedding))
                        print(f"  [OK] Sample {current_step}/{total_steps} captured "
                              f"({step['short']}) [yaw={yaw:.1f}, pitch={pitch:.1f}]")
                        flash_until = now + 1.2
                        pose_correct_since = None
                        last_face_center = None
                else:
                    last_face_center = None
                    pose_correct_since = None

                # ── Draw UI ───────────────────────────────
                h, w = display.shape[:2]

                # Flash overlay
                if now < flash_until:
                    ov = display.copy()
                    cv2.rectangle(ov, (0, 0), (w, h), DARK_GREEN, -1)
                    cv2.addWeighted(ov, 0.3, display, 0.7, 0, display)
                    t = f"CAPTURED! ({len(captured)}/{total_steps})"
                    ts = cv2.getTextSize(t, FONT, 1.2, 3)[0]
                    cv2.putText(display, t, ((w-ts[0])//2, h//2), FONT, 1.2, WHITE, 3)
                else:
                    # Top bar
                    cv2.rectangle(display, (0, 0), (w, 55), BLACK, -1)
                    cv2.putText(display, f"ENROLLMENT: {user.name}", (10, 20), FONT, 0.5, WHITE, 1)
                    cv2.putText(display, f"Sample {min(current_step+1,total_steps)}/{total_steps}",
                                (10, 45), FONT, 0.55, GREEN, 2)

                    # Progress dots
                    for i in range(total_steps):
                        c = GREEN if i < len(captured) else GRAY
                        cx = w - 30*total_steps - 10 + i*30 + 15
                        cv2.circle(display, (cx, 30), 8, c, -1)
                        cv2.putText(display, str(i+1), (cx-4, 34), FONT, 0.35, WHITE, 1)

                    # Bottom bar
                    cv2.rectangle(display, (0, h-75), (w, h), BLACK, -1)

                    # Instruction
                    if current_step < total_steps:
                        inst = f">> {step['instruction']} <<"
                    else:
                        inst = ">> All captures done! <<"
                    ts = cv2.getTextSize(inst, FONT, 0.6, 2)[0]
                    cv2.putText(display, inst, ((w-ts[0])//2, h-50), FONT, 0.6, YELLOW, 2)

                    # Feedback
                    fb_color = GREEN if pose_ok else RED
                    ts = cv2.getTextSize(feedback, FONT, 0.45, 1)[0]
                    cv2.putText(display, feedback, ((w-ts[0])//2, h-22), FONT, 0.45, fb_color, 1)

                    # Quit hint
                    cv2.putText(display, "Q=cancel", (w-90, h-50), FONT, 0.35, (150,150,150), 1)

                cv2.imshow(WINDOW_NAME, display)

                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    print("[ENROLL] Cancelled by user.")
                    cv2.destroyAllWindows()
                    return None
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        return None
                except cv2.error:
                    return None

            # Completion screen
            for _ in range(60):
                f = cam.capture_frame()
                d = f.copy()
                h, w = d.shape[:2]
                ov = d.copy()
                cv2.rectangle(ov, (0,0), (w,h), DARK_GREEN, -1)
                cv2.addWeighted(ov, 0.4, d, 0.6, 0, d)
                for i, (t, s, th) in enumerate([
                    ("ENROLLMENT COMPLETE!", 1.0, 3),
                    (f"{user.name} - {total_steps} samples", 0.6, 2),
                    ("Window closing...", 0.45, 1),
                ]):
                    ts = cv2.getTextSize(t, FONT, s, th)[0]
                    cv2.putText(d, t, ((w-ts[0])//2, h//2-30+i*40), FONT, s, WHITE, th)
                cv2.imshow(WINDOW_NAME, d)
                if cv2.waitKey(33) & 0xFF in (ord("q"), 27):
                    break

        cv2.destroyAllWindows()
        return captured

    def delete_enrollment(self, user_id: int) -> dict:
        """Delete all embeddings for a user (allows re-enrollment)."""
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"User with ID {user_id} not found")

        deleted = (
            self.db.query(FacialEmbedding)
            .filter(FacialEmbedding.user_id == user_id)
            .delete()
        )
        self.db.commit()
        faiss_manager.remove_user(user_id)
        faiss_manager.save_to_disk()

        print(f"[ENROLL] Deleted {deleted} embeddings for {user.name}")
        return {
            "success": True,
            "user_id": user.id,
            "user_name": user.name,
            "deleted_samples": deleted,
            "message": f"Deleted {deleted} embeddings for {user.name}. User can now re-enroll.",
        }
