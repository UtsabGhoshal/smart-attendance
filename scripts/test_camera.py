"""
test_camera.py — Quick visual test for webcam + face detection + embedding.

Run this script to verify:
  1. Your webcam opens correctly
  2. InsightFace detects faces
  3. Embeddings are extracted (512D vectors)

Usage:
    cd "smart attendance"
    .\\venv\\Scripts\\activate
    python scripts/test_camera.py

Controls:
    q / ESC  — Quit
    s        — Print embedding for current face (snapshot)
"""

import sys
import os

# Add project root to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from app.services.face_engine import face_engine
from app.utils.camera import Camera


def draw_face_box(frame: np.ndarray, face, label: str = "") -> None:
    """Draw a bounding box and label on a face in the frame."""
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label with confidence
    confidence = face.det_score
    text = f"{label} ({confidence:.2f})" if label else f"Conf: {confidence:.2f}"

    # Background for text
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(
        frame,
        (x1, y1 - text_size[1] - 10),
        (x1 + text_size[0] + 4, y1),
        (0, 255, 0),
        -1,
    )
    cv2.putText(
        frame, text, (x1 + 2, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
    )


def main():
    print("=" * 50)
    print("  Smart Attendance — Camera & Face Detection Test")
    print("=" * 50)

    # Step 1: Initialize FaceEngine
    print("\n[1/3] Initializing FaceEngine (first run downloads model ~30MB)...")
    face_engine.initialize()
    print("[OK] FaceEngine ready.\n")

    # Step 2: Open webcam
    print("[2/3] Opening webcam...")
    cam = Camera()
    cam.open()
    print("[OK] Webcam opened.\n")

    # Step 3: Live preview loop
    print("[3/3] Starting live preview...")
    print("       Press 'q' or ESC to quit")
    print("       Press 's' to print embedding of current face\n")

    frame_count = 0
    faces = []

    try:
        while True:
            frame = cam.capture_frame()
            frame_count += 1

            # Process every 3rd frame for performance
            if frame_count % 3 == 0:
                faces = face_engine.detect_faces(frame)

            # Draw boxes for detected faces
            for i, face in enumerate(faces):
                draw_face_box(frame, face, label=f"Face {i + 1}")

            # Show face count on screen
            status = f"Faces: {len(faces)} | Frame: {frame_count} | Press 'q' to quit"
            cv2.putText(
                frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

            cv2.imshow("Smart Attendance - Test Camera", frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # q or ESC
                print("\n[QUIT] Exiting...")
                break

            elif key == ord("s") and faces:
                # Snapshot: print embedding info
                best_face = faces[0]
                embedding = face_engine.extract_embedding(frame, best_face)
                print(f"\n--- Embedding Snapshot ---")
                print(f"  Shape:     {embedding.shape}")
                print(f"  Dtype:     {embedding.dtype}")
                print(f"  L2 norm:   {np.linalg.norm(embedding):.4f}")
                print(f"  Min/Max:   {embedding.min():.4f} / {embedding.max():.4f}")
                print(f"  First 5:   {embedding[:5]}")
                print(f"--------------------------\n")

    finally:
        cam.close()
        cv2.destroyAllWindows()
        print("[OK] Webcam released. Test complete.")


if __name__ == "__main__":
    main()
