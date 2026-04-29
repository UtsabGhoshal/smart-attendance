"""
Camera — OpenCV webcam helper for capturing frames.

Opens and closes the webcam per session to avoid lock issues on Windows.

Usage:
    from app.utils.camera import Camera

    with Camera() as cam:
        frame = cam.capture_frame()
"""

import cv2
import numpy as np


class Camera:
    """
    Simple webcam wrapper using OpenCV.

    Supports context manager (with statement) for clean resource management.
    Defaults to the built-in laptop webcam (index 0).
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """
        Args:
            camera_index: Webcam device index (0 = built-in laptop cam).
            width: Capture resolution width.
            height: Capture resolution height.
        """
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._cap = None

    def open(self) -> None:
        """Open the webcam connection."""
        if self._cap is not None and self._cap.isOpened():
            return  # Already open

        self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam at index {self._camera_index}. "
                "Make sure no other application is using the camera."
            )

        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        # Reduce buffer to get fresh frames (not stale buffered ones)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def capture_frame(self) -> np.ndarray:
        """
        Capture a single frame from the webcam.

        Returns:
            BGR image as a numpy array (height, width, 3).

        Raises:
            RuntimeError: If webcam is not open or frame capture fails.
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera is not open. Call open() first.")

        ret, frame = self._cap.read()

        if not ret or frame is None:
            raise RuntimeError("Failed to capture frame from webcam.")

        return frame

    def is_open(self) -> bool:
        """Check if the webcam is currently open."""
        return self._cap is not None and self._cap.isOpened()

    def close(self) -> None:
        """Release the webcam."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── Context Manager ───────────────────────────────────

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()
