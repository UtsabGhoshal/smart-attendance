"""
Recognition Service — Detects faces and matches them against enrolled users.

Pipeline:
  Frame → Face Detection → Extract Embedding → FAISS Search → User Identity

Usage:
    from app.services.recognition import RecognitionService

    service = RecognitionService()
    result = service.recognize_from_frame(frame)
"""

import numpy as np

from app.config import settings
from app.services.face_engine import face_engine
from app.services.faiss_index import faiss_manager


class RecognitionResult:
    """Result of a face recognition attempt."""

    def __init__(
        self,
        recognized: bool,
        user_id: int | None = None,
        confidence: float = 0.0,
        face_bbox: tuple | None = None,
    ):
        self.recognized = recognized
        self.user_id = user_id
        self.confidence = confidence
        self.face_bbox = face_bbox  # (x1, y1, x2, y2)

    def __repr__(self) -> str:
        if self.recognized:
            return f"<Recognized user_id={self.user_id} conf={self.confidence:.2f}>"
        return "<Unknown face>"


class RecognitionService:
    """Handles face recognition against enrolled users."""

    def recognize_from_frame(
        self, frame: np.ndarray, threshold: float | None = None
    ) -> list[RecognitionResult]:
        """
        Detect and recognize all faces in a single frame.

        Args:
            frame: BGR image from webcam.
            threshold: Min similarity score (uses settings default if None).

        Returns:
            List of RecognitionResult, one per detected face.
        """
        if threshold is None:
            threshold = settings.SIMILARITY_THRESHOLD

        faces = face_engine.detect_faces(frame, confidence_threshold=0.5)

        if not faces:
            return []

        results = []
        for face in faces:
            bbox = tuple(face.bbox.astype(int))
            embedding = face_engine.extract_embedding(frame, face)

            user_id, score = faiss_manager.search_best_match(
                embedding, threshold=threshold
            )

            results.append(RecognitionResult(
                recognized=user_id is not None,
                user_id=user_id,
                confidence=float(score),
                face_bbox=bbox,
            ))

        return results

    def recognize_single(
        self, frame: np.ndarray, threshold: float | None = None
    ) -> RecognitionResult | None:
        """
        Recognize the best (largest) face in a frame.

        Returns:
            RecognitionResult for the best face, or None if no face detected.
        """
        results = self.recognize_from_frame(frame, threshold)

        if not results:
            return None

        # Return the one with highest confidence
        return max(results, key=lambda r: r.confidence)
