"""
FaceEngine — Singleton wrapper around InsightFace for face detection
and 512D embedding extraction.

Uses the lightweight `buffalo_sc` model (~30MB) optimized for CPU inference
on low-resource hardware (4GB RAM, Intel i3).

Usage:
    from app.services.face_engine import face_engine

    face_engine.initialize()                    # Call once at startup
    faces = face_engine.detect_faces(frame)     # Detect faces in a frame
    embedding = face_engine.extract_embedding(frame, faces[0])  # Get 512D vector
"""

import numpy as np

from app.config import settings


class FaceEngine:
    """
    Singleton face detection and embedding extraction engine.

    The model is loaded once via `initialize()` and reused across
    all requests to minimize memory usage on constrained hardware.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_model"):
            self._model = None
            self._model_name = settings.FACE_MODEL

    def initialize(self) -> None:
        """
        Load the InsightFace model. Call this once at application startup.

        On an Intel i3-5005U this takes ~5-8 seconds for first load
        (model download + ONNX session creation).
        """
        if self._initialized:
            print("[FACE ENGINE] Already initialized, skipping.")
            return

        # Import here to avoid slow import at module level
        from insightface.app import FaceAnalysis

        print(f"[FACE ENGINE] Loading model '{self._model_name}'...")
        print("[FACE ENGINE] First run will download the model (~30MB)...")

        self._model = FaceAnalysis(
            name=self._model_name,
            providers=["CPUExecutionProvider"],
        )

        # Detection size 320x320 for faster inference on CPU
        # Smaller = faster (~200ms per frame vs ~500ms at 640)
        self._model.prepare(ctx_id=-1, det_size=(320, 320))

        self._initialized = True
        print("[FACE ENGINE] Model loaded successfully.")

    @property
    def is_initialized(self) -> bool:
        """Check if the engine has been initialized."""
        return self._initialized

    def _ensure_initialized(self) -> None:
        """Raise an error if the engine hasn't been initialized."""
        if not self._initialized:
            raise RuntimeError(
                "FaceEngine not initialized. Call face_engine.initialize() first."
            )

    def detect_faces(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> list:
        """
        Detect all faces in a BGR image frame.

        Args:
            frame: BGR image as numpy array (from OpenCV).
            confidence_threshold: Minimum detection confidence (0.0 to 1.0).

        Returns:
            List of face objects, each containing:
              - face.bbox: [x1, y1, x2, y2] bounding box
              - face.det_score: detection confidence
              - face.embedding: 512D embedding vector (numpy array)
        """
        self._ensure_initialized()

        faces = self._model.get(frame)

        # Filter by confidence
        faces = [f for f in faces if f.det_score >= confidence_threshold]

        # Sort by face area (largest first) — most prominent face comes first
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

        return faces

    def extract_embedding(self, frame: np.ndarray, face) -> np.ndarray:
        """
        Extract the 512D embedding vector from a detected face.

        InsightFace computes embeddings during detect_faces() (via model.get()),
        so this method simply retrieves the pre-computed embedding and
        normalizes it for cosine similarity search.

        Args:
            frame: The original BGR frame (not used, kept for API consistency).
            face: A face object returned by detect_faces().

        Returns:
            Normalized 512D numpy array (float32).
        """
        self._ensure_initialized()

        embedding = face.embedding

        if embedding is None:
            raise ValueError("Face object does not contain an embedding.")

        # L2-normalize for cosine similarity (FAISS IndexFlatIP)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def get_best_face(self, frame: np.ndarray, confidence_threshold: float = 0.5):
        """
        Convenience method: detect faces and return the largest (most prominent) one.

        Args:
            frame: BGR image as numpy array.
            confidence_threshold: Minimum detection confidence.

        Returns:
            The largest face object, or None if no face detected.
        """
        faces = self.detect_faces(frame, confidence_threshold)
        return faces[0] if faces else None

    def get_embedding_for_frame(self, frame: np.ndarray, confidence_threshold: float = 0.5):
        """
        All-in-one: detect the best face and extract its embedding.

        Args:
            frame: BGR image as numpy array.
            confidence_threshold: Minimum detection confidence.

        Returns:
            Tuple of (embedding, face) or (None, None) if no face found.
        """
        face = self.get_best_face(frame, confidence_threshold)
        if face is None:
            return None, None

        embedding = self.extract_embedding(frame, face)
        return embedding, face


# ── Singleton instance ────────────────────────────────────────
# Import this across the app: `from app.services.face_engine import face_engine`
face_engine = FaceEngine()
