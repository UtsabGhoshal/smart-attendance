"""
FAISS Index Manager — Fast vector similarity search for face embeddings.

Uses FAISS IndexFlatIP (inner product) on L2-normalized embeddings,
which is equivalent to cosine similarity.

With 10 users × 5 samples = 50 embeddings, exact search is instant.

Usage:
    from app.services.faiss_index import faiss_manager

    faiss_manager.initialize(dimension=512)
    faiss_manager.add_embedding(embedding, user_id)
    user_id, score = faiss_manager.search(query_embedding)
"""

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from app.config import settings


class FAISSManager:
    """
    Manages the FAISS index for fast face embedding search.

    Maintains two files on disk:
      - faiss_index.bin:   The FAISS index (binary)
      - faiss_mapping.json: Maps index position → user_id

    On startup, the index is rebuilt from PostgreSQL embeddings.
    """

    def __init__(self):
        self._index: Optional[faiss.IndexFlatIP] = None
        self._dimension: int = 512
        # Maps FAISS internal position (0, 1, 2, ...) → user_id
        self._id_mapping: list[int] = []
        self._initialized: bool = False

    def initialize(self, dimension: int = 512) -> None:
        """
        Create an empty FAISS index.

        Args:
            dimension: Embedding vector dimension (512 for InsightFace).
        """
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._id_mapping = []
        self._initialized = True
        print(f"[FAISS] Initialized empty index (dim={dimension})")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def total_embeddings(self) -> int:
        """Number of embeddings currently in the index."""
        if self._index is None:
            return 0
        return self._index.ntotal

    def add_embedding(self, embedding: np.ndarray, user_id: int) -> None:
        """
        Add a single embedding to the FAISS index.

        Args:
            embedding: L2-normalized 512D float32 vector.
            user_id: The user this embedding belongs to.
        """
        self._ensure_initialized()

        # Reshape to (1, dimension) for FAISS
        vector = embedding.reshape(1, -1).astype(np.float32)
        self._index.add(vector)
        self._id_mapping.append(user_id)

    def add_embeddings_batch(
        self, embeddings: list[np.ndarray], user_ids: list[int]
    ) -> None:
        """
        Add multiple embeddings at once (used during index rebuild).

        Args:
            embeddings: List of L2-normalized 512D vectors.
            user_ids: Corresponding user IDs.
        """
        self._ensure_initialized()

        if not embeddings:
            return

        matrix = np.vstack(embeddings).astype(np.float32)
        self._index.add(matrix)
        self._id_mapping.extend(user_ids)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[int, float]]:
        """
        Search for the nearest embeddings to the query.

        Args:
            query_embedding: L2-normalized 512D float32 vector.
            top_k: Number of nearest neighbors to return.

        Returns:
            List of (user_id, similarity_score) tuples, sorted by score descending.
            Score is cosine similarity (0.0 to 1.0, higher = more similar).
        """
        self._ensure_initialized()

        if self._index.ntotal == 0:
            return []

        # Clamp top_k to the number of embeddings we have
        k = min(top_k, self._index.ntotal)

        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._id_mapping):
                results.append((self._id_mapping[idx], float(score)))

        return results

    def search_best_match(
        self, query_embedding: np.ndarray, threshold: float = None
    ) -> tuple[Optional[int], float]:
        """
        Find the single best matching user.

        Args:
            query_embedding: L2-normalized 512D vector.
            threshold: Minimum similarity score to consider a match.
                       Uses settings.SIMILARITY_THRESHOLD if not provided.

        Returns:
            (user_id, score) if match found above threshold, else (None, 0.0).
        """
        if threshold is None:
            threshold = settings.SIMILARITY_THRESHOLD

        results = self.search(query_embedding, top_k=10)

        if not results:
            return None, 0.0

        # Group scores by user_id and average them
        user_scores: dict[int, list[float]] = {}
        for user_id, score in results:
            user_scores.setdefault(user_id, []).append(score)

        # Find user with highest average score
        best_user = None
        best_avg_score = 0.0

        for user_id, scores in user_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_user = user_id

        # Apply threshold (cosine similarity: higher = more similar)
        # Threshold of 0.4 means we need at least 0.4 similarity
        if best_avg_score >= threshold:
            return best_user, best_avg_score

        return None, best_avg_score

    def remove_user(self, user_id: int) -> None:
        """
        Remove all embeddings for a user by rebuilding the index.

        FAISS IndexFlatIP doesn't support deletion, so we rebuild.
        This is fine for ≤50 embeddings.
        """
        self._ensure_initialized()

        if not self._id_mapping:
            return

        # Get all embeddings that DON'T belong to this user
        keep_indices = [
            i for i, uid in enumerate(self._id_mapping) if uid != user_id
        ]

        if len(keep_indices) == len(self._id_mapping):
            return  # User not found in index

        if not keep_indices:
            # All embeddings belonged to this user — reset index
            self._index.reset()
            self._id_mapping = []
            return

        # Reconstruct vectors for indices we want to keep
        vectors = np.vstack(
            [self._index.reconstruct(i) for i in keep_indices]
        ).astype(np.float32)
        new_mapping = [self._id_mapping[i] for i in keep_indices]

        # Rebuild index
        self._index.reset()
        self._index.add(vectors)
        self._id_mapping = new_mapping

    def save_to_disk(self) -> None:
        """Persist the FAISS index and mapping to disk."""
        self._ensure_initialized()
        settings.ensure_data_dir()

        faiss.write_index(self._index, str(settings.FAISS_INDEX_PATH))

        with open(settings.FAISS_MAPPING_PATH, "w") as f:
            json.dump(self._id_mapping, f)

        print(
            f"[FAISS] Saved index ({self._index.ntotal} vectors) to disk"
        )

    def load_from_disk(self) -> bool:
        """
        Load FAISS index and mapping from disk.

        Returns:
            True if loaded successfully, False if files don't exist.
        """
        index_path = Path(settings.FAISS_INDEX_PATH)
        mapping_path = Path(settings.FAISS_MAPPING_PATH)

        if not index_path.exists() or not mapping_path.exists():
            return False

        self._index = faiss.read_index(str(index_path))
        self._dimension = self._index.d

        with open(mapping_path, "r") as f:
            self._id_mapping = json.load(f)

        self._initialized = True
        print(
            f"[FAISS] Loaded index from disk ({self._index.ntotal} vectors)"
        )
        return True

    def rebuild_from_db(self, db_session) -> None:
        """
        Rebuild the FAISS index from all embeddings in PostgreSQL.

        Called at startup to ensure the index is in sync with the database.

        Args:
            db_session: SQLAlchemy session.
        """
        from app.models import FacialEmbedding

        self.initialize(self._dimension)

        records = db_session.query(FacialEmbedding).all()

        if not records:
            print("[FAISS] No embeddings in database, starting fresh")
            return

        embeddings = []
        user_ids = []

        for record in records:
            vector = np.frombuffer(record.embedding, dtype=np.float32)
            embeddings.append(vector)
            user_ids.append(record.user_id)

        self.add_embeddings_batch(embeddings, user_ids)
        self.save_to_disk()

        print(
            f"[FAISS] Rebuilt index from DB: "
            f"{len(records)} embeddings for "
            f"{len(set(user_ids))} users"
        )

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "FAISSManager not initialized. Call initialize() first."
            )


# ── Singleton instance ────────────────────────────────────────
faiss_manager = FAISSManager()
