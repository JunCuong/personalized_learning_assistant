"""Sentence-transformer embedding wrapper."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._model.get_sentence_embedding_dimension())
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()
