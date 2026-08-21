# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: local.py
# Description: Embeddings computed in this process by sentence-transformers.
#   No API, no key, and nothing leaves the machine once the model is present.
# =================================================================================

# type hints
from typing import List, Optional

from .base import EmbeddingProvider

# Small, fast, and the model most people start from. A local backend needs a
# default that is actually local-shaped; an OpenAI model name would fail here.
DEFAULT_LOCAL_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


# SentenceTransformerEmbedding class
class SentenceTransformerEmbedding(EmbeddingProvider):
    '''
    In-process embeddings. The model is loaded once and reused, because loading
    costs far more than encoding.
    '''
    def __init__(
        self,
        model: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
        device: Optional[str] = None,
        encoder: Optional[object] = None
    ) -> None:
        '''
        Args:
            model (str): A sentence-transformers model name, or a path to one \
                already on disk — which is what a genuinely offline machine \
                needs, since a bare name is fetched on first use.
            device (str, optional): Torch device, e.g. 'cpu' or 'cuda'. \
                Defaults to whatever sentence-transformers picks.
            encoder (object, optional): A prepared encoder, for tests.

        Raises:
            ImportError: If sentence-transformers is not installed.
        '''
        self.model = model

        if encoder is not None:
            self.encoder = encoder
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError(
                "the sentence-transformers provider needs the "
                "'sentence-transformers' package: pip install osintgpt[local]"
            ) from error

        self.encoder = SentenceTransformer(model, device=device)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # Batching is the encoder's job; it already groups by length.
        vectors = self.encoder.encode(texts)

        # Returns a NumPy array, which is not what the interface promises.
        return [
            vector.tolist() if hasattr(vector, 'tolist') else list(vector)
            for vector in vectors
        ]
