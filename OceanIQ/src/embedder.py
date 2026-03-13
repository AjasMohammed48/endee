from sentence_transformers import SentenceTransformer
from typing import List

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(text: str) -> List[float]:
    return [float(v) for v in get_model().encode(text)]


def embed_batch(texts: List[str]) -> List[List[float]]:
    return [[float(v) for v in vec] for vec in get_model().encode(texts)]