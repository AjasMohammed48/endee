from sentence_transformers import SentenceTransformer
from typing import List

_model = None

def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed_text(text: str) -> List[float]:
    """Convert a single string into a vector."""
    model = get_model()
    return [float(v) for v in model.encode(text)]

def embed_batch(texts: List[str]) -> List[List[float]]:
    """Convert a list of strings into vectors."""
    model = get_model()
    return [[float(v) for v in vec] for vec in model.encode(texts)]