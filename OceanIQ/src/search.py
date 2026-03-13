from endee_client import EndeeClient
from embedder import embed_text
from typing import List, Dict, Any

_client = None


def get_client() -> EndeeClient:
    global _client
    if _client is None:
        _client = EndeeClient()
    return _client


def local_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    try:
        client = get_client()
        vector = embed_text(query)
        response = client.search(vector, k=k)
        return response.get("results", [])
    except Exception as e:
        print(f"Search error: {e}")
        return []


def confidence_label(similarity: float) -> tuple[str, str]:
    if similarity >= 0.75:
        return "Strong match", "#00c896"
    elif similarity >= 0.60:
        return "Good match", "#3b9eff"
    elif similarity >= 0.45:
        return "Weak match", "#f5a623"
    else:
        return "Low confidence", "#e74c3c"