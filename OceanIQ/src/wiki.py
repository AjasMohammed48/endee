import wikipedia
import pandas as pd
from sentence_transformers import SentenceTransformer
from endee_client import EndeeClient

TOPICS = [
    "ARGO float", "Ocean acidification", "Coral reef",
    "Marine biology", "Tsunami", "El Niño Southern Oscillation",
    "Deep sea fish", "Ocean current", "Sea level rise",
    "Marine pollution", "Whale", "Phytoplankton",
    "Hydrothermal vent", "Ocean thermal energy", "Seawater",
]


def fetch_articles() -> pd.DataFrame:
    rows = []
    for i, topic in enumerate(TOPICS):
        try:
            page      = wikipedia.page(topic)
            sentences = page.content.split(". ")
            chunks    = [". ".join(sentences[j:j+3]) for j in range(0, len(sentences), 3)]

            for k, chunk in enumerate(chunks[:10]):
                rows.append({
                    "id":     f"{i}_{k}",
                    "title":  page.title,
                    "text":   chunk.strip(),
                    "topic":  topic.lower().replace(" ", "_"),
                    "source": "Wikipedia",
                })
            print(f"✓ {page.title}")
        except Exception as e:
            print(f"✗ Skipped {topic}: {e}")

    return pd.DataFrame(rows)


def ingest_to_endee(df: pd.DataFrame):
    model   = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client  = EndeeClient()
    vectors = []

    for _, row in df.iterrows():
        vectors.append({
            "id":     f"wiki_{row['id']}",
            "vector": [float(v) for v in model.encode(row["text"])],
            "meta": {
                "title":  str(row["title"]),
                "text":   str(row["text"]),
                "topic":  str(row["topic"]),
                "source": str(row["source"]),
            },
            "filter": {}
        })

        if len(vectors) == 50:
            client.insert_vectors(vectors)
            print(f"Inserted batch of 50")
            vectors = []

    if vectors:
        client.insert_vectors(vectors)

    print(f"Ingestion complete. Total: {client._index.describe()}")


if __name__ == "__main__":
    df = fetch_articles()
    df.to_csv("../data/ocean_wikipedia.csv", index=False)
    print(f"\nSaved {len(df)} chunks to ocean_wikipedia.csv")
    ingest_to_endee(df)