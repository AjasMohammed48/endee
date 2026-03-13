import pandas as pd
from sentence_transformers import SentenceTransformer
from endee_client import EndeeClient


def main():
    print("Loading dataset...")
    df = pd.read_csv("../data/ocean_wikipedia.csv")

    print("Loading embedding model...")
    model  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client = EndeeClient()

    vectors = []

    for i, row in df.iterrows():
        text      = str(row["text"])
        embedding = [float(v) for v in model.encode(text)]

        vectors.append({
            "id":     f"doc_{i}",
            "vector": embedding,
            "meta": {
                "title":  str(row.get("title", "")),
                "text":   text,
                "topic":  str(row.get("topic", "")),
                "source": str(row.get("source", "")),
            },
            "filter": {}
        })

        if len(vectors) == 50:
            response = client.insert_vectors(vectors)
            print(f"Inserted up to row {i}: {response}")
            vectors = []

    if vectors:
        response = client.insert_vectors(vectors)
        print(f"Final batch inserted: {response}")

    print("Ingestion complete.")


if __name__ == "__main__":
    main()