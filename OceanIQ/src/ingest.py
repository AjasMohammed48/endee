import pandas as pd
from sentence_transformers import SentenceTransformer
from endee_client import EndeeClient

def main():
    print("Loading dataset...")
    df = pd.read_csv("../data/dataset.csv")

    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    client = EndeeClient()

    vectors = []

    for i, row in df.iterrows():  # remove .head(10) for full dataset
        text = str(row["text"])
        embedding = model.encode(text)
        embedding = [float(v) for v in embedding]  # this is correct, keep it

        vectors.append({
            "id": f"doc_{i}",
            "vector": embedding,
            "meta": {
                "title": str(row.get("title", "")),
                "text": text,
                "topic": str(row.get("topic", "")),
                "source": str(row.get("source", "")),
            },
            "filter": {}
        })

        # Insert in batches of 50 to avoid large payloads
        if len(vectors) == 50:
            response = client.insert_vectors(vectors)
            print(f"Batch inserted up to row {i}: {response}")
            vectors = []

    # Insert remaining
    if vectors:
        response = client.insert_vectors(vectors)
        print(f"Final batch inserted: {response}")

    print("Ingestion complete!")

if __name__ == "__main__":
    main()