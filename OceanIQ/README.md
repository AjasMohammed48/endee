# 🌊 OceanIQ — Ocean Science Research Assistant

> A RAG-powered research discovery platform built on **Endee vector database** — find research papers, Wikipedia articles, PubMed journals and patents about ocean science through semantic search and AI-generated answers.

---

## 📌 Problem Statement

Ocean science research is scattered across dozens of sources — Wikipedia, academic journals, PubMed, patent databases. Researchers and students waste hours manually searching each source separately, with no way to find *semantically related* content across all of them at once.

**OceanIQ solves this** by ingesting ocean science knowledge into Endee's vector database, enabling similarity-based retrieval across all sources in a single query — and using a local LLM (phi3:mini) to synthesise the results into a clear, cited answer.

---

## 🎯 Use Cases Demonstrated

| Use Case | Implementation |
|---|---|
| **Semantic Search** | Query → embedding → Endee similarity search → ranked results |
| **RAG (Retrieval Augmented Generation)** | Retrieved context → phi3:mini → grounded AI answer |
| **Multi-source Discovery** | Wikipedia + Semantic Scholar + PubMed + Google Patents |

---

## 🏗️ System Design

```
┌─────────────────────────────────────────────────────────────┐
│                        USER QUERY                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              EMBEDDER  (all-MiniLM-L6-v2)                   │
│         Converts query text → 384-dim vector                │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴────────────┐
          ▼                        ▼
┌──────────────────┐    ┌─────────────────────────────────────┐
│  ENDEE VECTOR DB │    │         WEB SOURCES (live)          │
│  (Docker)        │    │  • Wikipedia REST API               │
│                  │    │  • Semantic Scholar API             │
│  Vector search   │    │  • PubMed / NCBI API                │
│  → top-k docs    │    │  • Google Patents (search link)     │
└────────┬─────────┘    └──────────────┬──────────────────────┘
          │                            │
          └───────────────┬────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE  (rag.py)                    │
│                                                             │
│  🔍 Semantic Mode → Extractive summary from top results     │
│  🤖 AI Mode      → phi3:mini via Ollama generates answer    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              STREAMLIT UI  (app.py)                         │
│  Dark theme · Chat interface · Source cards · CSV export    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🗄️ How Endee is Used

Endee is the **core vector database** powering all search in OceanIQ.

### Ingestion
Ocean science articles from Wikipedia are embedded using `all-MiniLM-L6-v2` and inserted into Endee as vectors with metadata (title, text, source, topic):

```python
# ingest.py
client = EndeeClient()
model  = SentenceTransformer("all-MiniLM-L6-v2")

for article in articles:
    vector = model.encode(article["text"]).tolist()
    client.insert(vector=vector, metadata={
        "title":  article["title"],
        "text":   article["text"],
        "source": "Wikipedia",
        "topic":  article["topic"],
    })
```

### Querying
Every search query is embedded into the same vector space and queried against Endee for the top-k most similar documents:

```python
# search.py
def local_search(query: str, k: int = 5):
    model  = get_model()
    vector = model.encode(query).tolist()
    client = EndeeClient()
    return client.search(vector=vector, top_k=k)
```

Endee returns results with **similarity scores** which OceanIQ uses to display confidence labels (Strong / Good / Weak / Low) on each result card.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Vector Database | **Endee** (Docker) |
| Embedding Model | `all-MiniLM-L6-v2` (sentence-transformers, CPU) |
| LLM | `phi3:mini` via Ollama (local, no GPU needed) |
| UI Framework | Streamlit |
| Web Sources | Wikipedia API, Semantic Scholar, PubMed, Google Patents |
| Language | Python 3.10+ |

---

## ⚙️ Setup & Installation

### Prerequisites
- Docker Desktop installed and running
- Python 3.10+
- Ollama installed → https://ollama.com/download

### 1. Clone the repo
```bash
git clone https://github.com/AjasMohammed48/endee.git
cd endee/OceanIQ
```

### 2. Start Endee vector database
```bash
docker pull endeeio/endee
docker run -d -p 6333:6333 endeeio/endee
```

### 3. Install Python dependencies
```bash
pip install streamlit sentence-transformers pandas requests
```

### 4. Install and start phi3:mini (AI Summary mode)
```bash
ollama pull phi3:mini
ollama serve
```

### 5. Ingest ocean science data into Endee
```bash
cd src
python ingest.py
```

### 6. Run the app
```bash
streamlit run App.py
```

Open your browser at `http://localhost:8501`

---

## 🚀 Features

- **Semantic Search** — Find ocean science content by meaning, not just keywords
- **AI Summary** — phi3:mini generates grounded answers from retrieved context
- **Multi-source Results** — Wikipedia, Semantic Scholar, PubMed and Google Patents in one place
- **Confidence Scoring** — Every local result shows a similarity score from Endee
- **Search History** — Sidebar tracks recent queries with one-click re-search
- **CSV Export** — Download any search results as a CSV file
- **No GPU needed** — Everything runs on CPU

---

## 📁 Project Structure

```
OceanIQ/
├── src/
│   ├── App.py               # Streamlit UI — chat interface, sidebar, result cards
│   ├── embedder.py          # Singleton sentence-transformer model loader
│   ├── search.py            # Endee vector search + confidence labelling
│   ├── rag.py               # Answer generation (phi3:mini + extractive fallback)
│   ├── endee_client.py      # Endee HTTP client wrapper
│   ├── research_sources.py  # Wikipedia, Scholar, PubMed, Patents API calls
│   ├── ingest.py            # Data ingestion pipeline into Endee
│   └── wiki.py              # Wikipedia data fetcher for ingestion
├── data/
│   └── ocean_wikipedia.csv  # Ocean science articles dataset
└── README.md
```

---

## 📸 Screenshot

> OceanIQ running with Semantic Search and AI Summary modes

## 📸 Screenshots

### 🔍 Semantic Search Mode
![Semantic Search](https://raw.githubusercontent.com/AjasMohammed48/endee/master/OceanIQ/assets/Screenshot%202026-03-14%20025908.png)

### 🤖 AI Summary Mode
![AI Summary](https://raw.githubusercontent.com/AjasMohammed48/endee/master/OceanIQ/assets/Screenshot%202026-03-14%20033829.png)
---

## 👤 Author

**Ajas Mohammed**
GitHub: [@AjasMohammed48](https://github.com/AjasMohammed48)

---

*Built as part of the Endee Vector Database project evaluation.*
