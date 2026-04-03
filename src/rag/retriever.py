import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query: str, index: faiss.Index, chunks: list[str], top_k: int = 5) -> list[str]:
    embedding = model.encode([query]).astype("float32")
    _, indices = index.search(embedding, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


if __name__ == "__main__":
    from src.rag.embedder import load_index
    index, chunks = load_index("cache/monaco_index")

    queries = [
        "Why did Leclerc pit so late at Monaco?",
        "What was the winning strategy at Monaco 2023?",
        "Which compound worked best at Monaco?",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        results = retrieve(q, index, chunks)
        for r in results:
            print(f"  - {r}")