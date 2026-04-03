import os
from groq import Groq
from dotenv import load_dotenv
from src.rag.retriever import retrieve

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


def explain(query: str, index, chunks: list[str]) -> str:
    context_chunks = retrieve(query, index, chunks, top_k=6)
    context = "\n".join(f"- {c}" for c in context_chunks)

    prompt = f"""You are PitWall, an expert F1 race strategy analyst.
Use ONLY the context below to answer. Do not invent lap times or results.
If the context doesn't have enough info, say so honestly.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content


def forecast(circuit: str, year: int, index, chunks: list[str]) -> str:
    query = f"Predict the race winner and likely strategy for {circuit} {year} based on historical patterns"
    context_chunks = retrieve(query, index, chunks, top_k=8)
    context = "\n".join(f"- {c}" for c in context_chunks)

    prompt = f"""You are PitWall, an expert F1 race strategy forecaster.
Based on historical race data from previous {circuit} Grands Prix, predict:
1. The most likely winning strategy (tyre compounds and stint lengths)
2. Which drivers/teams are likely to contend for the win
3. Key strategic risks at this circuit

Use ONLY the context below. Be clear this is a data-driven forecast, not a guarantee.

Context:
{context}

Forecast for {circuit} {year}:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    from src.rag.embedder import load_index
    index, chunks = load_index("cache/monaco_index")

    print("=== ANALYZER ===")
    print(explain("What was Leclerc's strategy at Monaco 2024?", index, chunks))

    print("\n=== FORECASTER ===")
    print(forecast("Monaco", 2025, index, chunks))