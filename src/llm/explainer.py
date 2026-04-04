import os
import requests
from groq import Groq
from dotenv import load_dotenv
from src.rag.retriever import retrieve

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
SERPER_KEY = os.getenv("SERPER_API_KEY")
MODEL = "llama-3.3-70b-versatile"


def _web_search(query: str) -> str:
    if not SERPER_KEY:
        return ""
    try:
        r = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 5},
            timeout=5,
        )
        results = r.json().get("organic", [])
        snippets = [f"- {res['title']}: {res['snippet']}" for res in results if "snippet" in res]
        return "\n".join(snippets[:5])
    except Exception:
        return ""


def explain(query: str, index, chunks: list[str]) -> str:
    context_chunks = retrieve(query, index, chunks, top_k=6)
    context = "\n".join(f"- {c}" for c in context_chunks)

    prompt = f"""You are PitWall, an expert F1 race strategy analyst.
Use the context below to answer. Explain the strategic reasoning behind decisions —
why a compound was chosen, what the undercut/overcut logic was, how track position
played a role. Do not invent lap times or results not in the context.

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
    query = f"winning strategy and race results at {circuit}"
    context_chunks = retrieve(query, index, chunks, top_k=8)
    context = "\n".join(f"- {c}" for c in context_chunks)

    # fetch current season context from web
    web_context = ""
    if SERPER_KEY:
        search_results = _web_search(f"F1 {year} season driver lineups team performance standings")
        if search_results:
            web_context = f"\nCurrent season context (from web):\n{search_results}"

    prompt = f"""You are PitWall, an expert F1 race strategy forecaster.
Using historical race data AND current season context, predict:
1. The most likely winning strategy (compounds, stint lengths, number of stops) and WHY
2. Which drivers and teams are likely to contend for the win based on current form
3. Key strategic risks specific to this circuit
4. How current car/driver performance affects strategy choices

Be specific about strategic reasoning — explain WHY a strategy works at this circuit,
not just what it is. Make clear this is a data-driven forecast, not a guarantee.

Historical race data from previous {circuit} Grands Prix:
{context}
{web_context}

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
    print(explain("What was Leclerc's strategy at Monaco 2024 and why did it work?", index, chunks))

    print("\n=== FORECASTER ===")
    print(forecast("Monaco", 2025, index, chunks))