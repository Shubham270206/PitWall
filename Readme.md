# 🏎️ PitWall — F1 Strategy Intelligence System

PitWall is an AI-powered F1 race strategy analyzer and forecaster built with RAG (Retrieval-Augmented Generation). It uses real telemetry data from the FastF1 API, semantic search via FAISS, and Groq's Llama 3.3 70B to explain historical race strategies and forecast upcoming GP outcomes.

## Features

- **Race Analyzer** — Ask natural language questions about any past F1 race strategy
- **Race Forecaster** — Get data-driven predictions for upcoming GPs based on historical patterns and current season context
- **Real F1 Data** — Powered by FastF1 (lap times, tyre compounds, stint lengths, pit stops)
- **Strategic Reasoning** — Explains *why* strategies were used, not just *what* they were
- **Web-Augmented Forecasting** — Injects current driver lineups and team form via live web search

## Tech Stack

| Layer | Technology |
|---|---|
| F1 Data | FastF1 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| LLM | Groq (Llama 3.3 70B) |
| Web Search | Serper API |
| UI | Streamlit |

## Project Structure
pitwall/
├── src/
│   ├── data/
│   │   └── loader.py       # FastF1 data fetching
│   ├── rag/
│   │   ├── embedder.py     # Chunk creation + FAISS indexing
│   │   └── retriever.py    # Semantic retrieval
│   └── llm/
│       └── explainer.py    # Groq LLM + web search
├── app.py                  # Streamlit UI
└── requirements.txt

## Setup
```bash
git clone https://github.com/Shubham270206/PitWall.git
cd PitWall
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:
GROQ_API_KEY=your_groq_key
SERPER_API_KEY=your_serper_key

Run the app:
```bash
streamlit run app.py
```

## How It Works

1. **Data Layer** — FastF1 pulls real lap-by-lap race data for any circuit and year
2. **Chunking** — Race data is converted into strategy-aware text chunks with reasoning context
3. **Embedding** — Chunks are embedded with SBERT and stored in a FAISS index
4. **Retrieval** — User queries are embedded and matched against the index semantically
5. **Generation** — Retrieved chunks + web context are passed to Llama 3.3 70B to generate grounded answers

## Accuracy & Limitations

PitWall is most reliable for seasons **2018–2025**. FastF1 provides consistent, detailed data across this window and historical patterns from these years form the core of the RAG index.

**For 2026 and beyond**, forecast accuracy drops significantly for several reasons:

- The 2026 regulations introduce a completely new power unit formula and active aerodynamics, meaning tyre behaviour and stint lengths from previous seasons are less applicable
- Team performance hierarchies have shifted — historical dominance patterns (e.g. Red Bull 2022–2024) are less predictive
- Driver lineup changes (e.g. Hamilton to Ferrari) mean team-driver historical combinations no longer apply cleanly

Circuit characteristics — Monaco favoring one-stoppers, Silverstone's high front tyre degradation, Baku's safety car frequency — remain valid across regulation cycles and still inform forecasts. But for post-2025 seasons, treat PitWall's output as a **strategic baseline grounded in circuit history**, not a confident prediction.

**Other limitations:**
- Does not account for qualifying results, which heavily influence race strategy
- Cannot predict mechanical failures, weather changes mid-race, or safety car timing
- Wet weather races produce noisier data in FastF1 which can affect chunk quality

## Framing

PitWall is a **strategy intelligence system**, not a race predictor. Forecasts are grounded in historical patterns and current season data — they reflect what the data suggests, not guaranteed outcomes. This framing is intentional: honest, defensible, and closer to how real F1 strategists actually think.