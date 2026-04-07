🏎️ PitWall — F1 Strategy Intelligence System

PitWall is an AI-powered Formula 1 race strategy intelligence system built using Retrieval-Augmented Generation (RAG).

It leverages real telemetry data from the FastF1 API, semantic search via FAISS, and Groq’s Llama 3.3 70B to analyze past race strategies and forecast upcoming Grand Prix outcomes with reasoning.

🚀 Why PitWall?

Formula 1 strategy is one of the most complex real-time decision systems in sports — involving tyre degradation, pit timing, race pace, and unpredictable events.

PitWall aims to make this intelligence accessible using AI, enabling users to:

Understand why strategies worked
Explore historical race decisions
Get data-grounded forecasts for upcoming races
✨ Features
Race Analyzer
Ask natural language questions about any past F1 race strategy
Race Forecaster
Get data-driven predictions for upcoming GPs based on historical patterns
Real F1 Data
Powered by FastF1 (lap times, tyre compounds, stint lengths, pit stops)
Strategic Reasoning
Explains why strategies were used — not just what happened
Web-Augmented Forecasting
Injects current driver lineups and team form using live web search
🧠 Example Queries
Q: Why did Verstappen pit early in Silverstone 2023?
A: Red Bull opted for an early undercut due to high front tyre degradation at Silverstone...
Q: What strategy is likely at Monaco?
A: Historically, Monaco favors one-stop strategies due to low overtaking probability...
🏗️ Architecture
FastF1 Data → Chunking → Embeddings (SBERT) → FAISS Index
        → Semantic Retrieval → LLM (Llama 3.3 70B via Groq)
        → Strategy Explanation / Forecast
⚙️ Tech Stack
Layer	Technology
F1 Data	FastF1
Embeddings	sentence-transformers (all-MiniLM-L6-v2)
Vector Store	FAISS
LLM	Groq (Llama 3.3 70B)
Web Search	Serper API
UI	Streamlit
📂 Project Structure
pitwall/
│── src/
│   ├── data/
│   │   └── loader.py        # FastF1 data fetching
│   ├── rag/
│   │   ├── embedder.py      # Chunk creation + FAISS indexing
│   │   └── retriever.py     # Semantic retrieval
│   └── llm/
│       └── explainer.py     # Groq LLM + web search
│── app.py                   # Streamlit UI
│── requirements.txt
🛠️ Setup
1. Clone the repository
git clone https://github.com/Shubham270206/PitWall.git
cd PitWall
2. Create virtual environment
python -m venv venv
Activate environment:
Windows
venv\Scripts\activate
Mac/Linux
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Add API keys

Create a .env file:

GROQ_API_KEY=your_groq_key
SERPER_API_KEY=your_serper_key
5. Run the app
streamlit run app.py
⚠️ Accuracy & Limitations

PitWall is most reliable for seasons 2018–2025, where FastF1 provides consistent, high-quality telemetry data.

🔻 Reduced accuracy for 2026+
New power unit regulations and active aerodynamics
Changing tyre behavior and stint patterns
Team performance shifts
Driver transfers affecting historical consistency
✅ Still reliable:
Circuit characteristics (e.g., Monaco, Silverstone, Baku patterns)
Historical strategic tendencies
❗ Additional limitations:
Does not account for qualifying results
Cannot predict safety cars, mechanical failures, or sudden weather changes
Wet race data is noisier
Forecasts are probabilistic, not deterministic
🧭 Framing

PitWall is a strategy intelligence system — not a race predictor.

Its outputs are:

Data-grounded
Historically informed
Strategically reasoned

This mirrors how real F1 strategists think:
👉 probabilities, trade-offs, and informed decisions — not certainty