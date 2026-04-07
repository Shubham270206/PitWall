import streamlit as st
from sentence_transformers import SentenceTransformer
from src.data.loader import load_multiple_races
from src.rag.embedder import build_index, load_index
from src.rag.retriever import retrieve
from src.llm.explainer import explain, forecast
from pathlib import Path


st.set_page_config(page_title="PitWall", page_icon="🏎️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;900&family=Barlow:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #0a0a0a;
    color: #f0f0f0;
}

.stApp { background-color: #0a0a0a; }

/* header */
.pitwall-header {
    background: linear-gradient(135deg, #e8002d 0%, #a00020 50%, #0a0a0a 100%);
    padding: 2.5rem 2rem 2rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    border-bottom: 3px solid #e8002d;
    position: relative;
    overflow: hidden;
}
.pitwall-header::before {
    content: "PitWall";
    position: absolute;
    right: -10px;
    top: -20px;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 9rem;
    font-weight: 900;
    color: rgba(255,255,255,0.04);
    letter-spacing: -4px;
    pointer-events: none;
}
.pitwall-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: 2px;
    color: #ffffff;
    text-transform: uppercase;
    line-height: 1;
    margin: 0;
}
.pitwall-subtitle {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 4px;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #111111;
    border-bottom: 2px solid #222;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #666;
    padding: 1rem 2rem;
    border-bottom: 3px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #e8002d !important;
    border-bottom: 3px solid #e8002d !important;
    background: transparent !important;
}

/* selectbox + inputs */
.stSelectbox > div > div, .stMultiSelect > div > div, .stTextInput > div > div {
    background-color: #161616 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    color: #f0f0f0 !important;
}
.stTextInput input {
    color: #f0f0f0 !important;
    background: #161616 !important;
}

/* buttons */
.stButton > button {
    background: #e8002d !important;
    color: white !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 0.6rem 2.5rem !important;
    transition: background 0.2s ease !important;
}
.stButton > button:hover {
    background: #ff1a45 !important;
}

/* answer box */
.answer-box {
    background: #111111;
    border-left: 4px solid #e8002d;
    border-radius: 0 6px 6px 0;
    padding: 1.5rem 1.8rem;
    margin-top: 1rem;
    font-size: 1rem;
    line-height: 1.7;
    color: #e8e8e8;
}

/* section labels */
.section-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #e8002d;
    margin-bottom: 0.5rem;
}

/* context chunks */
.context-chunk {
    background: #0f0f0f;
    border: 1px solid #1e1e1e;
    border-radius: 4px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.4rem;
    font-size: 0.85rem;
    color: #aaa;
    font-family: 'Barlow', sans-serif;
}

/* expander */
.streamlit-expanderHeader {
    background: #111 !important;
    color: #666 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.85rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

/* spinner */
.stSpinner > div { border-top-color: #e8002d !important; }

/* labels */
label, .stSelectbox label, .stTextInput label, .stMultiSelect label {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.8rem !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: #888 !important;
}
</style>

<div class="pitwall-header">
    <div class="pitwall-title">🏎 PitWall</div>
    <div class="pitwall-subtitle">F1 Strategy Intelligence System</div>
</div>
""", unsafe_allow_html=True)

CIRCUITS = [
    # Current calendar
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China",
    "Miami", "Emilia Romagna", "Monaco", "Canada", "Spain",
    "Austria", "British", "Hungarian", "Belgian", "Dutch",
    "Italian", "Azerbaijan", "Singapore", "United States", "Mexico City",
    "São Paulo", "Las Vegas", "Qatar", "Abu Dhabi",
    # Former circuits
    "Turkish", "Korean", "Indian", "European", "Malaysian",
    "Bahrain Outer", "Portuguese", "Styrian", "70th Anniversary",
    "French", "German", "Russian", "Vietnamese", "Dutch",
    "African", "Argentine", "Pacific", "Luxembourg",
]
YEARS = [2025,2024, 2023, 2022,2021,2020,2019,2018]


@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_index(circuit, years):
    index_path = f"cache/{circuit.lower()}_index"
    if Path(f"{index_path}/index.faiss").exists():
        return load_index(index_path)
    races = [(y, circuit) for y in years]
    all_data = load_multiple_races(races)
    return build_index(all_data, save_dir=index_path)


tab1, tab2 = st.tabs(["📊  Race Analyzer", "🔮  Race Forecaster"])

with tab1:
    st.markdown('<div class="section-label">Select Circuit & Years</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        circuit_a = st.selectbox("Circuit", CIRCUITS, key="circuit_a")
    with col2:
        years_a = st.multiselect("Years to load", YEARS, default=YEARS, key="years_a")

    if years_a:
        with st.spinner("Loading race data..."):
            index_a, chunks_a = get_index(circuit_a, tuple(years_a))

        st.markdown('<div class="section-label" style="margin-top:1.5rem">Your Question</div>', unsafe_allow_html=True)
        query = st.text_input("", placeholder="Why did Leclerc pit so late at Monaco 2024?", key="query_input")

        if st.button("Analyze Strategy", key="btn_analyze") and query:
            with st.spinner("Analyzing..."):
                answer = explain(query, index_a, chunks_a)
            st.markdown('<div class="section-label" style="margin-top:1.5rem">Analysis</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            with st.expander("Retrieved Context"):
                for chunk in retrieve(query, index_a, chunks_a, top_k=6):
                    st.markdown(f'<div class="context-chunk">— {chunk}</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-label">Select Circuit & Forecast Year</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        circuit_f = st.selectbox("Circuit", CIRCUITS, key="circuit_f")
    with col2:
        forecast_year = st.selectbox("Forecast Year", [2025, 2026], key="forecast_year")

    if st.button("Generate Forecast", key="btn_forecast"):
        with st.spinner("Loading historical data & generating forecast..."):
            index_f, chunks_f = get_index(circuit_f, tuple(YEARS))
            result = forecast(circuit_f, forecast_year, index_f, chunks_f)

        st.markdown('<div class="section-label" style="margin-top:1.5rem">Forecast</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{result}</div>', unsafe_allow_html=True)

        with st.expander("Historical Context Used"):
            query_f = f"winning strategy and race results at {circuit_f}"
            for chunk in retrieve(query_f, index_f, chunks_f, top_k=6):
                st.markdown(f'<div class="context-chunk">— {chunk}</div>', unsafe_allow_html=True)