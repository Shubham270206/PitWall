import pandas as pd
import faiss
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def build_index(all_data: list[dict], save_dir: str) -> tuple[faiss.Index, list[str]]:
    chunks = []
    for data in all_data:
        chunks.extend(_make_chunks(data))

    print(f"Total chunks: {len(chunks)}")

    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    _save(index, chunks, save_dir)
    return index, chunks


def _make_chunks(data: dict) -> list[str]:
    chunks = []
    info = data["race_info"]
    label = info["event_name"]
    circuit = info["grand_prix"]
    year = info["year"]

    for _, row in data["driver_stints"].iterrows():
        avg = f"{row['AvgLapTime_s']:.2f}s" if pd.notna(row["AvgLapTime_s"]) else "N/A"
        chunks.append(
            f"{label}: {row['Driver']} ran stint {int(row['Stint'])} on {row['Compound']} tyres "
            f"from lap {int(row['StartLap'])} to lap {int(row['EndLap'])} "
            f"({int(row['Laps'])} laps, avg lap time {avg})."
        )

    for _, row in data["pit_stops"].iterrows():
        pos = int(row['Position']) if pd.notna(row['Position']) else "N/A"
        tyre_life = int(row['TyreLife']) if pd.notna(row['TyreLife']) else "N/A"
        chunks.append(
            f"{label}: {row['Driver']} pitted on lap {int(row['LapNumber'])} "
            f"from position {pos}, tyre life {tyre_life} laps, compound {row['Compound']}."
        )

    for driver, group in data["laps"].groupby("Driver"):
        compounds = group["Compound"].dropna().unique().tolist()
        stops = int(group["PittedThisLap"].sum())
        chunks.append(
            f"{label}: {driver} used {len(compounds)} compound(s) — {', '.join(compounds)} — "
            f"and made {stops} pit stop(s)."
        )

    try:
        final_lap = data["laps"][data["laps"]["LapNumber"] == info["total_laps"]]
        winner_row = final_lap[final_lap["Position"] == 1]
        if not winner_row.empty:
            winner = winner_row.iloc[0]["Driver"]
            winner_stints = data["driver_stints"][data["driver_stints"]["Driver"] == winner]
            compounds_used = winner_stints["Compound"].tolist()
            chunks.append(
                f"{label}: Race winner was {winner}. "
                f"Winning strategy: {len(compounds_used)} stints — {', '.join(compounds_used)}."
            )
    except Exception:
        pass

    chunks.append(
        f"Historical pattern at {circuit} ({year}): {info['total_laps']} laps total. "
        f"Drivers: {', '.join(info['drivers'])}."
    )

    return chunks


def _save(index, chunks, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(save_dir / "index.faiss"))
    with open(save_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved to {save_dir}")


def load_index(save_dir: str) -> tuple[faiss.Index, list[str]]:
    save_dir = Path(save_dir)
    index = faiss.read_index(str(save_dir / "index.faiss"))
    with open(save_dir / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


if __name__ == "__main__":
    from src.data.loader import load_multiple_races
    races = [(2024, "Monaco"), (2023, "Monaco"), (2022, "Monaco")]
    all_data = load_multiple_races(races)
    index, chunks = build_index(all_data, save_dir="cache/monaco_index")

    print("\nSample chunks:")
    for c in chunks[:6]:
        print(" -", c)