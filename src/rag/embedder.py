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

    # circuit characteristics context
    CIRCUIT_NOTES = {
        "Monaco": "Monaco is a street circuit with very low tyre degradation, making one-stop strategies dominant. Overtaking is nearly impossible so track position is everything — undercuts are the primary strategic weapon.",
        "Silverstone": "Silverstone has high-speed corners causing high tyre degradation, especially on the fronts. Two-stop strategies are common. The undercut is powerful here.",
        "Monza": "Monza is a low-downforce, high-speed circuit with low tyre degradation. One-stop strategies dominate. Slipstreaming makes overtaking easier so track position is less critical.",
        "Spa": "Spa is a long circuit with mixed conditions. Rain is common and can completely change strategy. Tyre degradation is moderate. Safety cars are frequent.",
        "Bahrain": "Bahrain has high tyre degradation due to abrasive tarmac and heat. Two-stop strategies are common. The undercut works well here.",
        "Japan": "Suzuka has medium-high tyre degradation. Two-stop strategies are typical. MEDIUM and HARD compounds usually dominate. Track position matters as overtaking is difficult.",
        "Interlagos": "Interlagos has medium degradation and frequent safety cars which trigger strategy swings. One or two stops depending on conditions.",
        "British": "Silverstone GP — see Silverstone notes.",
        "Hungarian": "Hungary is a tight, twisty circuit similar to Monaco in terms of overtaking difficulty. One-stop strategies are common but the soft compound degrades quickly.",
        "Dutch": "Zandvoort has banking in corners causing unique tyre wear patterns. One or two stops. Track position critical as overtaking is very difficult.",
        "Italian": "Monza — see Monza notes.",
        "Austrian": "Red Bull Ring is a short circuit with moderate degradation. Aggressive strategies and multiple stops are viable.",
        "Belgian": "Spa — see Spa notes.",
        "Singapore": "Singapore is a street circuit like Monaco. Low degradation, one-stop dominant, track position critical.",
        "Azerbaijan": "Baku is a street circuit with a long straight. Safety cars are very common and can trigger opportunistic strategy swings.",
    }

    note = CIRCUIT_NOTES.get(circuit, f"{circuit} is an F1 circuit with its own unique tyre degradation and strategy characteristics.")
    chunks.append(f"Circuit insight for {circuit}: {note}")

    # stint chunks with strategic context
    for _, row in data["driver_stints"].iterrows():
        avg = f"{row['AvgLapTime_s']:.2f}s" if pd.notna(row["AvgLapTime_s"]) else "N/A"
        laps = int(row['Laps'])
        compound = row['Compound']

        # add strategic reasoning based on stint length and compound
        if compound == "SOFT" and laps > 20:
            strategy_note = "Running SOFT for this long suggests either a safety car extended the stint or the driver managed tyres exceptionally well."
        elif compound == "HARD" and laps > 35:
            strategy_note = "A long HARD stint suggests a one-stop strategy targeting tyre longevity over outright pace."
        elif compound == "MEDIUM" and laps < 10:
            strategy_note = "A short MEDIUM stint suggests an early pit stop, possibly an undercut attempt or reaction to a safety car."
        elif compound in ["WET", "INTERMEDIATE"]:
            strategy_note = "Wet weather compounds indicate the race had rainfall, which resets strategy entirely."
        else:
            strategy_note = f"A {laps}-lap stint on {compound} is typical strategic usage of this compound."

        chunks.append(
            f"{label}: {row['Driver']} ran stint {int(row['Stint'])} on {compound} tyres "
            f"from lap {int(row['StartLap'])} to lap {int(row['EndLap'])} "
            f"({laps} laps, avg lap time {avg}). {strategy_note}"
        )

    # pit stop chunks with undercut/overcut context
    for _, row in data["pit_stops"].iterrows():
        pos = int(row['Position']) if pd.notna(row['Position']) else "N/A"
        tyre_life = int(row['TyreLife']) if pd.notna(row['TyreLife']) else "N/A"
        lap = int(row['LapNumber'])

        if tyre_life != "N/A" and tyre_life < 15:
            pit_note = "Early pit stop — likely an undercut attempt to gain track position or reaction to a safety car."
        elif tyre_life != "N/A" and tyre_life > 35:
            pit_note = "Very late pit stop — one-stop strategy, prioritising track position over tyre freshness."
        else:
            pit_note = "Standard pit stop window."

        chunks.append(
            f"{label}: {row['Driver']} pitted on lap {lap} from position {pos}, "
            f"tyre life {tyre_life} laps, compound {row['Compound']}. {pit_note}"
        )

    # per-driver strategy summary
    for driver, group in data["laps"].groupby("Driver"):
        compounds = group["Compound"].dropna().unique().tolist()
        stops = int(group["PittedThisLap"].sum())
        stop_desc = "one-stop" if stops == 1 else f"{stops}-stop"
        chunks.append(
            f"{label}: {driver} ran a {stop_desc} strategy using {', '.join(compounds)} compounds."
        )

    # winner chunk
    try:
        final_lap = data["laps"][data["laps"]["LapNumber"] == info["total_laps"]]
        winner_row = final_lap[final_lap["Position"] == 1]
        if not winner_row.empty:
            winner = winner_row.iloc[0]["Driver"]
            winner_stints = data["driver_stints"][data["driver_stints"]["Driver"] == winner]
            compounds_used = winner_stints["Compound"].tolist()
            stop_count = len(compounds_used) - 1
            chunks.append(
                f"{label}: Race winner was {winner} with a {stop_count}-stop strategy "
                f"using {', '.join(compounds_used)} compounds in order. "
                f"This winning strategy reflects what worked best at {circuit} that year."
            )
    except Exception:
        pass

    # circuit historical pattern
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