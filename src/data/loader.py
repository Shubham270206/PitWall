import fastf1
import pandas as pd
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[3] / "cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


def load_race(year: int, grand_prix: str) -> dict:
    print(f"Loading {year} {grand_prix} GP...")
    session = fastf1.get_session(year, grand_prix, 'R')
    session.load(telemetry=False, laps=True, weather=False)
    print("Done.")

    laps = _clean_laps(session)
    return {
        "laps": laps,
        "pit_stops": _extract_pit_stops(laps),
        "driver_stints": _extract_stints(laps),
        "race_info": _extract_race_info(session, year, grand_prix),
    }


def _clean_laps(session) -> pd.DataFrame:
    laps = session.laps.copy()

    keep = ["Driver", "LapNumber", "LapTime", "Compound", "TyreLife",
            "FreshTyre", "PitInTime", "PitOutTime", "Position", "Stint"]
    laps = laps[[c for c in keep if c in laps.columns]].copy()

    for col in ["LapTime"]:
        if col in laps.columns:
            laps[f"{col}_s"] = laps[col].dt.total_seconds()

    laps["PittedThisLap"] = laps["PitInTime"].notna()
    return laps.reset_index(drop=True)


def _extract_pit_stops(laps: pd.DataFrame) -> pd.DataFrame:
    pits = laps[laps["PittedThisLap"]].copy()
    return pits[["Driver", "LapNumber", "Compound", "TyreLife", "Position", "Stint"]].reset_index(drop=True)


def _extract_stints(laps: pd.DataFrame) -> pd.DataFrame:
    if "Stint" not in laps.columns:
        return pd.DataFrame()
    return (
        laps.groupby(["Driver", "Stint", "Compound"])
        .agg(StartLap=("LapNumber", "min"), EndLap=("LapNumber", "max"),
             Laps=("LapNumber", "count"), AvgLapTime_s=("LapTime_s", "mean"))
        .reset_index()
        .sort_values(["Driver", "Stint"])
    )


def _extract_race_info(session, year, grand_prix) -> dict:
    return {
        "year": year,
        "grand_prix": grand_prix,
        "total_laps": int(session.laps["LapNumber"].max()),
        "drivers": session.laps["Driver"].unique().tolist(),
        "event_name": f"{year} {grand_prix} Grand Prix",
    }
def load_multiple_races(races: list[tuple[int, str]]) -> list[dict]:
    """
    races: list of (year, grand_prix) tuples
    e.g. [(2024, 'Monaco'), (2023, 'Monaco'), (2022, 'Monaco')]
    """
    results = []
    for year, gp in races:
        try:
            data = load_race(year, gp)
            results.append(data)
        except Exception as e:
            print(f"Failed to load {year} {gp}: {e}")
    return results

if __name__ == "__main__":
    races = [(2024, "Monaco"), (2023, "Monaco"), (2022, "Monaco")]
    all_data = load_multiple_races(races)
    for d in all_data:
        print(d["race_info"])
        print(d["driver_stints"][["Driver","Compound","StartLap","EndLap","Laps"]].head(3))
        print()