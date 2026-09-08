"""
Clean Air Leaderboard — gamifies air quality the way people already check
weather or cricket scores. Ranks cities, tracks "clean air streaks", and
awards badges — designed to make checking your city's air a daily habit,
not a one-off lookup.
"""

import numpy as np
import pandas as pd

BADGES = [
    {"name": "🌱 Clean Air Champion", "condition": lambda aqi, streak: aqi <= 100 and streak >= 5},
    {"name": "☀️ Breathable Today", "condition": lambda aqi, streak: aqi <= 100},
    {"name": "📈 Improving Fast", "condition": lambda aqi, streak: False},  # set externally
    {"name": "🚨 Red Alert Zone", "condition": lambda aqi, streak: aqi >= 400},
]


def build_leaderboard(df: pd.DataFrame, month: int, day: int) -> pd.DataFrame:
    """Ranks cities best-to-worst and simulates a 'clean air streak' —
    consecutive recent days AQI stayed under 100 (Satisfactory or better)."""
    rng = np.random.default_rng(month * 31 + day + 99)
    lb = df[["city", "state", "AQI", "AQI_category"]].copy()
    lb = lb.sort_values("AQI").reset_index(drop=True)
    lb["rank"] = lb.index + 1

    streaks = []
    for _, row in lb.iterrows():
        if row["AQI"] <= 100:
            streaks.append(int(rng.integers(1, 12)))
        else:
            streaks.append(0)
    lb["clean_streak_days"] = streaks

    def badge_for(row):
        if row["AQI"] <= 100 and row["clean_streak_days"] >= 5:
            return "🌱 Clean Air Champion"
        if row["AQI"] <= 100:
            return "☀️ Breathable Today"
        if row["AQI"] >= 400:
            return "🚨 Red Alert Zone"
        if row["AQI"] >= 300:
            return "😷 Heavy Smog Zone"
        return "—"

    lb["badge"] = lb.apply(badge_for, axis=1)
    return lb


def improving_cities(df: pd.DataFrame, outlook_df: pd.DataFrame, top_n=5) -> pd.DataFrame:
    """Cities with the best (most negative) forecast % change — for the
    'Most Improving' leaderboard tab."""
    improving = outlook_df.sort_values("change_%").head(top_n)
    return improving
