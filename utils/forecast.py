"""
30-Day AI Forecast Engine — Trend + Seasonal decomposition forecasting.

Implements a manual Holt-Winters-style additive decomposition (level, trend,
seasonal/monthly-climatology component) so it's fully explainable in a pitch
without needing a black-box call. In production this head would be a
Transformer/LSTM sequence-to-sequence model trained on multi-year satellite
+ CPCB history; the interface (`forecast_city`) is designed to be a drop-in
replacement target for that.
"""

import numpy as np
import pandas as pd
from .data_simulator import _season_factor, _burning_intensity


def _monthly_climatology(base_mult: float, target_month: int) -> float:
    """Expected seasonal baseline pollution level for a given month."""
    return base_mult * _season_factor(target_month)


def forecast_city(city_row: dict, current_month: int, current_day: int, horizon_days: int = 30):
    """
    Produces a day-by-day AQI-driving PM2.5 forecast for the next `horizon_days`,
    decomposed into: persistence (today's level) + seasonal drift (climatology
    shift into next month) + burning-season adjustment + uncertainty band that
    widens with lead time (classic forecast-skill decay).

    Returns a DataFrame: date, forecast_PM2.5, lower_bound, upper_bound, driver_note
    """
    rng = np.random.default_rng(hash(city_row["city"]) % (2**31) + current_day)

    start_date = pd.Timestamp(2025, current_month, current_day) + pd.Timedelta(days=1)
    dates = pd.date_range(start=start_date, periods=horizon_days)

    today_level = city_row["PM2.5"]
    base_mult = city_row.get("_base_mult", today_level / max(_season_factor(current_month), 0.1) / 55)

    records = []
    for i, d in enumerate(dates):
        t = i + 1
        # 1. Seasonal climatology target for that future date's month
        seasonal_target = base_mult * 55 * _season_factor(d.month)

        # 2. Persistence decays toward seasonal climatology over ~14 days (mean reversion)
        decay = np.exp(-t / 14)
        level = today_level * decay + seasonal_target * (1 - decay)

        # 3. Burning-season kicker for belt cities transitioning into Oct 15-Nov 15
        burn_adj = 0
        if city_row.get("in_burning_belt", False):
            burn_adj = (_burning_intensity(d.month) - _burning_intensity(current_month)) * 12

        forecast_val = np.clip(level + burn_adj, 5, 500)

        # 4. Uncertainty widens with lead time (sqrt-time growth, standard forecast-skill decay)
        uncertainty = forecast_val * 0.06 * np.sqrt(t)
        lower = max(5, forecast_val - uncertainty)
        upper = min(500, forecast_val + uncertainty)

        driver = "Seasonal transition"
        if burn_adj > 8:
            driver = "Stubble-burning season onset"
        elif d.month in (11, 12, 1) and current_month not in (11, 12, 1):
            driver = "Winter inversion onset"
        elif d.month in (7, 8, 9):
            driver = "Monsoon washout expected"

        records.append({
            "date": d, "forecast_PM2.5": round(forecast_val, 1),
            "lower_bound": round(lower, 1), "upper_bound": round(upper, 1),
            "driver": driver,
        })

    return pd.DataFrame(records)


def forecast_summary(forecast_df: pd.DataFrame, current_pm25: float) -> dict:
    """Generates a plain-language advisory summarizing the 30-day trajectory."""
    end_val = forecast_df["forecast_PM2.5"].iloc[-1]
    peak_val = forecast_df["forecast_PM2.5"].max()
    peak_date = forecast_df.loc[forecast_df["forecast_PM2.5"].idxmax(), "date"]
    delta_pct = (end_val - current_pm25) / max(current_pm25, 1) * 100
    dominant_driver = forecast_df["driver"].mode().iloc[0]

    if delta_pct > 15:
        trend_label, trend_icon = "Worsening", "📈"
    elif delta_pct < -15:
        trend_label, trend_icon = "Improving", "📉"
    else:
        trend_label, trend_icon = "Stable", "➡️"

    return {
        "trend_label": trend_label, "trend_icon": trend_icon,
        "delta_pct": round(delta_pct, 1),
        "peak_value": round(peak_val, 1), "peak_date": peak_date,
        "dominant_driver": dominant_driver,
        "end_value": round(end_val, 1),
    }
