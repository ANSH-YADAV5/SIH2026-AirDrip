"""
CPCB National AQI Calculation Engine
Implements the official Indian AQI sub-index breakpoint methodology
(CPCB, 2014 - National Air Quality Index) for 8 pollutants.

Final AQI = max(sub-indices) — the "worst pollutant" governs the reported AQI,
exactly as CPCB does it for public reporting.
"""

import numpy as np

# CPCB breakpoints: {pollutant: [(C_low, C_high, I_low, I_high), ...]}
# Units: PM2.5/PM10/NO2/SO2/O3 in ug/m3 (24-hr avg), CO in mg/m3 (8-hr), NH3 in ug/m3
BREAKPOINTS = {
    "PM2.5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
              (91, 120, 201, 300), (121, 250, 301, 400), (251, 380, 401, 500)],
    "PM10":  [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
              (251, 350, 201, 300), (351, 430, 301, 400), (431, 510, 401, 500)],
    "NO2":   [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
              (181, 280, 201, 300), (281, 400, 301, 400), (401, 500, 401, 500)],
    "SO2":   [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
              (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2100, 401, 500)],
    "CO":    [(0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10, 101, 200),
              (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 43, 401, 500)],
    "O3":    [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
              (169, 208, 201, 300), (209, 748, 301, 400), (749, 940, 401, 500)],
    "HCHO":  [(0, 5, 0, 50), (5.1, 10, 51, 100), (10.1, 20, 101, 200),
              (20.1, 35, 201, 300), (35.1, 55, 301, 400), (55.1, 80, 401, 500)],
    # HCHO has no official CPCB AQI category (not a criteria pollutant) — this
    # scale is a research proxy loosely modeled on WHO/EPA indoor-air guidance
    # bands, used here ONLY for hotspot severity tiers, never blended into AQI.
}

CATEGORIES = [
    (0, 50, "Good", "#009865"),
    (51, 100, "Satisfactory", "#a3c853"),
    (101, 200, "Moderate", "#ffb302"),
    (201, 300, "Poor", "#ff6b02"),
    (301, 400, "Very Poor", "#e0301e"),
    (401, 500, "Severe", "#7e0023"),
]


def sub_index(pollutant: str, conc: float) -> float:
    """Piecewise-linear interpolation per CPCB formula:
    I = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low
    """
    if conc is None or np.isnan(conc) or conc < 0:
        return np.nan
    bps = BREAKPOINTS[pollutant]
    for c_lo, c_hi, i_lo, i_hi in bps:
        if c_lo <= conc <= c_hi:
            return round(((i_hi - i_lo) / (c_hi - c_lo)) * (conc - c_lo) + i_lo, 1)
    # above top breakpoint -> extrapolate off the last band, capped at 500
    c_lo, c_hi, i_lo, i_hi = bps[-1]
    val = ((i_hi - i_lo) / (c_hi - c_lo)) * (conc - c_lo) + i_lo
    return float(min(val, 500))


def compute_aqi(pollutant_concs: dict, already_index: bool = False) -> dict:
    """
    pollutant_concs: {"PM2.5": val, "PM10": val, "NO2": val, "SO2": val, "CO": val, "O3": val}

    already_index: set True when the input values are ALREADY per-pollutant
    AQI sub-indices (0-500 scale) rather than raw concentrations — this is
    the case for data pulled from WAQI, whose `iaqi` values are that
    pollutant's own computed AQI sub-index, not a µg/m3 concentration.
    Feeding those through the CPCB concentration->sub-index breakpoint
    table a second time silently inflates the result (e.g. a real
    "Moderate" reading can come out capped at 500/"Severe"). When True,
    this skips the breakpoint conversion and uses the values directly
    (clipped to [0, 500]) as the sub-indices.

    Returns dict with overall AQI, category, dominant pollutant, and all sub-indices.
    (HCHO deliberately excluded from AQI blending — see note above.)
    """
    core = {k: v for k, v in pollutant_concs.items() if k != "HCHO"}
    if already_index:
        def _clip(v):
            if v is None or (isinstance(v, float) and np.isnan(v)) or v < 0:
                return np.nan
            return float(min(v, 500))
        sub_indices = {p: _clip(c) for p, c in core.items()}
    else:
        sub_indices = {p: sub_index(p, c) for p, c in core.items()}
    valid = {p: v for p, v in sub_indices.items() if not np.isnan(v)}
    if not valid:
        return {"AQI": np.nan, "category": "No Data", "dominant": None, "sub_indices": sub_indices}
    dominant = max(valid, key=valid.get)
    aqi_val = valid[dominant]
    category, color = get_category(aqi_val)
    return {
        "AQI": round(aqi_val),
        "category": category,
        "color": color,
        "dominant": dominant,
        "sub_indices": sub_indices,
    }


def get_category(aqi_val: float):
    for lo, hi, name, color in CATEGORIES:
        if lo <= aqi_val <= hi:
            return name, color
    return "Severe", "#7e0023"


def hcho_severity_tier(hcho_conc: float) -> str:
    """Qualitative HCHO column-density severity tier for hotspot maps
    (research proxy scale, not an official regulatory index)."""
    idx = sub_index("HCHO", hcho_conc)
    if np.isnan(idx):
        return "Unknown"
    name, _ = get_category(idx)
    return name
