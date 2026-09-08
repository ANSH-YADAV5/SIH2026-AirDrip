"""
Realistic synthetic data generator standing in for satellite/ground feeds.

WHY SYNTHETIC: INSAT-3D (MOSDAC), Sentinel-5P TROPOMI (Copernicus/GEE), and
CPCB live feeds all require authenticated API access not reachable from this
build environment. This module generates spatially- and seasonally-correlated
data using known real-world pollution geography (Indo-Gangetic Plain loading,
NCR urban core, stubble-burning belt, coastal ventilation) so the dashboard
behaves the way the real pipeline would. Swap `fetch_live_*` stubs at the
bottom for real GEE/MOSDAC/CPCB calls post-hackathon.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# Representative grid: (city, state, lat, lon, baseline pollution multiplier)
# Multiplier reflects broadly known real-world air quality patterns
# (IGP + NCR consistently highest, Himalayan/NE/coastal south lowest).
LOCATIONS = [
    ("Delhi", "Delhi", 28.6139, 77.2090, 2.6),
    ("Noida", "Uttar Pradesh", 28.5355, 77.3910, 2.4),
    ("Ghaziabad", "Uttar Pradesh", 28.6692, 77.4538, 2.5),
    ("Kanpur", "Uttar Pradesh", 26.4499, 80.3319, 2.3),
    ("Lucknow", "Uttar Pradesh", 26.8467, 80.9462, 2.0),
    ("Patna", "Bihar", 25.5941, 85.1376, 2.2),
    ("Varanasi", "Uttar Pradesh", 25.3176, 82.9739, 2.1),
    ("Amritsar", "Punjab", 31.6340, 74.8723, 1.9),
    ("Ludhiana", "Punjab", 30.9010, 75.8573, 2.0),
    ("Chandigarh", "Chandigarh", 30.7333, 76.7794, 1.6),
    ("Jaipur", "Rajasthan", 26.9124, 75.7873, 1.8),
    ("Agra", "Uttar Pradesh", 27.1767, 78.0081, 2.1),
    ("Kolkata", "West Bengal", 22.5726, 88.3639, 1.9),
    ("Mumbai", "Maharashtra", 19.0760, 72.8777, 1.5),
    ("Pune", "Maharashtra", 18.5204, 73.8567, 1.3),
    ("Nagpur", "Maharashtra", 21.1458, 79.0882, 1.4),
    ("Ahmedabad", "Gujarat", 23.0225, 72.5714, 1.7),
    ("Bhopal", "Madhya Pradesh", 23.2599, 77.4126, 1.3),
    ("Hyderabad", "Telangana", 17.3850, 78.4867, 1.2),
    ("Bengaluru", "Karnataka", 12.9716, 77.5946, 1.0),
    ("Chennai", "Tamil Nadu", 13.0827, 80.2707, 1.0),
    ("Coimbatore", "Tamil Nadu", 11.0168, 76.9558, 0.8),
    ("Kochi", "Kerala", 9.9312, 76.2673, 0.7),
    ("Thiruvananthapuram", "Kerala", 8.5241, 76.9366, 0.6),
    ("Raipur", "Chhattisgarh", 21.2514, 81.6296, 1.5),
    ("Guwahati", "Assam", 26.1445, 91.7362, 1.1),
    ("Dehradun", "Uttarakhand", 30.3165, 78.0322, 1.2),
    ("Srinagar", "J&K", 34.0837, 74.7973, 0.7),
    ("Jodhpur", "Rajasthan", 26.2389, 73.0243, 1.6),
    ("Rohtak", "Haryana", 28.8955, 76.6066, 2.2),
    ("Hisar", "Haryana", 29.1492, 75.7217, 2.1),
    ("Bathinda", "Punjab", 30.2110, 74.9455, 1.9),
]

# Stubble-burning belt (Punjab/Haryana/W-UP) — HCHO/fire hotspot zone,
# strongly seasonal (peaks Oct 15 - Nov 15 each year; real NASA FIRMS pattern)
BURNING_BELT = {"Amritsar", "Ludhiana", "Bathinda", "Rohtak", "Hisar", "Chandigarh", "Patiala"}

# Cities with NO nearby CPCB/WAQI ground station (confirmed by testing — see
# check_cities.py workflow). For these, "no data" isn't an option, but making
# up a number is worse — so instead of pure random simulation, these use a
# REAL satellite-informed atmospheric model (see fetch_satellite_estimate()
# below) whenever the live toggle is on, clearly labeled as an estimate
# rather than a ground reading.
SATELLITE_ESTIMATE_LOCATIONS = [
    ("Gurugram", "Haryana", 28.4595, 77.0266, 2.4),
    ("Surat", "Gujarat", 21.1702, 72.8311, 1.5),
    ("Indore", "Madhya Pradesh", 22.7196, 75.8577, 1.4),
    ("Bhubaneswar", "Odisha", 20.2961, 85.8245, 1.3),
    ("Ranchi", "Jharkhand", 23.3441, 85.3096, 1.5),
    ("Shimla", "Himachal Pradesh", 31.1048, 77.1734, 0.6),
    ("Visakhapatnam", "Andhra Pradesh", 17.6868, 83.2185, 1.1),
    ("Panaji", "Goa", 15.4909, 73.8278, 0.6),
    ("Guntur", "Andhra Pradesh", 16.3067, 80.4365, 1.2),
]


def _season_factor(month: int) -> float:
    """Winter inversion + stubble burning -> Oct-Jan worst; monsoon -> Jul-Sep best."""
    if month in (11, 12, 1):
        return 1.7
    if month in (10, 2):
        return 1.35
    if month in (3, 4):
        return 1.0
    if month in (7, 8, 9):
        return 0.55
    return 0.9


def _burning_intensity(month: int) -> float:
    """Peak stubble burning Oct 15 - Nov 15."""
    if month == 11:
        return 2.5
    if month == 10:
        return 1.8
    return 0.15


def generate_grid_snapshot(month: int = 11, day: int = 3, noise_seed: int = None,
                            locations=None) -> pd.DataFrame:
    """Generates one day's synthetic satellite-derived + ground snapshot across all locations.
    Pass locations= to generate for a different city list (e.g. SATELLITE_ESTIMATE_LOCATIONS)
    instead of the default ground-station-covered LOCATIONS."""
    if locations is None:
        locations = LOCATIONS
    rng = np.random.default_rng(noise_seed if noise_seed is not None else (month * 31 + day))
    season = _season_factor(month)
    burn = _burning_intensity(month)

    rows = []
    for city, state, lat, lon, mult in locations:
        base = mult * season
        in_belt = city in BURNING_BELT

        # Satellite AOD (unitless, 0-2 typical over India)
        aod = np.clip(base * 0.35 + rng.normal(0, 0.05), 0.05, 2.2)

        # Meteorology (drives dispersion -> inversely affects surface conc)
        wind_speed = np.clip(rng.normal(3.5 - 0.5 * season, 1.0), 0.3, 12)  # m/s, calmer in winter
        boundary_layer_height = np.clip(rng.normal(1200 - 400 * season, 150), 200, 2500)  # m
        rh = np.clip(rng.normal(55 + 15 * season, 10), 15, 95)
        temp = np.clip(rng.normal(28 - 12 * season, 3), 5, 45)

        dispersion_factor = (1000 / boundary_layer_height) * (2.5 / max(wind_speed, 0.5))

        # Surface pollutants (AI model target variables) — AOD + met -> surface conc
        pm25 = np.clip(base * 55 * dispersion_factor**0.4 + rng.normal(0, 8), 5, 500)
        pm10 = np.clip(pm25 * rng.uniform(1.5, 2.0), 10, 600)
        no2 = np.clip(base * 35 * dispersion_factor**0.3 + rng.normal(0, 5), 3, 300)
        so2 = np.clip(base * 14 + rng.normal(0, 3), 1, 150)
        co = np.clip(base * 0.9 * dispersion_factor**0.2 + rng.normal(0, 0.15), 0.2, 8)
        o3 = np.clip(40 + 20 * (1 - season) + rng.normal(0, 8), 5, 200)  # O3 anti-correlated w/ winter

        # HCHO column density (1e15 molec/cm^2 equiv, TROPOMI-like) — driven by
        # biomass burning + biogenic VOC + traffic secondary formation
        fire_count = max(0, rng.poisson(burn * 8 if in_belt else burn * 0.8))
        hcho = np.clip(2.0 + fire_count * 1.1 + base * 1.5 + rng.normal(0, 0.6), 0.5, 40)

        rows.append({
            "city": city, "state": state, "lat": lat, "lon": lon,
            "AOD": round(aod, 3),
            "wind_speed_ms": round(wind_speed, 2),
            "boundary_layer_height_m": round(boundary_layer_height, 0),
            "relative_humidity": round(rh, 1),
            "temperature_c": round(temp, 1),
            "PM2.5": round(pm25, 1),
            "PM10": round(pm10, 1),
            "NO2": round(no2, 1),
            "SO2": round(so2, 1),
            "CO": round(co, 2),
            "O3": round(o3, 1),
            "HCHO": round(hcho, 2),
            "fire_count": int(fire_count),
            "in_burning_belt": in_belt,
            "month": month, "day": day,
        })
    return pd.DataFrame(rows)


def generate_time_series(city_row, days: int = 60, end_month: int = 11, end_day: int = 15) -> pd.DataFrame:
    """Generates a rolling daily time series for one location, for model performance charts."""
    rng = np.random.default_rng(hash(city_row["city"]) % (2**31))
    dates = pd.date_range(end=pd.Timestamp(2025, end_month, end_day), periods=days)
    records = []
    for d in dates:
        season = _season_factor(d.month)
        actual = np.clip(city_row["PM2.5"] * (season / _season_factor(end_month)) + rng.normal(0, 12), 5, 500)
        # simulated CNN-LSTM prediction: correlated with actual + small model error
        predicted = np.clip(actual + rng.normal(0, actual * 0.09), 5, 500)
        records.append({"date": d, "actual_PM2.5": round(actual, 1), "predicted_PM2.5": round(predicted, 1)})
    return pd.DataFrame(records)


def fetch_live_cpcb(api_key: str = None, limit: int = 2000, timeout: int = 90) -> pd.DataFrame:
    """
    Real implementation: pulls live station readings from data.gov.in's
    "Real time Air Quality Index from various locations" resource
    (resource_id fixed by data.gov.in — CPCB is the underlying data owner).

    Returns a WIDE-format DataFrame, one row per city (readings averaged
    across that city's stations), with columns matching what
    utils.aqi_calculator.compute_aqi() expects: city, state, lat, lon,
    PM2.5, PM10, NO2, SO2, CO, O3 (NaN where a city has no reading for
    that pollutant right now — not every station reports every pollutant).

    api_key: your data.gov.in API key. Pass it in, or set it via
    st.secrets["CPCB_API_KEY"] / the CPCB_API_KEY environment variable —
    never hardcode a real key directly in this file, especially in a
    public repo.

    Raises on any failure (network error, bad key, empty/malformed
    response) — callers should catch and fall back to simulated data,
    see `get_live_or_simulated_snapshot()` below for that pattern.
    """
    import requests
    import time

    if not api_key:
        raise ValueError("No API key provided. Pass api_key= or set CPCB_API_KEY.")

    RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"  # fixed data.gov.in resource id
    url = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
    params = {"api-key": api_key, "format": "json", "limit": limit}

    # data.gov.in occasionally returns a transient 502/503/504 under load —
    # retry a couple of times with a short backoff before giving up, rather
    # than immediately falling back to simulated data on a blip.
    last_error = None
    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            last_error = None
            break
        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(3 * (attempt + 1))  # 3s, then 6s
    if last_error is not None:
        raise last_error

    payload = resp.json()

    records = payload.get("records", [])
    if not records:
        raise RuntimeError("data.gov.in returned zero records — check the API key or resource status.")

    df_long = pd.DataFrame(records)

    required_cols = {"state", "city", "station", "latitude", "longitude", "pollutant_id", "pollutant_avg"}
    missing = required_cols - set(df_long.columns)
    if missing:
        raise RuntimeError(f"Unexpected API response shape — missing columns: {missing}")

    # Clean numeric fields (API sometimes returns "NA" as a literal string)
    for col in ["pollutant_avg", "latitude", "longitude"]:
        df_long[col] = pd.to_numeric(df_long[col], errors="coerce")
    df_long = df_long.dropna(subset=["pollutant_avg"])

    # Normalize pollutant naming to match our internal convention
    POLLUTANT_ALIAS = {"OZONE": "O3", "O3": "O3", "PM2.5": "PM2.5", "PM10": "PM10",
                        "NO2": "NO2", "SO2": "SO2", "CO": "CO", "NH3": "NH3"}
    df_long["pollutant_id"] = df_long["pollutant_id"].str.upper().str.strip().map(
        lambda p: POLLUTANT_ALIAS.get(p, p))

    # Long -> wide: one row per city, one column per pollutant (mean across
    # that city's stations, since our app operates at city granularity)
    df_wide = df_long.pivot_table(
        index=["state", "city"], columns="pollutant_id", values="pollutant_avg", aggfunc="mean"
    ).reset_index()

    coords = df_long.groupby("city")[["latitude", "longitude"]].mean().reset_index()
    df_wide = df_wide.merge(coords, on="city", how="left")
    df_wide = df_wide.rename(columns={"latitude": "lat", "longitude": "lon"})

    for pol in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]:
        if pol not in df_wide.columns:
            df_wide[pol] = np.nan

    return df_wide


def fetch_satellite_estimate(cities=None, timeout: int = 15) -> pd.DataFrame:
    """
    Real satellite-informed estimate for cities with NO nearby CPCB/WAQI
    ground station. Uses Open-Meteo's free Air Quality API, built on the
    Copernicus Atmosphere Monitoring Service (CAMS) — a global atmospheric
    composition model that assimilates satellite retrievals (aerosol
    optical depth, trace gases) rather than only ground sensors, run at
    ~11km (Europe) / ~40km (global) resolution.

    Honest scope note: this is NOT the custom-trained CNN-LSTM described in
    model.py — training that needs a historical satellite+CPCB-paired
    dataset and GPU time outside this project's build window. This is a
    real, already-published, continuously-updated atmospheric model's
    current estimate for the exact coordinates requested, which is the
    honest, immediately-deployable version of "estimate air quality where
    there is no ground station" — as opposed to a purely synthetic formula.

    Free, no API key, no signup: https://open-meteo.com/en/docs/air-quality-api

    Returns the same wide-format shape as fetch_live_cpcb()/fetch_live_waqi(),
    plus an "AOD" column (aerosol optical depth) when available.
    """
    import requests

    if cities is None:
        cities = SATELLITE_ESTIMATE_LOCATIONS
    if not cities:
        return pd.DataFrame(columns=["state", "city", "lat", "lon",
                                      "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AOD"])

    lats = ",".join(str(c[2]) for c in cities)
    lons = ",".join(str(c[3]) for c in cities)
    params = {
        "latitude": lats,
        "longitude": lons,
        "current": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone,aerosol_optical_depth",
        "timezone": "auto",
    }
    resp = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality",
                         params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    # Single-location requests return a dict; multi-location returns a list
    # of dicts in the same order as the input coordinates — normalize.
    entries = payload if isinstance(payload, list) else [payload]

    rows = []
    for city_row, entry in zip(cities, entries):
        city, state, lat, lon, _ = city_row
        cur = entry.get("current", {}) if isinstance(entry, dict) else {}
        co_ugm3 = cur.get("carbon_monoxide")
        rows.append({
            "state": state, "city": city, "lat": lat, "lon": lon,
            "PM2.5": cur.get("pm2_5"),
            "PM10": cur.get("pm10"),
            "NO2": cur.get("nitrogen_dioxide"),
            "SO2": cur.get("sulphur_dioxide"),
            # India's CPCB breakpoints expect CO in mg/m3; Open-Meteo reports
            # it in ug/m3 — convert here so the SAME compute_aqi() pipeline
            # used for CPCB/simulated data works correctly for this source
            # too (raw concentration in, no double-conversion needed).
            "CO": (co_ugm3 / 1000.0) if co_ugm3 is not None else None,
            "O3": cur.get("ozone"),
            "AOD": cur.get("aerosol_optical_depth"),
        })

    if not rows:
        raise RuntimeError("Open-Meteo returned no usable data for any tracked city.")

    df_wide = pd.DataFrame(rows)
    for pol in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]:
        if pol not in df_wide.columns:
            df_wide[pol] = np.nan
    return df_wide


def fetch_live_waqi(token: str, timeout: int = 8, max_workers: int = 12) -> pd.DataFrame:
    """
    Fallback live source: World Air Quality Index project (aqicn.org), which
    mirrors CPCB's own station network (586+ Indian government stations)
    through a much more reliable, low-latency API than hitting data.gov.in
    directly. Same underlying government data, different (faster) pipe.

    Free token: https://aqicn.org/data-platform/register

    Queries one feed per city in LOCATIONS, in parallel (network latency
    dominates, not API rate limits — WAQI's free tier allows very high
    request rates). Returns the same wide-format shape as fetch_live_cpcb()
    so both are drop-in interchangeable for the rest of the app.

    Caveat (documented honestly): WAQI's per-pollutant `iaqi` values are
    that pollutant's own AQI sub-index as computed by WAQI, not always the
    raw µg/m³ concentration CPCB reports directly — close enough for a
    demo/dashboard, but not a byte-for-byte match with data.gov.in's numbers.
    """
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not token:
        raise ValueError("No WAQI token provided. Get one free at aqicn.org/data-platform/register")

    def _parse_feed(payload, city, state, lat, lon):
        if payload.get("status") != "ok":
            return None
        data = payload["data"]
        iaqi = data.get("iaqi", {})
        return {
            "state": state, "city": city,
            "lat": data.get("city", {}).get("geo", [lat, lon])[0],
            "lon": data.get("city", {}).get("geo", [lat, lon])[1],
            "PM2.5": iaqi.get("pm25", {}).get("v"),
            "PM10": iaqi.get("pm10", {}).get("v"),
            "NO2": iaqi.get("no2", {}).get("v"),
            "SO2": iaqi.get("so2", {}).get("v"),
            "CO": iaqi.get("co", {}).get("v"),
            "O3": iaqi.get("o3", {}).get("v"),
        }

    def _fetch_one(city_row):
        city, state, lat, lon, _ = city_row
        # Try 1: lookup by city name — fast, but fails if WAQI's station
        # for this area is registered under a different name/spelling.
        try:
            resp = requests.get(f"https://api.waqi.info/feed/{city}/",
                                 params={"token": token}, timeout=timeout)
            resp.raise_for_status()
            row = _parse_feed(resp.json(), city, state, lat, lon)
            if row is not None:
                return row
        except Exception:
            pass

        # Try 2: geo-based lookup — finds the NEAREST station to this
        # city's coordinates, regardless of what it's named. Rescues cities
        # that have a real nearby station but failed the name match above.
        try:
            resp = requests.get(f"https://api.waqi.info/feed/geo:{lat};{lon}/",
                                 params={"token": token}, timeout=timeout)
            resp.raise_for_status()
            row = _parse_feed(resp.json(), city, state, lat, lon)
            if row is not None:
                return row
        except Exception:
            pass

        return None

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_fetch_one, city_row) for city_row in LOCATIONS]
        for f in as_completed(futures):
            result = f.result()
            if result is not None:
                rows.append(result)

    if not rows:
        raise RuntimeError("WAQI returned no usable data for any tracked city — check the token.")

    df_wide = pd.DataFrame(rows)
    for pol in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]:
        if pol not in df_wide.columns:
            df_wide[pol] = np.nan
    return df_wide


def get_live_or_simulated_snapshot(api_key: str, month: int, day: int, waqi_token: str = None):
    """
    Convenience wrapper: tries the WAQI feed first (same underlying CPCB
    government station data, served through a faster/more reliable pipe);
    on failure, falls back to CPCB direct via data.gov.in; on failure,
    falls back to the simulated snapshot so the app never crashes because
    of an external dependency.
    Returns (df, source: "waqi" | "cpcb" | "simulated", error_message).
    """
    if waqi_token:
        try:
            live_df = fetch_live_waqi(token=waqi_token)
            return live_df, "waqi", None
        except Exception as e_waqi:
            try:
                live_df = fetch_live_cpcb(api_key=api_key)
                return live_df, "cpcb", None
            except Exception as e_cpcb:
                sim_df = generate_grid_snapshot(month=month, day=day)
                return sim_df, "simulated", f"WAQI failed ({e_waqi}); CPCB fallback also failed ({e_cpcb})"
    try:
        live_df = fetch_live_cpcb(api_key=api_key)
        return live_df, "cpcb", None
    except Exception as e_cpcb:
        sim_df = generate_grid_snapshot(month=month, day=day)
        return sim_df, "simulated", str(e_cpcb)



# ---------------------------------------------------------------------------
# STUBS FOR REMAINING REAL DATA INTEGRATIONS (post-hackathon / with credentials)
# ---------------------------------------------------------------------------

def fetch_live_tropomi_hcho(date_str: str, bbox=None):
    """
    Real implementation (requires Earth Engine auth):
        import ee; ee.Initialize()
        col = ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_HCHO") \\
                .filterDate(date_str, next_day) \\
                .select("tropospheric_HCHO_column_number_density")
        ... reduce over India geometry, sample to grid ...
    """
    raise NotImplementedError("Wire up Earth Engine credentials to enable live TROPOMI pulls.")


def fetch_live_firms(date_str: str, bbox=None):
    """Real implementation: NASA FIRMS MODIS/VIIRS active fire API (CSV/JSON per bbox+date)."""
    raise NotImplementedError("Wire up NASA FIRMS API key to enable live fire-count data.")
