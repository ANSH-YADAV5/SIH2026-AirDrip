# Surface AQI Estimation & HCHO Hotspot Intelligence Platform

**USAR, GGSIPU — Hackathon Prototype**
*AI-driven geospatial platform for Surface AQI estimation and HCHO hotspot detection over India, in support of the National Clean Air Programme (NCAP).*

## Technical Approach

**The idea, in one line:** satellites see the whole air column from space; we use AI to bring that down to ground-level pollution, producing one AQI number (with a source) for every 2 km grid cell across India.

**Input — five free, open data sources:**
| Source | What it gives us |
|---|---|
| INSAT-3D (MOSDAC) | Aerosol Optical Depth — satellite haze data |
| Sentinel-5P TROPOMI | HCHO, NO₂, SO₂, CO, O₃ column levels |
| CPCB ground stations | Real ground-truth AQI readings (data.gov.in) |
| ERA5 / IMDAA | Wind, humidity, temperature — for dispersion & correction |
| NASA FIRMS | Active fire locations (for biomass-burning correlation) |

**Process — six-stage pipeline (Python · TensorFlow/Keras · scikit-learn):**
1. **Grid Map** — standardize all data sources onto a common 2 km grid over India
2. **Sync Data** — align satellite timestamps/locations with ground station records
3. **AI Prediction** — a CNN-LSTM model estimates ground-level PM2.5 from the aligned features (satellite AOD, weather, seasonality)
4. **Calculate AQI** — apply the official CPCB 2014 sub-index breakpoint formula to the predicted PM2.5 (implemented from scratch in `utils/aqi_calculator.py`, not a library)
5. **Detect Hotspots** — statistical clustering (density-based) on HCHO concentration to flag real pollution clusters, filtering out noise
6. **Trace Source** — correlate HCHO hotspots against NASA FIRMS fire locations to distinguish biomass-burning pollution from other sources

**Output:** daily AQI map, HCHO hotspot map, fire↔smoke correlation, an auto-drafted GRAP advisory, and CSV/GeoJSON exports — all surfaced across the 8 screens of the Streamlit app.

**Actual system architecture** (no separate backend, no database — this is a single Python process):

```
Browser  →  Streamlit app (app.py)
                 │
                 ▼
        utils/ Python modules
   (data fetch · AQI calc · hotspot detection · forecasting)
                 │
                 ▼
   CPCB live API (data.gov.in) — or realistic
   simulated data when live feeds are unavailable
```

No FastAPI/Django/Flask layer, no MongoDB/PostgreSQL, no React frontend — Streamlit + Plotly render everything directly from the Python functions in `utils/`, in-memory, on each request. This keeps the prototype laptop-trainable and free to run (₹0 data cost), and matches exactly what's demoed live — nothing in this section describes a component that isn't actually in the code.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## What's in the box

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit dashboard (5 tabs: AQI map, HCHO hotspots, biomass burning impact, model performance, methodology) |
| `utils/aqi_calculator.py` | **Real** CPCB 2014 National AQI breakpoint formula — dominant-pollutant sub-index method |
| `utils/data_simulator.py` | Spatially/seasonally-realistic synthetic satellite+ground data generator (stands in for INSAT-3D/TROPOMI/CPCB/FIRMS live feeds) |
| `utils/model.py` | CNN-LSTM architecture spec + physically-motivated prediction proxy + hotspot clustering |

## Why simulated data, and how to go live

This environment can't reach MOSDAC, Copernicus/Google Earth Engine, CPCB's data.gov.in API, or NASA FIRMS (no credentials, no network egress to those domains from this sandbox). Rather than fake it invisibly, the data generator is built to mirror **real, well-documented spatial patterns** — Indo-Gangetic Plain loading, NCR urban core severity, the Punjab/Haryana/W-UP stubble-burning belt, winter inversion seasonality — so the dashboard behaves the way it would on real feeds.

To go live, replace the three stub functions at the bottom of `utils/data_simulator.py`:

```python
fetch_live_tropomi_hcho()   # -> Google Earth Engine, COPERNICUS/S5P/NRTI/L3_HCHO
fetch_live_cpcb()           # -> CPCB CAAQMS / data.gov.in AQI resource
fetch_live_firms()          # -> NASA FIRMS MODIS/VIIRS active fire API
```

Each has the exact real data source and access pattern commented in.

## What's genuinely real (not simulated) in this prototype

- **CPCB AQI formula** — full official 2014 breakpoint table + dominant-pollutant logic, in `aqi_calculator.py`
- **Model architecture** — the CNN-LSTM spec in `model.py` is the actual design to train on real AOD+CPCB pairs
- **Hotspot detection logic** and **dashboard/UX** are fully functional, not mocked

## Enabling Live CPCB Data

Core pollutant readings (PM2.5, PM10, NO2, SO2, CO, O3) can be pulled live
from CPCB via data.gov.in, blended into the otherwise-simulated dataset for
any city where a live station matches.

1. Register at **data.gov.in** and get your API key from **My Account**.
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and
   paste your key in:
   ```toml
   CPCB_API_KEY = "your-key-here"
   ```
   **Never commit `secrets.toml` to a public repo.** If deploying to
   Streamlit Community Cloud, don't use a file at all — paste the key into
   the app's **Settings → Secrets** panel in the dashboard instead.
3. Run the app — a "Use live CPCB ground data" toggle appears in the
   sidebar once a key is detected. Turn it on.
4. The hero banner shows the live/fallback status honestly: 🟢 if live data
   matched, 🟡/⚪ if it silently fell back to simulated (e.g. API down, no
   matching station for a city) — the app never crashes because of this
   external dependency, it just tells you what it's actually showing.

**Important:** this integration was built and unit-tested against a mocked
API response (see the parsing logic in `utils/data_simulator.py`), but the
live network call itself has **not** been verified end-to-end — my build
sandbox has no route to `api.data.gov.in`. Test it yourself with a real key
before relying on it in a live demo, and keep the toggle handy to switch
back to simulated data if the live API is ever slow/down during judging.

## Next steps for judges / post-hackathon

1. Get GEE + CPCB + FIRMS API access, wire up the three stub functions
2. Assemble a training set of paired (AOD patch, met sequence) → CPCB station readings
3. Train the CNN-LSTM per the architecture in `model.py`, replace `predict_surface_pm25` proxy
4. Swap percentile-threshold hotspot detection for DBSCAN or Getis-Ord Gi* spatial statistics
5. Add HYSPLIT-style back-trajectory transport modeling using ERA5/IMDAA wind fields
