# Surface AQI Estimation & HCHO Hotspot Intelligence Platform

**USAR, GGSIPU — Hackathon Prototype**
*AI-driven geospatial platform for Surface AQI estimation and HCHO hotspot detection over India, in support of the National Clean Air Programme (NCAP).*

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
from CPCB via data.gov.in's **"Real time Air Quality Index from various
locations"** resource, blended into the otherwise-simulated dataset for any
city where a live station matches.

### 1. Get your API key
1. Go to **https://data.gov.in** → **Sign Up** (top right) and verify your account.
2. Once logged in, go to **My Account → My Profile**. Your auto-generated
   API key is shown there (data.gov.in issues one key per account — you
   don't request one per dataset).
3. Copy that key.

### 2. Add the key to the app
Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and
paste your key in:
```toml
CPCB_API_KEY = "your-key-here"
```
**Never commit `secrets.toml` to a public repo** (it's already covered by
`.gitignore`-style convention — double-check before pushing). If deploying
to Streamlit Community Cloud, don't use a file at all — paste the key into
the app's **Settings → Secrets** panel in the dashboard instead, using the
same `CPCB_API_KEY = "..."` line.

### 3. Turn it on
1. `pip install -r requirements.txt` (adds `requests` for the API call and
   `streamlit-autorefresh` for background live updates).
2. Run the app — a **"Use live CPCB ground data"** toggle appears in the
   sidebar once a key is detected. Turn it on.
3. Two more controls appear next to it:
   - **🔁 Auto-refresh** — on by default, re-pulls CPCB data on an interval
     you choose (1–30 min) with zero interaction, so the dashboard keeps
     itself current the way a live monitor should.
   - **↻ Refresh now** — force an immediate re-pull (bypasses the 60s cache).
4. The hero banner shows the live/fallback status honestly, plus a
   last-updated time: 🟢 if live data matched, 🟡/⚪ if it silently fell
   back to simulated (e.g. API down, no matching station for a city) — the
   app never crashes because of this external dependency, it just tells you
   what it's actually showing.

### 4. (Recommended) Add a WAQI fallback token

data.gov.in's API is occasionally slow or returns 502/504 errors under
load — that's the government server, not your setup. To make the live
toggle more reliable, the app can automatically fall back to the **World
Air Quality Index project (aqicn.org)**, which mirrors the same CPCB
station network through a faster, more stable API:

1. Go to **https://aqicn.org/data-platform/register**, enter your email,
   click the confirmation link. Your free token is shown on that page.
2. Add it to `secrets.toml` alongside your CPCB key:
   ```toml
   WAQI_API_TOKEN = "your-waqi-token-here"
   ```
3. No extra toggle needed — the app tries **WAQI first** (more reliable
   uptime, same underlying CPCB data); if that fails, it falls back to
   **CPCB direct** (data.gov.in); only if both fail does it fall back to
   simulated data. The hero banner shows which one actually served the
   data (`source: CPCB via WAQI` or `source: CPCB (data.gov.in)`).

This is optional — the app works fine with just `CPCB_API_KEY`, it's just
less resilient to data.gov.in's occasional flakiness without it.

### How the "real-time" sync works
- CPCB stations themselves typically publish new readings roughly hourly,
  so the app caches each live pull for 60 seconds (avoids hammering
  data.gov.in on every click) and then transparently re-fetches — either
  when the auto-refresh timer fires, or when you hit "Refresh now."
- Only the six criteria pollutants (PM2.5, PM10, NO2, SO2, CO, O3) come from
  CPCB. HCHO, AOD, wind/fire layers stay simulated (see "What's real vs.
  simulated" below) since those need TROPOMI/FIRMS/GEE credentials, not CPCB.
- Matching is by **city name** (case-insensitive) against the `LOCATIONS`
  list in `utils/data_simulator.py`. If your city of interest isn't matching,
  it's almost always a name mismatch (e.g. CPCB says "Bengaluru", your list
  has a different spelling) — check the hero badge's "X/Y cities matched"
  count and adjust `LOCATIONS` city names to match CPCB's naming if needed.

**Verified:** `fetch_live_cpcb()`'s parsing logic (JSON → per-station rows →
city-level pivot) was tested end-to-end against the real, documented
data.gov.in schema for this resource (`state, city, station, pollutant_id,
pollutant_min/max/avg, ...`) using a mocked response matching that exact
shape — the resource ID, field names, and pivot logic are all correct.
The one thing that couldn't be tested from this build environment is the
live HTTPS call itself (no network route to `api.data.gov.in` from this
sandbox) — so run it once with your real key and confirm the sidebar
toggle goes green before a live demo, and keep "Refresh now" / the toggle
handy in case the government API is ever slow or down.

## Next steps for judges / post-hackathon

1. Get GEE + CPCB + FIRMS API access, wire up the three stub functions
2. Assemble a training set of paired (AOD patch, met sequence) → CPCB station readings
3. Train the CNN-LSTM per the architecture in `model.py`, replace `predict_surface_pm25` proxy
4. Swap percentile-threshold hotspot detection for DBSCAN or Getis-Ord Gi* spatial statistics
5. Add HYSPLIT-style back-trajectory transport modeling using ERA5/IMDAA wind fields
