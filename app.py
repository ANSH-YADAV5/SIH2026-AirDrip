"""
AirDrip — India's Air Quality, Ranked.
USAR, GGSIPU — Hackathon Prototype v2

Run:  streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from utils.aqi_calculator import compute_aqi, CATEGORIES, hcho_severity_tier
from utils.data_simulator import (generate_grid_snapshot, generate_time_series, LOCATIONS,
                                   BURNING_BELT, get_live_or_simulated_snapshot,
                                   SATELLITE_ESTIMATE_LOCATIONS, fetch_satellite_estimate)
from utils.model import predict_surface_pm25, cluster_hotspots, MODEL_ARCHITECTURE_SUMMARY, REPORTED_METRICS
from utils.forecast import forecast_city, forecast_summary
from utils.community import _seed_posts, confidence_boost, SYMPTOM_TAGS, LANDMARKS
from utils.policy import generate_policy_brief, get_grap_stage, GRAP_STAGES
from utils.layman import (translate_pollutant, translate_category, real_life_equivalent,
                           best_window_today, diurnal_curve_for, step_out_advice, PROFILE_SENSITIVITY, ACTIVITY_MULT)
from utils.leaderboard import build_leaderboard, improving_cities
from utils.india_geo import (load_state_geojson, all_state_names, get_view_for_state,
                              to_display_name, nearest_city_for_state, default_india_view)
from utils.shared_store import load_shared_posts, add_shared_post, like_shared_post

st.set_page_config(
    page_title="AirDrip | Is Your City's Air Quality Dripping or Trash?",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "layman_mode" not in st.session_state:
    st.session_state.layman_mode = True
if "entered" not in st.session_state:
    st.session_state.entered = False

# Detect the "Enter" click coming back from the landing page's JS (it
# navigates the top-level window with ?entered=true, since an iframe can't
# set Streamlit session_state directly).
if st.query_params.get("entered") == "true":
    st.session_state.entered = True
    try:
        del st.query_params["entered"]
    except KeyError:
        pass

# ---------------------------------------------------------------------------
# LANDING PAGE — full-screen interactive starfield + glass wordmark panel.
# Shown once per session; "Enter AirDrip" transitions into the main app.
# ---------------------------------------------------------------------------
if not st.session_state.entered:
    # Make the landing screen true full-bleed: strip Streamlit's default page
    # padding/chrome and match the outer page background to the landing's
    # own black, so there's no visible border/letterboxing around the iframe.
    st.markdown("""
    <style>
        html, body { background: #07070b !important; margin: 0; padding: 0; }
        [data-testid="stAppViewContainer"] { background: #07070b !important; }
        [data-testid="stHeader"] { display: none !important; }
        [data-testid="stToolbar"] { display: none !important; }
        div.block-container, [data-testid="stMainBlockContainer"] {
            padding: 0 !important; margin: 0 !important; max-width: 100% !important;
        }
        iframe {
            position: fixed !important; top: 0 !important; left: 0 !important;
            width: 100vw !important; height: 100vh !important; border: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Keep the India-facing orbital visual local to the app so the landing page does not
    # depend on a third-party image host or an internet connection.
    earth_background_path = os.path.join(os.path.dirname(__file__), "assets", "india-orbit-hero.png")
    with open(earth_background_path, "rb") as earth_background_file:
        earth_background_data = base64.b64encode(earth_background_file.read()).decode("ascii")

    # Small, pre-compressed equirectangular texture used to spin a lightweight
    # CSS-only globe (no video/canvas 3D) — cheap on both bytes and CPU.
    earth_spin_path = os.path.join(os.path.dirname(__file__), "assets", "earth-rotating-texture.jpg")
    with open(earth_spin_path, "rb") as earth_spin_file:
        earth_spin_data = base64.b64encode(earth_spin_file.read()).decode("ascii")

    landing_html = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,opsz,wght@0,9..144,600;1,9..144,500;1,9..144,600&family=Manrope:wght@400;500;600;700;800&display=swap');
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body {
        width: 100%; height: 100%; background: #02060b; overflow: hidden;
        font-family: 'Manrope', -apple-system, sans-serif;
    }
    #space { position: fixed; z-index: 1; top: 0; left: 0; width: 100%; height: 100%; display: block; pointer-events: none; }
    .earth-scene {
        position: fixed; z-index: 0; inset: 0;
        background: linear-gradient(90deg, rgba(1,5,12,.55) 0%, rgba(1,5,12,.18) 48%, rgba(1,5,12,.04) 100%), url('data:image/png;base64,__INDIA_EARTH_IMAGE__') center / cover no-repeat;
        animation: orbital-float 24s ease-in-out infinite alternate;
    }
    .earth-scene::after { content: ''; position: absolute; inset: 0; background: linear-gradient(180deg, rgba(1,5,12,.15), rgba(1,5,12,.42)), radial-gradient(ellipse at 76% 76%, transparent 10%, rgba(0, 2, 8, .36) 100%); }
    @keyframes orbital-float { from { transform: scale(1.015) translate3d(-.35%, .25%, 0); } to { transform: scale(1.07) translate3d(.55%, -.40%, 0); } }
    /* Lightweight spinning globe: two copies of a small equirectangular
       texture placed side by side inside a circular mask, translated by
       exactly one copy-width in a linear loop. Pure CSS transform (GPU),
       no video/canvas — cheap even on low-end devices. */
    .earth-globe {
        position: fixed; z-index: 1; pointer-events: none;
        width: 30vmin; height: 30vmin; max-width: 380px; max-height: 380px; min-width: 190px; min-height: 190px;
        right: 6%; top: 16%; border-radius: 50%; overflow: hidden;
        box-shadow: 0 0 70px 6px rgba(90, 210, 185, 0.16), inset -46px -18px 70px rgba(0,0,0,.62), inset 26px 14px 46px rgba(255,255,255,.05);
        opacity: 0.9;
    }
    .earth-globe-strip { display: flex; width: 200%; height: 100%; animation: earth-spin 55s linear infinite; will-change: transform; }
    .earth-globe-strip img { width: 50%; height: 100%; object-fit: cover; display: block; flex-shrink: 0; }
    .earth-globe::after {
        content: ''; position: absolute; inset: 0; border-radius: 50%; pointer-events: none;
        background: radial-gradient(circle at 30% 28%, rgba(255,255,255,.20), transparent 42%),
                    radial-gradient(circle at 70% 78%, rgba(0,0,0,.58), transparent 62%);
    }
    @keyframes earth-spin { from { transform: translateX(0); } to { transform: translateX(-50%); } }
    @media (max-width: 680px) { .earth-globe { display: none; } }
    .glow {
        position: fixed; border-radius: 50%; filter: blur(130px); pointer-events: none;
    }
    .glow-teal { width: 52vw; height: 52vw; top: -22%; right: -16%; background: #0b8c85; opacity: 0.34; }
    .glow-amber { width: 44vw; height: 44vw; bottom: -20%; left: -14%; background: #dd9a35; opacity: 0.25; }
    .topbar {
        position: fixed; inset: 0 0 auto; z-index: 6; display: flex; align-items: center;
        justify-content: space-between; padding: 26px clamp(24px, 5vw, 72px); color: #f4f8f4;
    }
    .brand-mini { font-family: 'Fraunces', serif; font-size: 1.4rem; font-style: italic; }
    .network-status { display: flex; align-items: center; gap: 8px; font: 500 0.67rem 'DM Mono', monospace; letter-spacing: 0.06em; color: #b5ccc5; text-transform: uppercase; }
    .live-dot { width: 7px; height: 7px; border-radius: 50%; background: #74e0b9; box-shadow: 0 0 12px #74e0b9; }
    .landing-wrap {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        display: flex; align-items: center; justify-content: flex-start; padding: 80px 8vw 32px;
    }
    .panel {
        position: relative; z-index: 5; width: min(680px, 100%); padding: clamp(32px, 4.3vw, 60px);
        overflow: hidden; isolation: isolate;
        background: linear-gradient(125deg, rgba(241,255,250,0.16) 0%, rgba(106,181,168,0.10) 42%, rgba(10,31,38,0.42) 100%);
        backdrop-filter: blur(28px) saturate(155%); -webkit-backdrop-filter: blur(28px) saturate(155%);
        border: 1px solid rgba(222, 255, 245, 0.30); border-radius: 32px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.30), inset 14px 0 36px rgba(147,238,213,.06), 0 28px 90px rgba(0,0,0,0.42);
    }
    .panel::before { content: ''; position: absolute; z-index: -1; width: 54%; height: 150%; top: -44%; left: -17%; border-radius: 50%; background: linear-gradient(150deg, rgba(255,255,255,.32), rgba(164,236,218,0)); filter: blur(22px); transform: rotate(18deg); pointer-events: none; }
    .panel::after { content: ''; position: absolute; z-index: -1; width: 100%; height: 1px; left: 0; top: 17%; background: linear-gradient(90deg, transparent, rgba(226,255,247,.30), transparent); pointer-events: none; }
    .eyebrow {
        font: 500 0.7rem 'DM Mono', monospace; text-transform: uppercase; letter-spacing: 0.16em;
        color: #9dd8c8; margin-bottom: 18px;
    }
    .wordmark {
        font-family: 'Fraunces', serif; font-style: italic; font-weight: 550;
        font-size: clamp(3.5rem, 8vw, 6.8rem); color: #f5fbf7; line-height: .9; letter-spacing: -0.045em;
        background: linear-gradient(135deg, #ffffff 3%, #bff7e9 37%, #72cdbc 64%, #ffffff 100%);
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        text-shadow: 0 0 44px rgba(78, 201, 175, 0.22); filter: drop-shadow(0 2px 0 rgba(255,255,255,.10));
    }
    .tagline {
        max-width: 560px; margin-top: 23px; font-size: clamp(0.96rem, 2vw, 1.08rem); color: #c0d4ce; line-height: 1.65;
        font-weight: 400; letter-spacing: 0.005em;
    }
    .hero-content { display: block; }
    .aqi-preview { padding: 20px; border: 1px solid rgba(192, 235, 221, 0.14); border-radius: 20px; background: rgba(6, 20, 25, 0.35); }
    .aqi-preview .caption { font: 500 0.62rem 'DM Mono', monospace; color: #94bcb1; text-transform: uppercase; letter-spacing: .1em; }
    .aqi-value { margin: 7px 0 2px; color: #f4b75c; font: 600 3.4rem/1 'Fraunces', serif; }
    .aqi-label { color: #f3c67f; font-size: .78rem; font-weight: 700; }
    .mini-line { height: 4px; margin: 17px 0 11px; overflow: hidden; border-radius: 999px; background: rgba(255,255,255,.08); }
    .mini-line span { display: block; width: 68%; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #75d9bb, #f4b75c, #e7775a); }
    .mini-foot { color: #a4bcb4; font-size: .67rem; }
    .actions { display: flex; align-items: center; gap: 18px; margin-top: 34px; }
    .enter-btn {
        position: relative; display: inline-block; padding: 15px 27px; border-radius: 999px;
        font-family: 'Manrope', sans-serif; font-size: 0.75rem; font-weight: 800;
        text-transform: uppercase; letter-spacing: 0.14em;
        color: #08211f; background: linear-gradient(100deg, #8fe7ce, #d9f59a);
        border: none; cursor: pointer;
        box-shadow: 0 12px 28px rgba(85, 203, 167, 0.24), inset 0 1px 0 rgba(255,255,255,.58);
        transition: transform 0.22s cubic-bezier(.2,.9,.25,1.35), box-shadow 0.22s ease, filter 0.22s ease;
    }
    .enter-btn::before { content: ''; position: absolute; inset: -7px; z-index: -1; border-radius: inherit; background: rgba(143,231,206,.30); filter: blur(13px); opacity: 0; transition: opacity .22s ease; }
    .enter-btn:hover { transform: translateY(-8px) scale(1.08); filter: brightness(1.08); box-shadow: 0 24px 44px rgba(85, 203, 167, 0.50), inset 0 1px 0 rgba(255,255,255,.7); }
    .enter-btn:hover::before { opacity: 1; }
    .enter-btn:active { transform: translateY(-3px) scale(1.03); }
    .hint { font-size: 0.73rem; color: #94aaa3; letter-spacing: 0.02em; }
    .feature-list { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 38px; padding-top: 23px; border-top: 1px solid rgba(202,232,223,.11); }
    .feature { padding: 8px 11px; border-radius: 999px; color: #b7d1c7; background: rgba(190,235,219,.06); border: 1px solid rgba(190,235,219,.1); font-size: .68rem; font-weight: 600; }
    @media (max-width: 1200px) and (min-width: 681px) {
        .landing-wrap { padding-left: 6vw; } .panel { width: min(560px, 62vw); padding: 42px; }
        .wordmark { font-size: clamp(3.4rem, 7vw, 5.3rem); }
    }
    @media (max-width: 680px) {
        .topbar { padding: 20px 24px; } .network-status { font-size: .56rem; }
        .landing-wrap { padding: 70px 16px 16px; } .panel { border-radius: 24px; }
        .actions { flex-wrap: wrap; margin-top: 28px; } .feature-list { margin-top: 28px; }
    }
</style>
</head>
<body>
    <canvas id="space"></canvas>
    <div class="earth-scene" aria-hidden="true"></div>
    <div class="earth-globe" aria-hidden="true">
        <div class="earth-globe-strip">
            <img src="data:image/jpeg;base64,__EARTH_SPIN_IMAGE__" alt="">
            <img src="data:image/jpeg;base64,__EARTH_SPIN_IMAGE__" alt="">
        </div>
    </div>
    <div class="glow glow-teal"></div>
    <div class="glow glow-amber"></div>
    <header class="topbar">
        <div class="brand-mini">AirDrip</div>
        <div class="network-status"><span class="live-dot"></span> India air network</div>
    </header>
    <div class="landing-wrap">
        <div class="panel">
            <div class="hero-content">
                <div class="eyebrow">India's living air-quality network</div>
                <div class="wordmark">Breathe<br>informed.</div>
                <div class="tagline">AirDrip turns satellite signals, ground readings, and citizen reports into clear, local guidance for every day outside.</div>
                <div class="actions">
                    <button class="enter-btn" onclick="enterApp()">Explore your air &rarr;</button>
                    <div class="hint">Live insight. Plain language.</div>
                </div>
            </div>
            <div class="feature-list">
                <span class="feature">Satellite intelligence</span>
                <span class="feature">30-day forecasts</span>
                <span class="feature">Community signals</span>
                <span class="feature">Policy-ready alerts</span>
            </div>
        </div>
    </div>
<script>
    function enterApp() {
        try {
            window.top.location.href = window.top.location.pathname + "?entered=true";
        } catch (e) {
            window.location.href = window.location.pathname + "?entered=true";
        }
    }

    const canvas = document.getElementById('space');
    const ctx = canvas.getContext('2d');
    let W = window.innerWidth, H = window.innerHeight;
    canvas.width = W; canvas.height = H;
    let cx = W / 2, cy = H / 2;

    // Atmospheric particles drift behind the content, tying the motion to air
    // rather than a generic space scene. Their response to the pointer is subtle.
    const particles = Array.from({ length: 95 }, () => ({
        x: Math.random() * W, y: Math.random() * H,
        size: Math.random() * 2.4 + 0.5,
        speed: Math.random() * 0.24 + 0.06,
        drift: Math.random() * Math.PI * 2,
        opacity: Math.random() * 0.25 + 0.04,
    }));

    let mouseX = W / 2, mouseY = H / 2;
    let hasMouse = false;
    window.addEventListener('mousemove', (e) => {
        mouseX = e.clientX; mouseY = e.clientY; hasMouse = true;
    });
    window.addEventListener('resize', () => {
        W = window.innerWidth; H = window.innerHeight;
        canvas.width = W; canvas.height = H;
        cx = W / 2; cy = H / 2;
    });

    let t = 0;
    function animate() {
        t += 1;
        ctx.clearRect(0, 0, W, H);

        // A soft cursor parallax gives the background a little depth.
        let parX = 0, parY = 0;
        if (hasMouse) {
            parX = (mouseX - W / 2) * 0.02;
            parY = (mouseY - H / 2) * 0.02;
        }
        const haze = ctx.createLinearGradient(0, 0, W, H);
        haze.addColorStop(0, 'rgba(74, 187, 166, 0.08)');
        haze.addColorStop(0.56, 'rgba(5, 20, 27, 0)');
        haze.addColorStop(1, 'rgba(232, 166, 71, 0.08)');
        ctx.fillStyle = haze;
        ctx.fillRect(0, 0, W, H);

        particles.forEach((p, i) => {
            p.x += p.speed;
            p.y += Math.sin(t * 0.008 + p.drift) * 0.14;
            if (p.x > W + 8) p.x = -8;
            const shift = (i % 3 === 0) ? parX * 0.16 : parX * 0.05;
            ctx.beginPath();
            ctx.arc(p.x + shift, p.y + parY * 0.05, p.size, 0, Math.PI * 2);
            ctx.fillStyle = i % 5 === 0 ? `rgba(238, 183, 104, ${p.opacity})` : `rgba(169, 228, 211, ${p.opacity})`;
            ctx.fill();
        });

        requestAnimationFrame(animate);
    }
    animate();
</script>
</body>
</html>
"""
    landing_html = landing_html.replace("__INDIA_EARTH_IMAGE__", earth_background_data)
    landing_html = landing_html.replace("__EARTH_SPIN_IMAGE__", earth_spin_data)
    components.html(landing_html, height=900, scrolling=False)
    st.stop()

# ---------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 💧 AirDrip Control Panel")
    st.caption("USAR, GGSIPU — NCAP Decision Support Prototype")

    theme_choice = st.radio("🎨 Theme", ["🌙 Night", "☀️ Day"], horizontal=True,
                             index=0 if st.session_state.theme == "dark" else 1)
    st.session_state.theme = "dark" if theme_choice == "🌙 Night" else "day"

    layman_choice = st.toggle("🗣️ Simple language mode", value=st.session_state.layman_mode,
                               help="Switch between plain everyday language and technical/scientific terms")
    st.session_state.layman_mode = layman_choice

    st.markdown("---")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    sel_month = st.selectbox("Month (seasonal simulation)", options=list(range(1, 13)),
                              format_func=lambda m: month_names[m-1], index=10)
    sel_day = st.slider("Day of month", 1, 28, 3)

    st.markdown("---")
    pollutant_view = st.selectbox(
        "Map layer",
        ["AQI (composite)", "PM2.5", "NO2", "SO2", "O3", "HCHO", "AOD (satellite)"],
    )
    show_fire = st.checkbox("Overlay fire activity 🔥", value=True)
    show_belt = st.checkbox("Highlight burning belt", value=True)

    st.markdown("---")
    st.markdown("##### 🛰️ Data Source")
    cpcb_key_from_secrets = None
    try:
        cpcb_key_from_secrets = st.secrets.get("CPCB_API_KEY")
    except Exception:
        pass
    cpcb_key_from_env = os.environ.get("CPCB_API_KEY")
    cpcb_api_key = cpcb_key_from_secrets or cpcb_key_from_env

    waqi_key_from_secrets = None
    try:
        waqi_key_from_secrets = st.secrets.get("WAQI_API_TOKEN")
    except Exception:
        pass
    waqi_api_token = waqi_key_from_secrets or os.environ.get("WAQI_API_TOKEN")

    use_live_data = st.toggle(
        "Use live CPCB ground data",
        value=False,
        disabled=not (cpcb_api_key or waqi_api_token),
        help="Pulls real-time PM2.5/PM10/NO2/SO2/CO/O3 readings from CPCB "
             "government monitoring stations (via WAQI's mirror, with "
             "data.gov.in as a direct fallback), blended with simulated "
             "satellite layers (HCHO, AOD, fire, wind). Falls back to full "
             "simulation automatically if both live sources are unreachable.",
    )
    if not (cpcb_api_key or waqi_api_token):
        st.caption("No WAQI_API_TOKEN or CPCB_API_KEY found in secrets — add at least one to enable this (see README).")
    if waqi_api_token and not cpcb_api_key:
        st.caption("💡 Tip: add a CPCB_API_KEY too (see README) so the app has a second fallback "
                   "if WAQI is ever unreachable.")
    elif cpcb_api_key and not waqi_api_token:
        st.caption("💡 Tip: add a WAQI_API_TOKEN too (free, see README) — it's used as the primary "
                   "live source (faster, more reliable) with CPCB direct as fallback.")

    auto_refresh_live = False
    refresh_minutes = 5
    manual_refresh_clicked = False
    if use_live_data and (cpcb_api_key or waqi_api_token):
        rcol1, rcol2 = st.columns([1.3, 1])
        with rcol1:
            auto_refresh_live = st.toggle("🔁 Auto-refresh", value=True,
                                           help="Automatically re-pull live CPCB data on an interval, "
                                                "without you touching anything.")
        with rcol2:
            manual_refresh_clicked = st.button("↻ Refresh now", use_container_width=True)
        if auto_refresh_live:
            refresh_minutes = st.select_slider("Refresh every", options=[1, 2, 5, 10, 15, 30], value=5,
                                                format_func=lambda m: f"{m} min")
            try:
                from streamlit_autorefresh import st_autorefresh
                st_autorefresh(interval=refresh_minutes * 60 * 1000, key="live_cpcb_autorefresh")
            except ImportError:
                st.caption("Install `streamlit-autorefresh` (see requirements.txt) to enable "
                           "automatic background refresh; use ↻ Refresh now meanwhile.")
        if "last_live_fetch_ts" in st.session_state:
            st.caption(f"🕒 Last fetched: {st.session_state.last_live_fetch_ts}")

    st.markdown("---")
    st.caption(
        "⚠️ Demo mode: satellite/FIRMS feeds are realistically simulated "
        "(live APIs need credentials not available in this build sandbox — "
        "see README for the real integration path). Core pollutant readings "
        "can be live CPCB data when the toggle above is on."
    )

LAYMAN = st.session_state.layman_mode
THEME = st.session_state.theme

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
df = generate_grid_snapshot(month=sel_month, day=sel_day)
df["data_tier"] = "ground_station"  # this city has a real CPCB/WAQI station nearby

# Cities with no nearby ground station: start from the same simulated
# baseline (for HCHO/fire/wind/AOD, which stay simulated regardless of
# source) but tag them separately so pollutant values can be swapped for a
# real satellite-model estimate below instead of staying purely synthetic.
df_sat = generate_grid_snapshot(month=sel_month, day=sel_day, locations=SATELLITE_ESTIMATE_LOCATIONS)
df_sat["data_tier"] = "satellite_estimate"
df = pd.concat([df, df_sat], ignore_index=True)


@st.cache_data(ttl=60, show_spinner=False)
def _cached_live_fetch(api_key, month, day, waqi_token):
    """Caches the live pull for 60s so autorefresh ticks / widget reruns don't
    hammer the upstream API — still 'real time' for a live-monitoring
    dashboard, since stations themselves typically update hourly."""
    return get_live_or_simulated_snapshot(api_key, month, day, waqi_token=waqi_token)


@st.cache_data(ttl=60, show_spinner=False)
def _cached_satellite_estimate():
    """Caches the Open-Meteo/CAMS satellite-model pull for 60s. Raises on
    failure — caller falls back to the simulated baseline for these cities."""
    return fetch_satellite_estimate()


live_data_status = "simulated"  # "live", "partial", "simulated", or "error: ..."
df["_is_waqi_indexed"] = False  # tracks rows whose values are WAQI AQI-sub-indices, not raw concentrations
satellite_status = "simulated"  # status specifically for the no-ground-station cities
if use_live_data and (cpcb_api_key or waqi_api_token):
    if manual_refresh_clicked:
        _cached_live_fetch.clear()
    try:
        live_df, source, live_err = _cached_live_fetch(cpcb_api_key, sel_month, sel_day, waqi_api_token)
        st.session_state.last_live_fetch_ts = datetime.now().strftime("%H:%M:%S")
        if source in ("cpcb", "waqi") and len(live_df) > 0:
            live_lookup = live_df.set_index(live_df["city"].str.strip().str.lower())
            pollutant_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
            matched_cities = 0
            for idx, row in df.iterrows():
                key = row["city"].strip().lower()
                if key in live_lookup.index:
                    live_row = live_lookup.loc[key]
                    any_matched = False
                    for pol in pollutant_cols:
                        val = live_row[pol] if pol in live_row else np.nan
                        if pd.notna(val):
                            df.at[idx, pol] = float(val)
                            any_matched = True
                    if any_matched:
                        matched_cities += 1
                        # WAQI's per-pollutant values are already AQI sub-indices
                        # (0-500 scale), not raw µg/m3 concentrations — flag this
                        # row so compute_aqi() below doesn't re-run them through
                        # the CPCB concentration breakpoint table a second time
                        # (that double-conversion was inflating readings, e.g.
                        # a real "Moderate" city showing as 500/"Severe").
                        if source == "waqi":
                            df.at[idx, "_is_waqi_indexed"] = True
            source_label = "CPCB via WAQI" if source == "waqi" else "CPCB (data.gov.in)"
            live_data_status = f"live ({matched_cities}/{len(df)} cities matched — source: {source_label})" if matched_cities > 0 else \
                "simulated (live API returned data, but no city names matched our station list)"
        else:
            live_data_status = f"simulated (live fetch failed: {live_err})"
    except Exception as e:
        live_data_status = f"simulated (unexpected error: {e})"

    # Overlay a real satellite-informed estimate (Open-Meteo/CAMS) onto the
    # cities that have no ground station at all — this is what actually
    # answers "CPCB can't be everywhere," instead of leaving them on a
    # purely synthetic formula.
    try:
        sat_df = _cached_satellite_estimate()
        sat_lookup = sat_df.set_index(sat_df["city"].str.strip().str.lower())
        sat_pollutant_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
        sat_matched = 0
        for idx, row in df.iterrows():
            if row["data_tier"] != "satellite_estimate":
                continue
            key = row["city"].strip().lower()
            if key in sat_lookup.index:
                sat_row = sat_lookup.loc[key]
                any_matched = False
                for pol in sat_pollutant_cols:
                    val = sat_row[pol] if pol in sat_row else np.nan
                    if pd.notna(val):
                        df.at[idx, pol] = float(val)
                        any_matched = True
                if "AOD" in sat_row and pd.notna(sat_row["AOD"]):
                    df.at[idx, "AOD"] = float(sat_row["AOD"])
                if any_matched:
                    sat_matched += 1
                    df.at[idx, "data_tier"] = "satellite_estimate_live"
        n_sat_total = len(SATELLITE_ESTIMATE_LOCATIONS)
        satellite_status = f"live ({sat_matched}/{n_sat_total} — source: Open-Meteo/CAMS satellite model)" \
            if sat_matched > 0 else "simulated (satellite API returned no usable data)"
    except Exception as e:
        satellite_status = f"simulated (satellite estimate fetch failed: {e})"


aqi_results = df.apply(lambda r: compute_aqi({
    "PM2.5": r["PM2.5"], "PM10": r["PM10"], "NO2": r["NO2"],
    "SO2": r["SO2"], "CO": r["CO"], "O3": r["O3"],
}, already_index=bool(r["_is_waqi_indexed"])), axis=1)
df["AQI"] = aqi_results.apply(lambda x: x["AQI"])
df["AQI_category"] = aqi_results.apply(lambda x: x["category"])
df["AQI_color"] = aqi_results.apply(lambda x: x["color"])
df["dominant_pollutant"] = aqi_results.apply(lambda x: x["dominant"])
df["HCHO_tier"] = df["HCHO"].apply(hcho_severity_tier)

national_avg_aqi = df["AQI"].mean()
national_category, national_color = None, "#38bdf8"
for lo, hi, name, color in CATEGORIES:
    if lo <= national_avg_aqi <= hi:
        national_category, national_color = name, color
        break
if national_category is None:
    national_category, national_color = "Severe", "#7e0023"

worst_row = df.loc[df["AQI"].idxmax()]
best_row = df.loc[df["AQI"].idxmin()]
total_fires = int(df["fire_count"].sum())
hcho_hotspot_count = (df["HCHO"] >= np.percentile(df["HCHO"], 75)).sum()

# ---------------------------------------------------------------------------
# THEME CSS — Nexera-inspired: black cosmic backdrop, starfield, soft glow
# orbs, dark charcoal cards, italic serif wordmark, gradient pill CTAs.
# ---------------------------------------------------------------------------
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

acc_r, acc_g, acc_b = hex_to_rgb(national_color)


def _make_starfield_svg(n_stars=140, width=1600, height=1200, seed=7, star_color="255,255,255"):
    rng = np.random.default_rng(seed)
    dots = []
    for _ in range(n_stars):
        x = rng.uniform(0, width)
        y = rng.uniform(0, height)
        r = rng.choice([0.6, 0.9, 1.2, 1.6], p=[0.45, 0.30, 0.17, 0.08])
        op = rng.uniform(0.25, 0.9)
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="rgb({star_color})" opacity="{op:.2f}"/>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">{"".join(dots)}</svg>'
    import base64
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def _make_grid_dots_svg(spacing=42, width=1680, height=1260, dot_color="15,15,25", dot_opacity=0.14):
    """Clean, evenly-spaced dot grid for Day mode — a premium technical
    texture (Stripe/Linear-style), rather than trying to force 'stars' onto
    a bright background where they'd just read as gray smudges."""
    dots = []
    x = spacing / 2
    while x < width:
        y = spacing / 2
        while y < height:
            dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.1" fill="rgb({dot_color})" opacity="{dot_opacity}"/>')
            y += spacing
        x += spacing
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">{"".join(dots)}</svg>'
    import base64
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


if THEME == "dark":
    bg_base = "#07070b"
    text_primary = "#f5f5f7"
    text_secondary = "#9a9aa5"
    card_bg = "rgba(255,255,255,0.05)"
    card_bg_soft = "rgba(255,255,255,0.045)"
    card_border = "rgba(255,255,255,0.11)"
    card_shadow = "0 10px 36px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06)"
    card_blur = "blur(22px) saturate(180%)"
    glow_opacity = 0.5
    plot_paper = "rgba(0,0,0,0)"
    plot_area = "rgba(255,255,255,0.02)"
    grid_color = "rgba(255,255,255,0.06)"
    input_bg = "rgba(255,255,255,0.05)"
    BG_TEXTURE_URL = _make_starfield_svg(star_color="255,255,255")
    bg_texture_opacity = 1.0
else:
    bg_base = "#f7f5f1"
    text_primary = "#191919"
    text_secondary = "#5c5c68"
    card_bg = "rgba(255,255,255,0.62)"
    card_bg_soft = "rgba(255,255,255,0.58)"
    card_border = "rgba(25,25,25,0.09)"
    card_shadow = "0 10px 30px rgba(40,30,20,0.09), inset 0 1px 0 rgba(255,255,255,0.7)"
    card_blur = "blur(18px) saturate(160%)"
    glow_opacity = 0.20
    plot_paper = "rgba(0,0,0,0)"
    plot_area = "rgba(15,15,20,0.02)"
    grid_color = "rgba(15,15,20,0.07)"
    input_bg = "rgba(255,255,255,0.55)"
    BG_TEXTURE_URL = _make_grid_dots_svg()
    bg_texture_opacity = 1.0

# Premium jewel-tone accent duo: deep amethyst + champagne gold (replaces
# the brighter neon purple/amber pairing for a more refined, upscale feel).
ACCENT_PURPLE = "#8b5cf6"
ACCENT_GOLD = "#d4af37"
GRADIENT_CTA = f"linear-gradient(90deg, {ACCENT_PURPLE}, {ACCENT_GOLD})"

# Embed the landing hero image as a data URI so the main app can reuse
# the same orbital/india background without depending on external hosts.
earth_background_path = os.path.join(os.path.dirname(__file__), "assets", "india-orbit-hero.png")
try:
    with open(earth_background_path, "rb") as _f:
        EARTH_BG_B64 = base64.b64encode(_f.read()).decode("ascii")
except Exception:
    EARTH_BG_B64 = ""

# Small pre-compressed equirectangular texture for the spinning CSS globe
# (same lightweight trick used on the landing page — see there for notes).
earth_spin_path = os.path.join(os.path.dirname(__file__), "assets", "earth-rotating-texture.jpg")
try:
    with open(earth_spin_path, "rb") as _f:
        EARTH_SPIN_B64 = base64.b64encode(_f.read()).decode("ascii")
except Exception:
    EARTH_SPIN_B64 = ""


st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@1,9..144,500;1,9..144,600&family=Manrope:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], * {{ font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif !important; }}
    .brand-wordmark, .wordmark {{ font-family: 'Fraunces', serif !important; }}

    [data-testid="stAppViewContainer"] {{ background: {bg_base}; position: relative; overflow-x: hidden; }}
    [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}

    .cosmic-bg {{
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        z-index: 0; pointer-events: none; overflow: hidden;
        background-image: url('{BG_TEXTURE_URL}');
        background-size: cover; background-position: center;
        opacity: {bg_texture_opacity};
    }}
    .glow {{ position: absolute; border-radius: 50%; filter: blur(110px); opacity: {glow_opacity}; }}
    .glow-purple {{ width: 46vw; height: 46vw; top: -14%; right: -12%; background: radial-gradient(circle, {ACCENT_PURPLE} 0%, transparent 70%); animation: drift1 26s ease-in-out infinite; }}
    .glow-gold {{ width: 40vw; height: 40vw; bottom: -16%; left: -10%; background: radial-gradient(circle, {ACCENT_GOLD} 0%, transparent 70%); animation: drift2 30s ease-in-out infinite; }}
    .glow-aqi {{ width: 32vw; height: 32vw; top: 40%; left: 35%; background: radial-gradient(circle, rgb({acc_r},{acc_g},{acc_b}) 0%, transparent 70%); opacity: 0.18; animation: drift3 34s ease-in-out infinite; }}
    @keyframes drift1 {{ 0%, 100% {{ transform: translate(0,0); }} 50% {{ transform: translate(-4vw, 5vh); }} }}
    @keyframes drift2 {{ 0%, 100% {{ transform: translate(0,0); }} 50% {{ transform: translate(4vw, -4vh); }} }}
    @keyframes drift3 {{ 0%, 100% {{ transform: translate(0,0) scale(1); }} 50% {{ transform: translate(-3vw, -3vh) scale(1.1); }} }}

    /* Landing-derived background + glows (partial theme: background + accent colors) */
    .earth-scene {{
        position: fixed; z-index: 0; inset: 0;
        background: linear-gradient(90deg, rgba(1,5,12,.55) 0%, rgba(1,5,12,.18) 48%, rgba(1,5,12,.04) 100%), url('data:image/png;base64,{EARTH_BG_B64}') center / cover no-repeat;
        animation: orbital-float 24s ease-in-out infinite alternate;
        opacity: 0.92;
    }}
    @keyframes orbital-float {{ from {{ transform: scale(1.015) translate3d(-.35%, .25%, 0); }} to {{ transform: scale(1.07) translate3d(.55%, -.40%, 0); }} }}
    .glow-teal {{ width: 52vw; height: 52vw; top: -22%; right: -16%; background: #0b8c85; opacity: 0.34; filter: blur(130px); position: absolute; border-radius:50%; }}
    .glow-amber {{ width: 44vw; height: 44vw; bottom: -20%; left: -14%; background: #dd9a35; opacity: 0.25; filter: blur(130px); position: absolute; border-radius:50%; }}

    /* Lightweight spinning globe (CSS transform only — no video/canvas). */
    .earth-globe {{
        position: fixed; z-index: 0; pointer-events: none;
        width: 22vmin; height: 22vmin; max-width: 260px; max-height: 260px; min-width: 140px; min-height: 140px;
        right: 4%; top: 14%; border-radius: 50%; overflow: hidden;
        box-shadow: 0 0 60px 4px rgba(90, 210, 185, 0.14), inset -36px -14px 56px rgba(0,0,0,.6), inset 20px 10px 36px rgba(255,255,255,.05);
        opacity: 0.5;
    }}
    .earth-globe-strip {{ display: flex; width: 200%; height: 100%; animation: earth-spin 55s linear infinite; will-change: transform; }}
    .earth-globe-strip img {{ width: 50%; height: 100%; object-fit: cover; display: block; flex-shrink: 0; }}
    .earth-globe::after {{
        content: ''; position: absolute; inset: 0; border-radius: 50%; pointer-events: none;
        background: radial-gradient(circle at 30% 28%, rgba(255,255,255,.18), transparent 42%),
                    radial-gradient(circle at 70% 78%, rgba(0,0,0,.55), transparent 62%);
    }}
    @keyframes earth-spin {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}
    @media (max-width: 900px) {{ .earth-globe {{ display: none; }} }}

    [data-testid="stSidebar"] {{
        background: {card_bg}; backdrop-filter: {card_blur}; -webkit-backdrop-filter: {card_blur};
        border-right: 1px solid {card_border};
    }}
    h1, h2, h3, h4 {{ color: {text_primary} !important; letter-spacing: -0.01em; }}
    p, span, label, .stMarkdown {{ color: {text_secondary}; }}
    [data-testid="stMarkdownContainer"] p {{ color: {text_secondary}; }}

    /* Liquid glass everywhere: form controls (dropdowns, text inputs, sliders,
       expanders, dataframes) get the same frosted-glass treatment as cards,
       instead of Streamlit's flat default look, for a consistent feel. */
    [data-testid="stSelectbox"] > div > div, [data-testid="stTextInput"] > div > div,
    [data-testid="stTextArea"] > div, [data-testid="stMultiSelect"] > div > div,
    [data-testid="stDateInput"] > div > div,
    div[data-baseweb="select"] > div, div[data-baseweb="input"] {{
        background: {input_bg} !important; backdrop-filter: blur(14px) saturate(160%) !important;
        -webkit-backdrop-filter: blur(14px) saturate(160%) !important;
        border: 1px solid {card_border} !important; border-radius: 12px !important;
    }}
    [data-testid="stExpander"] {{
        background: {card_bg_soft}; backdrop-filter: {card_blur}; -webkit-backdrop-filter: {card_blur};
        border: 1px solid {card_border} !important; border-radius: 14px !important; overflow: hidden;
    }}
    [data-testid="stDataFrame"] {{
        background: {card_bg_soft}; backdrop-filter: {card_blur}; -webkit-backdrop-filter: {card_blur};
        border: 1px solid {card_border}; border-radius: 12px; overflow: hidden;
    }}
    [data-baseweb="popover"] [role="listbox"] {{
        background: {card_bg} !important; backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border: 1px solid {card_border} !important;
    }}

    .brand-wordmark {{
        font-family: 'Fraunces', serif; font-style: italic; font-weight: 550;
        color: {text_primary}; letter-spacing: -0.02em;
    }}
    .eyebrow {{
        font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.22em;
        color: {text_secondary}; font-weight: 600;
    }}

    /* Liquid glass card system: frosted blur + soft sheen highlight */
    .glass, .metric-card, .post-card {{
        background: {card_bg_soft}; backdrop-filter: {card_blur}; -webkit-backdrop-filter: {card_blur};
        border: 1px solid {card_border}; border-radius: 20px;
        box-shadow: {card_shadow}; position: relative; overflow: hidden;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }}
    .glass::before, .metric-card::before, .post-card::before {{
        content: ""; position: absolute; top: 0; left: -70%; width: 45%; height: 100%;
        background: linear-gradient(120deg, transparent, rgba(255,255,255,0.08), transparent);
        transform: skewX(-20deg); pointer-events: none;
    }}
    .metric-card:hover, .glass:hover {{ transform: translateY(-2px); }}
    .metric-card {{ padding: 18px 20px; text-align: center; }}
    .metric-card .value {{ font-size: 2.1rem; font-weight: 800; color: {text_primary}; }}
    .metric-card .label {{ font-size: 0.72rem; color: {text_secondary}; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2px; }}

    .hero-card {{
        background: linear-gradient(135deg, rgba({acc_r},{acc_g},{acc_b},0.10), {card_bg_soft});
        backdrop-filter: {card_blur}; -webkit-backdrop-filter: {card_blur};
        border: 1px solid {card_border}; border-radius: 24px; box-shadow: {card_shadow}; padding: 30px 34px;
        position: relative; overflow: hidden;
    }}

    .badge {{ display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: 0.72rem; font-weight: 700; color: white; letter-spacing: 0.05em; text-transform: uppercase; }}
    .disclaimer-box {{
        background: rgba(212,175,55,0.08); border: 1px solid rgba(212,175,55,0.25); border-left: 3px solid {ACCENT_GOLD};
        backdrop-filter: blur(10px); padding: 10px 16px; border-radius: 10px; font-size: 0.85rem; color: {text_primary};
    }}
    .post-card {{ padding: 14px 18px; margin-bottom: 12px; }}
    .post-card .handle {{ font-weight: 700; color: {ACCENT_PURPLE}; font-size: 0.9rem; }}
    .post-card .meta {{ color: {text_secondary}; font-size: 0.75rem; }}
    .post-card .tag-chip {{ display: inline-block; background: rgba(139,92,246,0.14); color: {ACCENT_PURPLE}; padding: 2px 10px; border-radius: 999px; font-size: 0.72rem; margin-top: 6px; }}
    .verified-chip {{ display: inline-block; background: rgba(34,197,94,0.16); color: #4ade80; padding: 2px 10px; border-radius: 999px; font-size: 0.68rem; margin-left: 6px; font-weight: 600; }}
    .grap-badge {{ display: inline-block; padding: 7px 18px; border-radius: 999px; font-weight: 800; font-size: 0.95rem; color: white; letter-spacing: 0.03em; }}

    .swipe-row {{ display: flex; overflow-x: auto; gap: 14px; padding: 6px 2px 14px 2px; scroll-snap-type: x mandatory; }}
    .swipe-row::-webkit-scrollbar {{ height: 6px; }}
    .swipe-row::-webkit-scrollbar-thumb {{ background: rgba(139,92,246,0.4); border-radius: 10px; }}
    .swipe-card {{ min-width: 260px; scroll-snap-align: start; flex-shrink: 0; }}

    /* Primary CTA: liquid-glass gradient pill with soft glow */
    .cta-primary {{
        display: inline-block; padding: 11px 26px; border-radius: 999px; text-decoration: none;
        font-size: 0.78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em;
        color: white; background: {GRADIENT_CTA};
        box-shadow: 0 0 24px rgba(139,92,246,0.4), 0 0 24px rgba(212,175,55,0.22);
        border: none; transition: all 0.2s ease;
    }}
    .cta-primary:hover {{ transform: translateY(-1px); box-shadow: 0 0 32px rgba(139,92,246,0.55), 0 0 32px rgba(212,175,55,0.3); }}

    .share-btn {{
        display: inline-block; padding: 6px 14px; border-radius: 999px; text-decoration: none;
        font-size: 0.75rem; font-weight: 700; margin-right: 6px; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.04em;
        background: {input_bg}; backdrop-filter: blur(10px); color: {text_primary}; border: 1px solid {card_border};
    }}

    div[data-testid="stMetricValue"] {{ color: {text_primary}; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; flex-wrap: wrap; }}
    .stTabs [data-baseweb="tab"] {{ background: {card_bg_soft}; border-radius: 10px 10px 0 0; color: {text_secondary}; }}

    .nav-chip {{
        display: inline-flex; flex-direction: column; align-items: center; justify-content: center;
        gap: 6px; text-align: center; padding: 10px 18px; border-radius: 999px; font-weight: 700;
        font-size: 0.88rem; background: {card_bg_soft}; backdrop-filter: {card_blur}; -webkit-backdrop-filter: {card_blur};
        border: 1px solid {card_border}; color: {text_secondary}; line-height: 1.05;
        position: relative; overflow: hidden; transition: transform .18s ease, box-shadow .18s ease, background .18s ease;
    }}
    .nav-chip::before {{
        content: ""; position: absolute; left: -40%; top: 0; width: 60%; height: 100%;
        background: linear-gradient(120deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        transform: skewX(-20deg); pointer-events: none; opacity: 0.0; transition: opacity .22s ease;
    }}
    .nav-chip:hover::before {{ opacity: 1; }}
    .nav-chip:hover {{ transform: translateY(-4px); box-shadow: 0 10px 26px rgba(0,0,0,0.28); }}

    .nav-chip-active {{
        background: linear-gradient(90deg, rgba(139,92,246,0.95), rgba(212,175,55,0.95));
        color: white; border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 12px 36px rgba(139,92,246,0.18), inset 0 -6px 18px rgba(255,255,255,0.04);
        transform: translateY(-2px); z-index: 10;
    }}
    .nav-chip-active::after {{
        content: ""; position: absolute; inset: -6px; z-index: -1; border-radius: inherit;
        background: radial-gradient(circle at 20% 30%, rgba(139,92,246,0.12), transparent 25%);
        filter: blur(12px); opacity: 0.95;
    }}
    .stTabs [aria-selected="true"] {{ background: rgba(139,92,246,0.16) !important; color: {text_primary} !important; }}

    .stButton>button {{
        background: {GRADIENT_CTA}; border: none;
        color: white; border-radius: 999px; font-weight: 700; letter-spacing: 0.03em;
        box-shadow: 0 0 18px rgba(139,92,246,0.32); transition: all 0.2s ease;
    }}
    .stButton>button:hover {{ box-shadow: 0 0 26px rgba(139,92,246,0.5); transform: translateY(-1px); }}

    [data-testid="stTextInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stTextArea"] textarea {{
        background: {input_bg} !important; backdrop-filter: blur(10px) !important;
        border-radius: 12px !important; border: 1px solid {card_border} !important;
    }}
</style>
<div class="cosmic-bg">
    <div class="earth-scene" aria-hidden="true"></div>
    <div class="earth-globe" aria-hidden="true">
        <div class="earth-globe-strip">
            <img src="data:image/jpeg;base64,{EARTH_SPIN_B64}" alt="">
            <img src="data:image/jpeg;base64,{EARTH_SPIN_B64}" alt="">
        </div>
    </div>
    <div class="glow glow-teal"></div>
    <div class="glow glow-amber"></div>
    <div class="glow glow-purple"></div>
    <div class="glow glow-gold"></div>
    <div class="glow glow-aqi"></div>
</div>
""", unsafe_allow_html=True)

CHART_LAYOUT = dict(paper_bgcolor=plot_paper, plot_bgcolor=plot_area, font=dict(color=text_secondary),
                     xaxis=dict(gridcolor=grid_color), yaxis=dict(gridcolor=grid_color))


def pollutant_label(code):
    if LAYMAN:
        t = translate_pollutant(code)
        return f"{t['emoji']} {t['simple_name']}"
    return code


def share_links(text, url="https://airdrip.example.app"):
    import urllib.parse
    q = urllib.parse.quote(text)
    wa = f"https://wa.me/?text={q}%20{url}"
    tw = f"https://twitter.com/intent/tweet?text={q}"
    return f"""<a class="share-btn" href="{wa}" target="_blank">📱 WhatsApp</a>
    <a class="share-btn" href="{tw}" target="_blank">🐦 X / Twitter</a>"""


# ---------------------------------------------------------------------------
# NAV — sticky top navbar, story order:
# Today -> Forecast -> Community -> Leaderboard -> Hotspots -> Policy -> Model -> About
# ---------------------------------------------------------------------------
NAV_SECTIONS = [
    ("today", "🏠", "Today"),
    ("forecast", "🔮", "Forecast"),
    ("community", "💬", "Community"),
    ("leaderboard", "🏆", "Leaderboard"),
    ("hotspots", "🔥", "Hotspots"),
    ("policy", "📜", "Policy"),
    ("model", "🧠", "Model"),
    ("about", "ℹ️", "About"),
]
if "nav_section" not in st.session_state:
    st.session_state.nav_section = "today"

st.markdown(f"""
<style>
    /* Scoped, professional top navbar: sticks to the top of the page,
       frosted-glass background so it reads clearly against any content
       scrolling underneath it, consistent pill styling for every tab
       (active vs inactive distinguished only by fill, never by a clashing
       rainbow gradient on unrelated buttons). */
    .st-key-airdrip_topnav {{
        position: sticky; top: 0; z-index: 999;
        background: {card_bg}; backdrop-filter: blur(24px) saturate(180%);
        -webkit-backdrop-filter: blur(24px) saturate(180%);
        border-bottom: 1px solid {card_border};
        padding: 10px 4px 12px 4px; margin: -1rem -1rem 18px -1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }}
    .st-key-airdrip_topnav div[data-testid="stButton"] > button {{
        background: transparent !important; box-shadow: none !important;
        color: {text_secondary} !important; border: 1px solid transparent !important;
        border-radius: 999px !important; font-weight: 600 !important;
        transition: all 0.15s ease !important;
    }}
    .st-key-airdrip_topnav div[data-testid="stButton"] > button:hover {{
        background: {card_bg_soft} !important; color: {text_primary} !important;
        border: 1px solid {card_border} !important;
    }}
</style>
""", unsafe_allow_html=True)

with st.container(key="airdrip_topnav"):
    nav_cols = st.columns(len(NAV_SECTIONS))
    for col, (key, icon, label) in zip(nav_cols, NAV_SECTIONS):
        with col:
            if st.session_state.nav_section == key:
                st.markdown(f'<div class="nav-chip nav-chip-active">{icon}<br>{label}</div>', unsafe_allow_html=True)
            else:
                if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
                    st.session_state.nav_section = key
                    st.rerun()

nav_choice = st.session_state.nav_section

# ---------------------------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------------------------
_fetch_ts = st.session_state.get("last_live_fetch_ts")
_fetch_ts_suffix = f" · updated {_fetch_ts}" if _fetch_ts else ""
if live_data_status.startswith("live"):
    status_badge = f'<span class="badge" style="background:#22c55e;">🟢 LIVE CPCB DATA — {live_data_status}{_fetch_ts_suffix}</span>'
elif use_live_data:
    status_badge = f'<span class="badge" style="background:#eab308; color:#1a1a22;">🟡 SIMULATED — {live_data_status}</span>'
else:
    status_badge = '<span class="badge" style="background:#64748b;">⚪ SIMULATED (demo mode)</span>'

if use_live_data and (cpcb_api_key or waqi_api_token):
    if satellite_status.startswith("live"):
        sat_badge = f'<span class="badge" style="background:#0ea5e9;">🛰️ NO-STATION CITIES — {satellite_status}</span>'
    else:
        sat_badge = f'<span class="badge" style="background:#64748b;">🛰️ NO-STATION CITIES — {satellite_status}</span>'
else:
    sat_badge = ""


st.markdown(f"""
<div class="hero-card">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
        <div>
            <div class="eyebrow">India's Living Air Quality Network</div>
            <h1 class="brand-wordmark" style="margin:2px 0 0 0; font-size:2.6rem;">AirDrip</h1>
            <p style="margin:6px 0 0 0; font-size:0.95rem;">Your city's air, ranked, roasted, and forecasted — satellites + citizens + AI, one platform.</p>
            <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">{status_badge}{sat_badge}</div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:2.6rem; font-weight:800; color:{national_color};">{national_avg_aqi:.0f}</div>
            <div class="eyebrow">National Avg AQI — {national_category}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card"><div class="value" style="color:{worst_row['AQI_color']}">{worst_row['city']}</div>
    <div class="label">Worst Air Today — {worst_row['AQI']:.0f}</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card"><div class="value" style="color:{best_row['AQI_color']}">{best_row['city']}</div>
    <div class="label">Cleanest Air Today — {best_row['AQI']:.0f}</div></div>""", unsafe_allow_html=True)
with c3:
    fire_label = "🔥 Fires Spotted" if LAYMAN else "Active Fire Detections (VIIRS)"
    st.markdown(f"""<div class="metric-card"><div class="value">{total_fires}</div>
    <div class="label">{fire_label}</div></div>""", unsafe_allow_html=True)
with c4:
    hs_label = "⚠️ Hotspot Zones" if LAYMAN else "HCHO Hotspot Zones (P75+)"
    st.markdown(f"""<div class="metric-card"><div class="value">{hcho_hotspot_count}</div>
    <div class="label">{hs_label}</div></div>""", unsafe_allow_html=True)

st.write("")

# ===================== SECTION: TODAY =====================
if nav_choice == "today":
    left, right = st.columns([1.6, 1])

    # --- Right column (state selector + translator) is computed FIRST so its
    # selection can drive the map's pan/zoom built in the left column below.
    with right:
        st.markdown("#### 🫁 What Does This Mean For You?")
        state_options = ["🇮🇳 All India"] + [f"{s} 📍" if s in df["state"].values or
                          (s == "Jammu and Kashmir" and "J&K" in df["state"].values) else s
                          for s in all_state_names()]
        # default to the worst-affected state we actually have station data for
        default_state_label = f"{worst_row['state']} 📍"
        default_idx = state_options.index(default_state_label) if default_state_label in state_options else 0
        state_choice_raw = st.selectbox("Select your state", state_options, index=default_idx, key="today_state")
        state_choice = state_choice_raw.replace(" 📍", "").strip()

        if state_choice == "🇮🇳 All India":
            map_view = default_india_view()
            state_rows = df
            focus_label = "All India"
        else:
            map_view = get_view_for_state(state_choice)
            internal_name = to_display_name(state_choice)  # e.g. "Jammu and Kashmir" -> "J&K"
            state_rows = df[df["state"] == internal_name]
            focus_label = state_choice

        no_direct_station = state_choice != "🇮🇳 All India" and len(state_rows) == 0
        if no_direct_station:
            nearest_row, nearest_dist = nearest_city_for_state(state_choice, df)
            prow = nearest_row
            data_note = f"⚠️ No direct monitoring station in {focus_label} yet — showing nearest regional estimate ({prow['city']}, {nearest_dist:.0f} km away)."
        else:
            prow = state_rows.loc[state_rows["AQI"].idxmax()] if len(state_rows) > 0 else worst_row
            data_note = None

        cat = translate_category(prow["AQI_category"])
        eq = real_life_equivalent(prow["PM2.5"])

        st.markdown(f"""
        <div class="glass" style="padding:18px 20px;">
            <div style="font-size:1rem; font-weight:700; color:{prow['AQI_color']};">{cat['emoji']} {prow['AQI_category']} — AQI {prow['AQI']:.0f}</div>
            <div style="margin-top:6px; font-size:0.9rem;">{cat['line']}</div>
            <hr style="border-color:{card_border}; margin:12px 0;">
            <div style="font-size:1.4rem;">{eq['icon']}</div>
            <div style="margin-top:4px; font-size:0.92rem; font-weight:600;">{eq['text']}</div>
        </div>
        """, unsafe_allow_html=True)
        if data_note:
            st.caption(data_note)
        st.markdown(share_links(f"{focus_label}'s air today = {eq['text']} 😳 Check your state's Drip on AirDrip"), unsafe_allow_html=True)

    with left:
        st.markdown("#### 📍 Live Air Map")
        st.caption("⚠️ Map boundaries are approximate and for pollution-visualization purposes "
                   "only — not an authoritative depiction of India's political boundaries.")
        layer_col_map = {"AQI (composite)": "AQI", "PM2.5": "PM2.5", "NO2": "NO2",
                          "SO2": "SO2", "O3": "O3", "HCHO": "HCHO", "AOD (satellite)": "AOD"}
        layer_col = layer_col_map[pollutant_view]

        # State-level average AQI for the choropleth
        state_avg = df.groupby("state")["AQI"].mean().to_dict()
        india_geo = load_state_geojson()
        geo_state_names = [f["properties"]["st_nm"] for f in india_geo["features"]]
        z_values = []
        for gname in geo_state_names:
            internal = to_display_name(gname)
            z_values.append(state_avg.get(internal, None))

        fig = go.Figure()
        fig.add_trace(go.Choroplethmapbox(
            geojson=india_geo, locations=geo_state_names, featureidkey="properties.st_nm",
            z=z_values, colorscale=[[0, "#009865"], [0.2, "#a3c853"], [0.4, "#ffb302"],
                                     [0.6, "#ff6b02"], [0.8, "#e0301e"], [1.0, "#7e0023"]],
            zmin=0, zmax=500, marker_opacity=0.38, marker_line_width=1,
            marker_line_color="rgba(255,255,255,0.35)", showscale=False,
            hoverinfo="skip", name="State avg AQI",
        ))

        if layer_col == "AQI":
            colorscale = [[0, "#009865"], [0.2, "#a3c853"], [0.4, "#ffb302"], [0.6, "#ff6b02"], [0.8, "#e0301e"], [1.0, "#7e0023"]]
            cmin, cmax = 0, 500
        else:
            colorscale = "YlOrRd"
            cmin, cmax = df[layer_col].min(), df[layer_col].max()

        fig.add_trace(go.Scattermapbox(
            lat=df["lat"], lon=df["lon"], mode="markers",
            marker=dict(size=np.clip(df[layer_col] / df[layer_col].max() * 35 + 8, 8, 40),
                        color=df[layer_col], colorscale=colorscale, cmin=cmin, cmax=cmax,
                        showscale=True, colorbar=dict(title=pollutant_view, tickfont=dict(color=text_secondary))),
            text=df.apply(lambda r: f"<b>{r['city']}, {r['state']}</b><br>AQI: {r['AQI']:.0f} ({r['AQI_category']})<br>"
                                     f"{pollutant_label('PM2.5')}: {r['PM2.5']:.0f} | {pollutant_label('NO2')}: {r['NO2']:.0f}<br>"
                                     f"{pollutant_label('HCHO')}: {r['HCHO']:.1f}", axis=1),
            hoverinfo="text", name="Stations",
        ))
        if show_fire and total_fires > 0:
            fire_df = df[df["fire_count"] > 0]
            fig.add_trace(go.Scattermapbox(
                lat=fire_df["lat"], lon=fire_df["lon"], mode="markers",
                marker=dict(size=fire_df["fire_count"] * 2 + 6, color="orange", opacity=0.5),
                text=fire_df.apply(lambda r: f"🔥 {r['fire_count']} fire detections", axis=1),
                hoverinfo="text", name="Fire activity",
            ))
        map_style = "carto-darkmatter" if THEME == "dark" else "carto-positron"
        fig.update_layout(
            mapbox=dict(style=map_style, zoom=map_view["zoom"],
                        center=dict(lat=map_view["center_lat"], lon=map_view["center_lon"])),
            margin=dict(l=0, r=0, t=0, b=0), height=460, paper_bgcolor=plot_paper,
            showlegend=show_fire, legend=dict(font=dict(color=text_secondary), bgcolor="rgba(0,0,0,0)"),
            transition=dict(duration=500, easing="cubic-in-out"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write("")
    st.markdown("#### 🚶 Can I Step Out Today?")
    q1, q2, q3, q4 = st.columns([1.2, 1, 1, 1])
    with q1:
        ask_city = st.selectbox("City", df["city"].tolist(), key="ask_city")
    with q2:
        ask_profile = st.selectbox("Who's asking?", list(PROFILE_SENSITIVITY.keys()), key="ask_profile")
    with q3:
        ask_activity = st.selectbox("What are you planning?", list(ACTIVITY_MULT.keys()), key="ask_activity")
    with q4:
        st.write("")
        st.write("")
        ask_btn = st.button("Ask AirDrip 🤖", use_container_width=True)

    if ask_btn or "step_out_result" in st.session_state:
        arow = df[df["city"] == ask_city].iloc[0]
        result = step_out_advice(arow["AQI"], arow["AQI_category"], ask_profile, ask_activity)
        st.session_state.step_out_result = result
        st.markdown(f"""
        <div class="glass" style="padding:18px 22px; border-left: 4px solid {result['color']};">
            <div style="font-size:1.15rem; font-weight:800;">{result['icon']} {result['verdict']}</div>
            <div style="margin-top:6px; font-size:0.88rem;">{result['reasoning']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("#### ⏰ Best Time to Go Outside Today")
    bw_city = st.selectbox("Pick city for hourly outlook", df["city"].tolist(), key="bw_city")
    bw_row = df[df["city"] == bw_city].iloc[0]
    curve = diurnal_curve_for(bw_row["PM2.5"] / 55)
    window = best_window_today(curve)
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=list(range(24)), y=curve, fill="tozeroy",
                                    line=dict(color="#38bdf8", width=2), fillcolor="rgba(56,189,248,0.15)"))
    fig_curve.add_vrect(x0=window["start_hour"], x1=window["start_hour"] + 2,
                         fillcolor="rgba(34,197,94,0.25)", line_width=0,
                         annotation_text="✅ Best window", annotation_font_color=text_primary)
    fig_curve.update_layout(**CHART_LAYOUT, height=280, margin=dict(t=20, b=20),
                             xaxis_title="Hour of day", yaxis_title="Relative pollution level", showlegend=False)
    st.plotly_chart(fig_curve, use_container_width=True)
    st.caption(f"💡 Best 2-hour window today in **{bw_city}**: **{window['label']}** — pollution is typically lowest then (post-morning-mixing, pre-evening-inversion).")

# ===================== SECTION: FORECAST =====================
elif nav_choice == "forecast":
    st.markdown("#### 🔮 30-Day AI Forecast — Where Is Air Quality Heading?")
    st.caption("Trend + seasonal decomposition: today's level, drift toward next month's climatology, "
               "plus a stubble-burning-season kicker for belt cities. Uncertainty widens with lead time.")

    fc1, fc2 = st.columns([1, 3])
    with fc1:
        forecast_city_name = st.selectbox("City", df["city"].tolist(),
                                           index=int(df["city"].tolist().index(worst_row["city"])), key="fc_city")
        horizon = st.slider("Forecast horizon (days)", 7, 30, 30)

    city_row = df[df["city"] == forecast_city_name].iloc[0].to_dict()
    fc_df = forecast_city(city_row, current_month=sel_month, current_day=sel_day, horizon_days=horizon)
    summary = forecast_summary(fc_df, city_row["PM2.5"])

    with fc2:
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(f"""<div class="metric-card"><div class="value">{summary['trend_icon']} {summary['trend_label']}</div>
            <div class="label">30-Day Trend ({summary['delta_pct']:+.1f}%)</div></div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""<div class="metric-card"><div class="value">{summary['peak_value']:.0f}</div>
            <div class="label">Peak {pollutant_label('PM2.5')} — {summary['peak_date'].strftime('%d %b')}</div></div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""<div class="metric-card"><div class="value" style="font-size:1.3rem;">{summary['dominant_driver']}</div>
            <div class="label">Primary Driver</div></div>""", unsafe_allow_html=True)

    st.write("")
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=list(fc_df["date"]) + list(fc_df["date"][::-1]),
                                 y=list(fc_df["upper_bound"]) + list(fc_df["lower_bound"][::-1]),
                                 fill="toself", fillcolor="rgba(56,189,248,0.14)", line=dict(color="rgba(0,0,0,0)"),
                                 name="Confidence band", hoverinfo="skip"))
    fig_fc.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["forecast_PM2.5"], name="Forecast", line=dict(color="#38bdf8", width=3)))
    fig_fc.add_hline(y=city_row["PM2.5"], line_dash="dot", line_color="#f97316",
                      annotation_text="Today", annotation_font_color="#f97316")
    fig_fc.update_layout(**CHART_LAYOUT, title=f"{forecast_city_name}: {horizon}-Day Forecast", height=420,
                          legend=dict(font=dict(color=text_secondary), bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown("##### National Outlook — Cities Trending Worst Next Month")
    outlook_rows = []
    for _, r in df.iterrows():
        fc_r = forecast_city(r.to_dict(), current_month=sel_month, current_day=sel_day, horizon_days=30)
        s = forecast_summary(fc_r, r["PM2.5"])
        outlook_rows.append({"city": r["city"], "state": r["state"], "current_PM2.5": r["PM2.5"],
                              "30d_peak_PM2.5": s["peak_value"], "change_%": s["delta_pct"], "driver": s["dominant_driver"]})
    outlook_df = pd.DataFrame(outlook_rows).sort_values("change_%", ascending=False)
    st.dataframe(outlook_df.head(10), hide_index=True, use_container_width=True)
    st.markdown('<div class="disclaimer-box">Explainable trend + seasonal decomposition, fully auditable. '
                "Production upgrade: Transformer/LSTM seq2seq trained on multi-year history — same interface.</div>",
                unsafe_allow_html=True)

# ===================== SECTION: COMMUNITY =====================
elif nav_choice == "community":
    st.markdown("#### 💬 AirDrip Community — Crowdsourced Ground Truth")
    st.caption("Satellites & CPCB stations can't be everywhere. Citizens report what they're actually "
               "breathing — filling monitoring gaps and independently verifying the model's estimates.")
    st.markdown('<div class="disclaimer-box">🧪 <b>Demo data below</b>: the feed is seeded with '
                'illustrative sample posts (marked "Demo") to show how the feature works. Only posts '
                'you or other real visitors submit through the form below are real.</div>',
                unsafe_allow_html=True)

    if "seed_posts_cache" not in st.session_state or st.session_state.get("seed_key") != (sel_month, sel_day):
        st.session_state.seed_posts_cache = _seed_posts(df, sel_month, sel_day, n=18)
        st.session_state.seed_key = (sel_month, sel_day)

    with st.expander("✍️ Share what you're experiencing right now", expanded=False):
        with st.form("new_post_form", clear_on_submit=True):
            pc1, pc2 = st.columns(2)
            with pc1:
                post_city = st.selectbox("Your city", df["city"].tolist(), key="post_city")
            with pc2:
                post_tag = st.selectbox("How's the air?", SYMPTOM_TAGS, key="post_tag")
            post_text = st.text_area("Add details (optional)", placeholder="e.g. Haze near the market since morning...")
            submitted = st.form_submit_button("Post to AirDrip")
            if submitted:
                new_post = {
                    "id": f"user_{datetime.now().timestamp()}", "handle": "@you", "city": post_city,
                    "text": post_text if post_text else f"Reporting: {post_tag}", "tag": post_tag,
                    "visibility_m": None, "timestamp": datetime.now(), "likes": 0,
                    "verified_ground_truth": False, "is_user_post": True,
                }
                add_shared_post(new_post)
                st.success("Posted! Everyone on AirDrip can now see this 🙌")

    shared_posts = load_shared_posts()
    all_posts = shared_posts + st.session_state.seed_posts_cache
    posts_df = pd.DataFrame(all_posts)

    st.markdown("##### 📱 Swipe through recent reports")
    swipe_html = '<div class="swipe-row">'
    for p in all_posts[:12]:
        verified = '<span class="verified-chip">✓ Confirmed</span>' if p["verified_ground_truth"] else ""
        demo_chip = '' if p["is_user_post"] else '<span class="verified-chip" style="background:rgba(148,163,184,0.18);color:#94a3b8;">Demo</span>'
        swipe_html += f"""<div class="swipe-card post-card">
            <span class="handle">{p['handle']}</span> · <span class="meta">{p['city']}</span>{demo_chip}{verified}
            <div style="margin-top:6px; font-size:0.85rem;">{p['text'][:110]}</div>
            <div class="tag-chip">{p['tag']}</div>
            <div class="meta" style="margin-top:6px;">❤️ {p['likes']}</div>
        </div>"""
    swipe_html += "</div>"
    st.markdown(swipe_html, unsafe_allow_html=True)

    feed_col, side_col = st.columns([1.7, 1])
    with feed_col:
        st.markdown("##### Full Feed")
        filter_city = st.selectbox("Filter by city", ["All cities"] + sorted(df["city"].tolist()), key="feed_filter")
        shown = all_posts if filter_city == "All cities" else [p for p in all_posts if p["city"] == filter_city]
        for p in shown[:15]:
            verified_html = '<span class="verified-chip">✓ Ground-truth match</span>' if p["verified_ground_truth"] else ""
            you_html = '<span class="verified-chip" style="background:rgba(56,189,248,0.18);color:#38bdf8;">You</span>' if p["is_user_post"] else \
                       '<span class="verified-chip" style="background:rgba(148,163,184,0.18);color:#94a3b8;">Demo</span>'
            st.markdown(f"""<div class="post-card">
                <span class="handle">{p['handle']}</span> · <span class="meta">{p['city']} · {p['timestamp'].strftime('%d %b, %H:%M')}</span>
                {you_html}{verified_html}
                <div style="margin-top:6px;">{p['text']}</div>
                <div class="tag-chip">{p['tag']}</div>
                <div class="meta" style="margin-top:6px;">❤️ {p['likes']} · 💬 reply</div>
            </div>""", unsafe_allow_html=True)

    with side_col:
        st.markdown("##### Community Confidence Signal")
        conf_df = confidence_boost(df, posts_df)
        confirmed = conf_df[conf_df["community_confirmed"]].sort_values("AQI", ascending=False)
        st.metric("Ground-truth confirmed hotspots", len(confirmed))
        if len(confirmed) > 0:
            st.dataframe(confirmed[["city", "AQI", "negative_report_ratio"]].rename(
                columns={"negative_report_ratio": "% negative reports"}), hide_index=True, use_container_width=True)
        st.markdown('<div class="disclaimer-box">Flagged "confirmed" when ≥60% of recent reports are '
                    "negative AND model AQI ≥200 — independent corroboration.</div>", unsafe_allow_html=True)
        st.markdown("##### Top Voices")
        for p in sorted(all_posts, key=lambda p: p["likes"], reverse=True)[:5]:
            st.markdown(f"**{p['handle']}** · ❤️ {p['likes']} — *{p['city']}*")

# ===================== SECTION: LEADERBOARD =====================
elif nav_choice == "leaderboard":
    st.markdown("#### 🏆 Clean Air Leaderboard")
    st.caption("Check your city's air the way you'd check a cricket score. Streaks and badges make "
               "clean air something to root for, not just a number.")

    lb = build_leaderboard(df, sel_month, sel_day)
    lb1, lb2 = st.columns([1.6, 1])
    with lb1:
        st.markdown("##### 🥇 Cleanest Cities")
        top_clean = lb.head(8)
        for _, r in top_clean.iterrows():
            streak_txt = f"🔥 {r['clean_streak_days']}-day clean streak" if r["clean_streak_days"] > 0 else ""
            st.markdown(f"""<div class="glass" style="padding:12px 18px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center;">
                <div><b>#{r['rank']} {r['city']}</b>, {r['state']} — {r['badge']}</div>
                <div style="text-align:right;"><b style="color:#22c55e;">{r['AQI']:.0f}</b><br><span class="meta" style="font-size:0.72rem;">{streak_txt}</span></div>
            </div>""", unsafe_allow_html=True)

    with lb2:
        st.markdown("##### 🚨 Needs Attention")
        worst5 = lb.tail(5).iloc[::-1]
        for _, r in worst5.iterrows():
            st.markdown(f"""<div class="glass" style="padding:12px 18px; margin-bottom:8px;">
                <b>{r['city']}</b>, {r['state']} — <span style="color:#e0301e; font-weight:700;">{r['AQI']:.0f}</span><br>
                <span class="meta">{r['badge']}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("##### 📈 Most Improving (30-day forecast)")
    outlook_rows2 = []
    for _, r in df.iterrows():
        fc_r = forecast_city(r.to_dict(), current_month=sel_month, current_day=sel_day, horizon_days=30)
        s = forecast_summary(fc_r, r["PM2.5"])
        outlook_rows2.append({"city": r["city"], "change_%": s["delta_pct"]})
    outlook_df2 = pd.DataFrame(outlook_rows2)
    improving = improving_cities(df, outlook_df2, top_n=5)
    st.dataframe(improving, hide_index=True, use_container_width=True)

# ===================== SECTION: HOTSPOTS & BURNING =====================
elif nav_choice == "hotspots":
    st.markdown(f"#### 🔥 {pollutant_label('HCHO')} Hotspot Detection")
    st.caption("Satellite-derived formaldehyde column density — a proxy for biomass burning, "
               "traffic, and biogenic emissions.")

    hotspots, threshold = cluster_hotspots(df, "HCHO", threshold_percentile=75)
    hcol1, hcol2 = st.columns([2.4, 1])
    with hcol1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scattermapbox(
            lat=df["lat"], lon=df["lon"], mode="markers",
            marker=dict(size=np.clip(df["HCHO"] * 2.2, 8, 45), color=df["HCHO"], colorscale="Hot",
                        cmin=0, cmax=df["HCHO"].max(), showscale=True,
                        colorbar=dict(title="HCHO", tickfont=dict(color=text_secondary))),
            text=df.apply(lambda r: f"<b>{r['city']}</b><br>HCHO: {r['HCHO']:.1f}<br>Tier: {r['HCHO_tier']}<br>Fires: {r['fire_count']}", axis=1),
            hoverinfo="text", name="HCHO column",
        ))
        fig2.add_trace(go.Scattermapbox(
            lat=hotspots["lat"], lon=hotspots["lon"], mode="markers",
            marker=dict(size=hotspots["hotspot_rank"].apply(lambda r: max(38 - r, 20)), color="rgba(255,255,255,0)"),
            text=hotspots.apply(lambda r: f"⚠️ HOTSPOT #{r['hotspot_rank']}: {r['city']}", axis=1),
            hoverinfo="text", name=f"Detected hotspots (HCHO≥{threshold:.1f})",
        ))
        map_style2 = "carto-darkmatter" if THEME == "dark" else "carto-positron"
        fig2.update_layout(mapbox=dict(style=map_style2, zoom=3.5, center=dict(lat=22.5, lon=80)),
                            margin=dict(l=0, r=0, t=0, b=0), height=480, paper_bgcolor=plot_paper,
                            legend=dict(font=dict(color=text_secondary), bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig2, use_container_width=True)
    with hcol2:
        st.markdown(f"**Threshold (P75):** {threshold:.2f} | **Hotspots:** {len(hotspots)}")
        st.dataframe(hotspots[["city", "state", "HCHO", "HCHO_tier", "fire_count"]].reset_index(drop=True),
                     hide_index=True, use_container_width=True, height=340)

    st.markdown("---")
    st.markdown("#### 🌾 Biomass Burning Impact")
    belt_df = df[df["in_burning_belt"]]
    other_df = df[~df["in_burning_belt"]]
    bc1, bc2 = st.columns(2)
    with bc1:
        fig3 = px.scatter(df, x="fire_count", y="HCHO", color="in_burning_belt", size="PM2.5", hover_name="city",
                           color_discrete_map={True: "#f97316", False: "#38bdf8"}, title="Fire Count vs HCHO")
        fig3.update_layout(**CHART_LAYOUT, legend=dict(font=dict(color=text_secondary)))
        st.plotly_chart(fig3, use_container_width=True)
        st.metric("Fire ↔ HCHO correlation (r)", f"{df['fire_count'].corr(df['HCHO']):.2f}")
    with bc2:
        comparison = pd.DataFrame({"Zone": ["Burning belt", "Rest of India"],
                                    "Avg HCHO": [belt_df["HCHO"].mean(), other_df["HCHO"].mean()],
                                    "Avg PM2.5": [belt_df["PM2.5"].mean(), other_df["PM2.5"].mean()],
                                    "Avg Fires": [belt_df["fire_count"].mean(), other_df["fire_count"].mean()]})
        fig4 = px.bar(comparison.melt(id_vars="Zone", var_name="Metric", value_name="Value"),
                      x="Metric", y="Value", color="Zone", barmode="group",
                      color_discrete_map={"Burning belt": "#f97316", "Rest of India": "#38bdf8"})
        fig4.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig4, use_container_width=True)
    st.info("🌬️ Under NW winds (Nov), plumes from Punjab/Haryana transport SE into NCR — the well-documented "
            "mechanism behind Delhi's Nov spikes. Production: ERA5/IMDAA wind fields + HYSPLIT back-trajectories.")

# ===================== SECTION: POLICY COPILOT =====================
elif nav_choice == "policy":
    st.markdown("#### 📜 Policy Copilot — Automated GRAP Stage Advisory")
    st.caption("Real CAQM GRAP thresholds + mandated actions, combined with the 30-day forecast and "
               "community confirmation into one auto-brief for policymakers.")

    pc1, pc2 = st.columns([1, 2])
    with pc1:
        policy_city = st.selectbox("Region", df["city"].tolist(),
                                    index=int(df["city"].tolist().index(worst_row["city"])), key="policy_city")

    prow2 = df[df["city"] == policy_city].iloc[0]
    p_fc = forecast_city(prow2.to_dict(), current_month=sel_month, current_day=sel_day, horizon_days=30)
    p_summary = forecast_summary(p_fc, prow2["PM2.5"])
    posts_df_p = pd.DataFrame(load_shared_posts() + st.session_state.get("seed_posts_cache", []))
    conf_df_p = confidence_boost(df, posts_df_p)
    is_confirmed = bool(conf_df_p[conf_df_p["city"] == policy_city]["community_confirmed"].any())

    brief = generate_policy_brief(city=policy_city, current_aqi=prow2["AQI"],
        forecast_peak_aqi=(p_summary["peak_value"] / prow2["PM2.5"]) * prow2["AQI"] if prow2["PM2.5"] > 0 else prow2["AQI"],
        dominant_pollutant=prow2["dominant_pollutant"], forecast_driver=p_summary["dominant_driver"],
        community_confirmed=is_confirmed)

    stage_to_show = brief["forecast_stage"] if brief["escalation"] else brief["current_stage"]
    with pc2:
        if stage_to_show:
            st.markdown(f'<span class="grap-badge" style="background:{stage_to_show["color"]};">{stage_to_show["stage"]} — {stage_to_show["label"]}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="grap-badge" style="background:#16a34a;">No GRAP stage active</span>', unsafe_allow_html=True)

    st.write("")
    st.markdown(f'<div class="glass" style="padding:22px 26px;">{brief["brief_text"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

    st.write("")
    stage_cols = st.columns(4)
    for i, stage in enumerate(GRAP_STAGES):
        with stage_cols[i]:
            st.markdown(f"""<div class="glass" style="padding:14px; border-top: 3px solid {stage['color']};">
                <b>{stage['stage']}</b><br><span style="color:{stage['color']}; font-weight:700;">{stage['label']}</span>
                <div class="meta" style="font-size:0.75rem; margin-top:4px;">AQI {stage['aqi_range'][0]}–{stage['aqi_range'][1]}</div>
            </div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown("##### All-India GRAP Escalation Watchlist")
    watchlist = []
    for _, r in df.iterrows():
        r_fc = forecast_city(r.to_dict(), current_month=sel_month, current_day=sel_day, horizon_days=30)
        r_summary = forecast_summary(r_fc, r["PM2.5"])
        forecast_aqi_proxy = (r_summary["peak_value"] / r["PM2.5"]) * r["AQI"] if r["PM2.5"] > 0 else r["AQI"]
        cur_stage = get_grap_stage(r["AQI"])
        fut_stage = get_grap_stage(forecast_aqi_proxy)
        if fut_stage is not None and (cur_stage is None or GRAP_STAGES.index(fut_stage) > GRAP_STAGES.index(cur_stage)):
            watchlist.append({"city": r["city"], "current_stage": cur_stage["stage"] if cur_stage else "None",
                               "forecast_stage": fut_stage["stage"], "current_AQI": r["AQI"]})
    if watchlist:
        st.dataframe(pd.DataFrame(watchlist), hide_index=True, use_container_width=True)
    else:
        st.info("No regions currently forecast to escalate a GRAP stage in the next 30 days.")

# ===================== SECTION: MODEL & TECH =====================
elif nav_choice == "model":
    st.markdown("#### 🧠 CNN-LSTM Surface Estimation Model")
    mc1, mc2 = st.columns([1, 1.3])
    with mc1:
        st.markdown("##### Reported Validation Metrics (target architecture)")
        for pollutant, m in REPORTED_METRICS.items():
            st.markdown(f"""<div class="metric-card" style="margin-bottom:10px;">
                <div class="value">{m['R2']:.2f}</div>
                <div class="label">{pollutant} R² (RMSE {m['RMSE']} {m['unit']})</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('<div class="disclaimer-box">Representative targets from comparable AOD→PM2.5 '
                    "downscaling literature over the IGP — not measured on real held-out data in this "
                    "sandbox. Retrain on real INSAT-3D/TROPOMI + CPCB pairs for your own numbers.</div>",
                    unsafe_allow_html=True)
    with mc2:
        sample_city = st.selectbox("Preview time series for:", df["city"].tolist(), index=0, key="model_city")
        row = df[df["city"] == sample_city].iloc[0]
        ts = generate_time_series(row, days=60, end_month=sel_month, end_day=sel_day if sel_day > 5 else 15)
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=ts["date"], y=ts["actual_PM2.5"], name="Actual (CPCB)", line=dict(color="#38bdf8", width=2)))
        fig5.add_trace(go.Scatter(x=ts["date"], y=ts["predicted_PM2.5"], name="CNN-LSTM predicted", line=dict(color="#f97316", width=2, dash="dot")))
        fig5.update_layout(**CHART_LAYOUT, title=f"PM2.5: Actual vs Predicted — {sample_city}",
                            legend=dict(font=dict(color=text_secondary)), height=340)
        st.plotly_chart(fig5, use_container_width=True)
    st.markdown("##### Architecture")
    st.code(MODEL_ARCHITECTURE_SUMMARY, language="text")

# ===================== SECTION: ABOUT =====================
elif nav_choice == "about":
    st.markdown(f"""
##### Data Sources
| Layer | Source | Role |
|---|---|---|
| Ground pollutant readings (32 cities) | CPCB, via WAQI (primary) + data.gov.in (fallback) | Live PM2.5/PM10/NO2/SO2/CO/O3 where a government station exists |
| No-station cities (9 cities) | Open-Meteo Air Quality API (Copernicus CAMS) | Real satellite-informed atmospheric-model estimate where no CPCB/WAQI station exists |
| Aerosol Optical Depth, fire activity, meteorology | Simulated | Illustrative stand-in for INSAT-3D/Sentinel-5P/MODIS — no live credentials wired up for these layers yet |
| Community reports | Demo-seeded + real user submissions | Crowd-sourced corroboration signal |

##### What makes AirDrip different
1. **Two-tier real data** — live government ground stations where they exist, a real satellite-informed model where they don't, instead of leaving gaps unfilled or guessing
2. **AirDrip Community** — fills the ground-monitoring gap the problem statement itself names, with crowd reports that independently corroborate the model's estimates
3. **30-Day AI Forecast** — turns a monitoring tool into a planning tool
4. **Policy Copilot** — auto-generates GRAP (CAQM) stage triggers and actions
5. **AQI-to-Real-Life Translator, Step-Out Assistant, Best-Window Alert, Clean Air Leaderboard** — make air quality legible and habit-forming for people who've never heard of PM2.5, not just for scientists

##### What's real vs. simulated — read this before presenting
- ✅ **Real, live, verified working**: CPCB ground station data for 32 major cities (via WAQI, with a CPCB-direct fallback); satellite-model estimates (Open-Meteo/CAMS) for 9 cities with no ground station; CPCB's official AQI formula; GRAP stage definitions (CAQM)
- 🔶 **Simulated for demo**: aerosol optical depth, fire-detection counts, meteorology (wind/BLH/humidity/temp), and HCHO — these need TROPOMI/FIRMS/GEE credentials this build doesn't have; the CNN-LSTM in the Model tab is an architecture proposal with a physics-based stand-in, not a trained model; Community feed is demo-seeded unless you or another visitor posts
- 🔜 **Honest next step**: wire up TROPOMI/FIRMS/GEE for the remaining simulated layers, and train the CNN-LSTM on real historical satellite+CPCB pairs once available
""")
