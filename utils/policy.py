"""
Policy Copilot — automated GRAP (Graded Response Action Plan) stage
recommendation engine.

GRAP is the REAL framework used by India's Commission for Air Quality
Management (CAQM) in NCR, legally tying specific enforcement actions to AQI
bands. This module encodes the actual public GRAP stage definitions and
their associated actions, then auto-generates a policy brief given current
+ forecasted AQI — directly serving the problem statement's call to
"support policymakers in air quality management under NCAP."
"""

import pandas as pd

# Real GRAP stage definitions (CAQM, revised 2023) — AQI bands + category
GRAP_STAGES = [
    {
        "stage": "Stage I", "label": "Poor", "aqi_range": (201, 300), "color": "#ff6b02",
        "actions": [
            "Mechanized road sweeping & water sprinkling on major roads",
            "Strict enforcement against garbage burning",
            "Close monitoring of PUC (Pollution Under Control) compliance for vehicles",
            "Ensure smooth traffic flow to reduce congestion-linked idling emissions",
        ],
    },
    {
        "stage": "Stage II", "label": "Very Poor", "aqi_range": (301, 400), "color": "#e0301e",
        "actions": [
            "Intensify public transport (increase bus/metro frequency, differential parking fees)",
            "Ban on diesel generator sets (except essential/emergency services)",
            "Stop use of coal/firewood in hotels, restaurants, open eateries (tandoors)",
            "Enhanced mechanized sweeping/sprinkling on identified hotspot roads",
        ],
    },
    {
        "stage": "Stage III", "label": "Severe", "aqi_range": (401, 450), "color": "#7e0023",
        "actions": [
            "Ban on non-essential construction & demolition activity",
            "Ban on BS-III petrol / BS-IV diesel four-wheelers (NCR)",
            "Stone crushers and mining operations suspended",
            "Consider closing primary schools; shift to hybrid/online where possible",
        ],
    },
    {
        "stage": "Stage IV", "label": "Severe+", "aqi_range": (451, 500), "color": "#4c0519",
        "actions": [
            "Ban entry of trucks into Delhi (except essential goods/services)",
            "Stop all construction & demolition activity, including highways/pipelines",
            "State governments to consider odd-even vehicle rationing and further school closures",
            "Work-from-home advisory for govt. and private establishments where feasible",
        ],
    },
]


def get_grap_stage(aqi_value: float):
    if aqi_value < 201:
        return None
    for stage in GRAP_STAGES:
        lo, hi = stage["aqi_range"]
        if lo <= aqi_value <= hi:
            return stage
    return GRAP_STAGES[-1]  # cap at Stage IV for anything above 500-equivalent


def generate_policy_brief(city: str, current_aqi: float, forecast_peak_aqi: float,
                           dominant_pollutant: str, forecast_driver: str,
                           community_confirmed: bool = False) -> dict:
    """Auto-generates a structured, decision-ready policy brief for a region."""
    current_stage = get_grap_stage(current_aqi)
    forecast_stage = get_grap_stage(forecast_peak_aqi)

    escalation = (
        forecast_stage is not None
        and (current_stage is None or GRAP_STAGES.index(forecast_stage) > GRAP_STAGES.index(current_stage))
    )

    lines = []
    lines.append(f"**Region:** {city}")
    lines.append(f"**Current AQI:** {current_aqi:.0f} — dominant pollutant: {dominant_pollutant}")
    if current_stage:
        lines.append(f"**Current GRAP status:** {current_stage['stage']} ({current_stage['label']}) — ACTIVE")
    else:
        lines.append("**Current GRAP status:** Below Stage I threshold — no mandatory action")

    lines.append(f"**30-day forecast peak:** {forecast_peak_aqi:.0f} (driver: {forecast_driver})")

    if escalation:
        lines.append(
            f"⚠️ **Recommendation: PRE-EMPTIVE ESCALATION to {forecast_stage['stage']}** "
            f"advised within the forecast window, ahead of the AQI actually crossing the threshold — "
            f"GRAP is designed to trigger proactively on forecast, not reactively on breach."
        )

    if community_confirmed:
        lines.append(
            "✅ **Ground-truth confidence: HIGH** — citizen reports from AirPulse Community "
            "corroborate the satellite/model estimate for this region (independent verification signal)."
        )

    action_stage = forecast_stage if escalation else current_stage
    if action_stage:
        lines.append(f"\n**Mandated/recommended actions ({action_stage['stage']}):**")
        for a in action_stage["actions"]:
            lines.append(f"- {a}")

    return {
        "brief_text": "\n".join(lines),
        "current_stage": current_stage,
        "forecast_stage": forecast_stage,
        "escalation": escalation,
    }
