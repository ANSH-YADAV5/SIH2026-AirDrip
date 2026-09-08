"""
Layman Translation Layer — turns technical air-quality jargon into plain,
relatable language for people who've never heard of PM2.5 or HCHO.

Also powers the "AQI-to-Real-Life" translator (the flagship USP): converts
an AQI/PM2.5 value into a vivid, instantly-understandable real-world
equivalent (cigarettes, bonfire exposure, etc.) using published exposure
equivalence research as the basis for the conversion factor.
"""

import numpy as np

# Plain-language pollutant glossary
POLLUTANT_LAYMAN = {
    "PM2.5": {"emoji": "🫁", "simple_name": "Lung Dust", "desc": "Tiny particles small enough to get deep into your lungs and blood."},
    "PM10":  {"emoji": "🌫️", "simple_name": "Dust & Smoke", "desc": "Bigger dust/smoke particles — irritate your nose, throat, eyes."},
    "NO2":   {"emoji": "🚗", "simple_name": "Traffic Gas", "desc": "Mainly from vehicle exhaust and fuel burning."},
    "SO2":   {"emoji": "🏭", "simple_name": "Factory Gas", "desc": "Mostly from industry and power plants burning fuel."},
    "CO":    {"emoji": "🔥", "simple_name": "Silent Gas", "desc": "Colorless, odorless gas from incomplete combustion — reduces oxygen in blood."},
    "O3":    {"emoji": "☀️", "simple_name": "Ground Smog", "desc": "Forms when sunlight reacts with traffic/industrial gases — worse on hot sunny days."},
    "HCHO":  {"emoji": "🌾", "simple_name": "Burning Signal", "desc": "Released by crop/forest burning and traffic — an early warning sign of fires nearby."},
    "AOD":   {"emoji": "🛰️", "simple_name": "Sky Haze Level", "desc": "How hazy the whole column of sky looks from space — satellites use this to estimate ground pollution."},
}

AQI_LAYMAN = {
    "Good":         {"emoji": "😄", "line": "Air's clean — breathe easy, get outside!"},
    "Satisfactory": {"emoji": "🙂", "line": "Air's decent. Most people are totally fine outdoors."},
    "Moderate":     {"emoji": "😐", "line": "Okay for most, but sensitive folks (asthma, elderly, kids) should take it easy outside."},
    "Poor":         {"emoji": "😷", "line": "Air's rough. Consider a mask outdoors, especially for longer activity."},
    "Very Poor":    {"emoji": "🥴", "line": "Pretty bad. Cut down outdoor time, especially exercise."},
    "Severe":       {"emoji": "🚨", "line": "Hazardous. Stay indoors if you can, especially kids & elderly."},
}


def translate_pollutant(code: str) -> dict:
    return POLLUTANT_LAYMAN.get(code, {"emoji": "❓", "simple_name": code, "desc": "Air pollutant."})


def translate_category(category: str) -> dict:
    return AQI_LAYMAN.get(category, {"emoji": "❓", "line": ""})


# ---------------------------------------------------------------------------
# AQI-to-Real-Life equivalence translator
# ---------------------------------------------------------------------------
# Cigarette equivalence based on the widely-cited Berkeley Earth methodology:
# ~22 ug/m3 of sustained PM2.5 exposure over 24h ≈ 1 cigarette smoked.
CIGARETTE_PM25_FACTOR = 22.0


def cigarette_equivalent(pm25: float) -> float:
    return round(pm25 / CIGARETTE_PM25_FACTOR, 1)


REAL_LIFE_EQUIVALENTS = [
    # (min_pm25, max_pm25, template, icon)
    (0, 30, "Fresh mountain air — {cigs} cigarettes/day equivalent. Basically nothing.", "🏔️"),
    (30, 60, "Like being in a slightly stuffy room — {cigs} cigarettes/day equivalent.", "🚪"),
    (60, 120, "Like sitting in light traffic smoke for the whole day — {cigs} cigarettes/day equivalent.", "🚦"),
    (120, 250, "Like standing near a bonfire for 2 hours — {cigs} cigarettes/day equivalent.", "🔥"),
    (250, 400, "Like smoking {cigs} cigarettes today, just by breathing.", "🚬"),
    (400, 1000, "Like being inside a smoky kitchen with no ventilation all day — {cigs} cigarettes/day equivalent.", "🏠"),
]


def real_life_equivalent(pm25: float) -> dict:
    cigs = cigarette_equivalent(pm25)
    for lo, hi, template, icon in REAL_LIFE_EQUIVALENTS:
        if lo <= pm25 < hi:
            return {"text": template.format(cigs=cigs), "icon": icon, "cigarettes": cigs}
    lo, hi, template, icon = REAL_LIFE_EQUIVALENTS[-1]
    return {"text": template.format(cigs=cigs), "icon": icon, "cigarettes": cigs}


# ---------------------------------------------------------------------------
# "Best Window" advisor — best 2-hour outdoor window based on forecast shape
# ---------------------------------------------------------------------------

def best_window_today(hourly_like_curve: np.ndarray, hours=None) -> dict:
    """
    hourly_like_curve: array of 24 relative pollution multipliers (synthetic
    diurnal curve — pollution typically lowest early afternoon post-mixing,
    highest early morning + evening rush + night inversion).
    """
    if hours is None:
        hours = np.arange(24)
    # find best contiguous 2-hour window (lowest average)
    best_start, best_val = 0, np.inf
    for i in range(23):
        avg = (hourly_like_curve[i] + hourly_like_curve[(i + 1) % 24]) / 2
        if avg < best_val:
            best_val = avg
            best_start = i
    fmt = lambda h: f"{h%12 or 12}{'AM' if h < 12 else 'PM'}"
    return {"start_hour": best_start, "end_hour": (best_start + 2) % 24,
            "label": f"{fmt(best_start)}–{fmt((best_start + 2) % 24)}", "relative_level": round(best_val, 2)}


def diurnal_curve_for(base_multiplier: float) -> np.ndarray:
    """Synthetic realistic daily pollution curve: morning rush peak, midday dip
    (better mixing/sun), evening rush + night inversion peak."""
    hours = np.arange(24)
    curve = (
        0.55
        + 0.35 * np.exp(-((hours - 8) ** 2) / 8)     # morning rush peak ~8am
        + 0.30 * np.exp(-((hours - 21) ** 2) / 10)   # evening/night peak ~9pm
        - 0.20 * np.exp(-((hours - 14) ** 2) / 12)   # midday dip ~2pm
    )
    return np.clip(curve, 0.3, 1.6) * base_multiplier


# ---------------------------------------------------------------------------
# Personalized risk advice for "Can I Step Out?" assistant
# ---------------------------------------------------------------------------

PROFILE_SENSITIVITY = {
    "General / healthy adult": 1.0,
    "Child (under 12)": 0.7,
    "Elderly (60+)": 0.65,
    "Pregnant": 0.7,
    "Asthma / respiratory condition": 0.5,
    "Athlete / heavy exercise plans": 0.6,
}

ACTIVITY_MULT = {
    "Just stepping out / errands": 1.0,
    "Walking / light activity": 0.85,
    "Running / cycling / sports": 0.6,
}


def step_out_advice(aqi: float, category: str, profile: str, activity: str) -> dict:
    sensitivity = PROFILE_SENSITIVITY.get(profile, 1.0)
    activity_factor = ACTIVITY_MULT.get(activity, 1.0)
    effective_risk = aqi / (sensitivity * (1 / (2 - activity_factor)))

    if effective_risk < 100:
        verdict, color, icon = "Yes, go ahead!", "#22c55e", "✅"
    elif effective_risk < 200:
        verdict, color, icon = "Should be okay, just don't overdo it.", "#a3c853", "🙂"
    elif effective_risk < 300:
        verdict, color, icon = "Maybe — keep it short, consider a mask.", "#ffb302", "⚠️"
    elif effective_risk < 450:
        verdict, color, icon = "Better not to, honestly.", "#ff6b02", "🚫"
    else:
        verdict, color, icon = "Please stay indoors right now.", "#e0301e", "🚨"

    reasons = []
    if profile != "General / healthy adult":
        reasons.append(f"you selected '{profile}', which is more sensitive to pollution")
    if activity != "Just stepping out / errands":
        reasons.append(f"'{activity}' means deeper, faster breathing — you inhale more pollutants")
    reasons.append(f"current air quality is '{category}' (AQI {aqi:.0f})")

    return {"verdict": verdict, "color": color, "icon": icon,
            "reasoning": "Because " + " and ".join(reasons) + "."}
