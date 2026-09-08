"""
AirPulse Community — crowdsourced ground-truthing social layer.

Directly addresses the problem statement's stated gap: "most regions lack
sufficient ground-based monitoring stations, making it difficult to assess
real-time air quality." Satellites + CPCB stations give top-down estimates;
this layer lets citizens report what they're actually experiencing
(symptoms, visibility, smell) the way Waze crowdsources traffic — filling
spatial/temporal gaps between fixed stations and giving the model a
human-verified confidence signal.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SYMPTOM_TAGS = [
    "😤 Throat irritation", "👁️ Eye burning", "😮‍💨 Breathlessness",
    "🌫️ Haze/low visibility", "🤧 Coughing", "👃 Smoky smell",
    "✅ Feels clear today", "🏃 Good for outdoor activity",
]

SAMPLE_HANDLES = [
    "@aditi_delhi", "@rohan.ncr", "@priya_lko", "@stubble_watch_punjab",
    "@kanpur_resident", "@clean_air_patna", "@breathe_easy_mum", "@bengaluru_air",
    "@amritsar_farmer_voice", "@noida_parent", "@varanasi_ghats", "@chd_cyclist",
]

SAMPLE_TEMPLATES = [
    "Visibility down to ~{vis}m near {landmark} this morning. {symptom}",
    "Been feeling {symptom_lower} since yesterday evening in {area}. Anyone else?",
    "{symptom} — third day in a row here in {area}. Kids kept indoors from school today.",
    "Smoke haze rolling in from the fields near {area}, HCHO smell is strong tonight.",
    "AQI app says moderate but it genuinely feels worse outside near {landmark}. {symptom}",
    "Morning walk cancelled again — {symptom_lower} within 10 mins outdoors in {area}.",
    "Air actually felt clean today near {landmark}! Wind must have shifted. {symptom}",
]

LANDMARKS = {
    "Delhi": "India Gate", "Noida": "Sector 62", "Gurugram": "Cyber Hub",
    "Ghaziabad": "Vaishali", "Kanpur": "Mall Road", "Lucknow": "Hazratganj",
    "Patna": "Gandhi Maidan", "Amritsar": "Golden Temple", "Ludhiana": "Ferozepur Road",
    "Mumbai": "Marine Drive", "Kolkata": "Park Street", "Bengaluru": "MG Road",
    "Chandigarh": "Sector 17", "Jaipur": "Hawa Mahal", "Varanasi": "Assi Ghat",
}


def _seed_posts(df, month, day, n=22):
    """Generates a realistic-looking seeded community feed correlated with
    actual simulated pollution levels (worse air -> more/worse-toned posts)."""
    rng = np.random.default_rng(month * 100 + day + 7)
    now = datetime(2025, month, min(day, 28), 9, 0)

    # bias city selection toward higher-pollution cities (more complaints where air is worse)
    weights = np.clip(df["PM2.5"].values, 1, None)
    weights = weights / weights.sum()
    picked_cities = rng.choice(df["city"].values, size=n, p=weights, replace=True)

    posts = []
    for i, city in enumerate(picked_cities):
        row = df[df["city"] == city].iloc[0]
        pm25 = row["PM2.5"]
        area = f"{city}"
        landmark = LANDMARKS.get(city, f"{city} city center")

        # tone of report correlates with actual PM2.5 (ground truth signal!)
        if pm25 > 250:
            symptom = rng.choice(SYMPTOM_TAGS[:6])
            visibility = int(rng.uniform(100, 600))
        elif pm25 > 120:
            symptom = rng.choice(SYMPTOM_TAGS[:6])
            visibility = int(rng.uniform(500, 1500))
        else:
            symptom = rng.choice(SYMPTOM_TAGS[6:])
            visibility = int(rng.uniform(2000, 8000))

        template = rng.choice(SAMPLE_TEMPLATES)
        text = template.format(
            vis=visibility, landmark=landmark, area=area,
            symptom=symptom, symptom_lower=symptom.split(" ", 1)[1].lower() if " " in symptom else symptom.lower(),
        )
        timestamp = now - timedelta(hours=int(rng.uniform(0, 30)), minutes=int(rng.uniform(0, 59)))

        posts.append({
            "id": f"seed_{i}",
            "handle": rng.choice(SAMPLE_HANDLES),
            "city": city,
            "text": text,
            "tag": symptom,
            "visibility_m": visibility,
            "timestamp": timestamp,
            "likes": int(rng.uniform(2, 340)),
            "verified_ground_truth": bool(pm25 > 150 and "🌫️" in symptom or "😤" in symptom or "🤧" in symptom),
            "is_user_post": False,
        })

    return sorted(posts, key=lambda p: p["timestamp"], reverse=True)


def confidence_boost(df, posts_df):
    """
    Community-corroboration signal: for each city, what fraction of recent
    posts report negative symptoms vs. positive/clear? Where crowd reports
    strongly agree with the satellite/model-estimated hotspot, we flag
    'ground-truth confirmed' — the actual product differentiator: model
    confidence rises when independent citizen reports corroborate it.
    """
    if len(posts_df) == 0:
        return df.assign(community_confirmed=False, negative_report_ratio=0.0)

    neg_tags = set(SYMPTOM_TAGS[:6])
    grp = posts_df.groupby("city").apply(
        lambda g: (g["tag"].isin(neg_tags).sum()) / len(g)
    ).rename("negative_report_ratio")

    merged = df.merge(grp, on="city", how="left")
    merged["negative_report_ratio"] = merged["negative_report_ratio"].fillna(0)
    merged["community_confirmed"] = (merged["negative_report_ratio"] >= 0.6) & (merged["AQI"] >= 200)
    return merged
