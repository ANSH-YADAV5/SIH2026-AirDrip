"""
India state geography utilities — state-level view params for map pan/zoom,
and name-alignment between our simulated station data and the GeoJSON's
state naming.

GeoJSON source: dissolved from udit-001/india-maps-data (district-level,
2019 reorganization), simplified for web use. See data/india_states_dissolved.geojson.

⚠️ KNOWN ISSUE (verified by rendering the polygons): this dataset's internal
line between "Jammu and Kashmir" and "Ladakh" does not match India's
official position — it assigns the Gilgit-Baltistan area to Ladakh, when
the Government of India's own maps place Gilgit-Baltistan (and the rest of
Pakistan-administered "Azad Kashmir") within Jammu & Kashmir, with Ladakh
covering the Aksai Chin/Shaksgam area further east. Do NOT present this map
as an authoritative political boundary. It is left in place for the
pollution-visualization use case only (state-level color shading, city
pins), captioned accordingly in app.py. Before using this in any context
where the state split itself matters, replace india_states_dissolved.geojson
with a dataset explicitly verified against an official Survey of India /
Ministry of Home Affairs source.
"""

import json
import os
import math

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_GEOJSON_PATH = os.path.join(_DATA_DIR, "india_states_dissolved.geojson")
_VIEW_PARAMS_PATH = os.path.join(_DATA_DIR, "state_view_params.json")

# Maps names used in our simulated station data (utils/data_simulator.py)
# to the exact st_nm used in the GeoJSON.
STATE_NAME_ALIASES = {
    "J&K": "Jammu and Kashmir",
}

_geojson_cache = None
_view_params_cache = None


def load_state_geojson():
    global _geojson_cache
    if _geojson_cache is None:
        with open(_GEOJSON_PATH, "r") as f:
            _geojson_cache = json.load(f)
    return _geojson_cache


def load_view_params():
    global _view_params_cache
    if _view_params_cache is None:
        with open(_VIEW_PARAMS_PATH, "r") as f:
            _view_params_cache = json.load(f)
    return _view_params_cache


def to_geojson_name(state_name: str) -> str:
    """Translate a station-data state name to the GeoJSON's st_nm."""
    return STATE_NAME_ALIASES.get(state_name, state_name)


def to_display_name(geojson_state_name: str) -> str:
    """Reverse of to_geojson_name, for showing our own short labels back."""
    reverse = {v: k for k, v in STATE_NAME_ALIASES.items()}
    return reverse.get(geojson_state_name, geojson_state_name)


def all_state_names() -> list:
    """All 36 states/UTs, in the GeoJSON's naming, sorted alphabetically."""
    params = load_view_params()
    return sorted(params.keys())


def get_view_for_state(state_name: str):
    """Returns {center_lat, center_lon, zoom, bounds} for a state name
    (accepts either our internal alias like 'J&K' or the GeoJSON name)."""
    params = load_view_params()
    geo_name = to_geojson_name(state_name)
    return params.get(geo_name)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def nearest_city_for_state(state_name: str, df):
    """
    For states/UTs we don't have a simulated monitoring city in, find the
    nearest simulated station by great-circle distance to the state's
    centroid — used to give a 'regional estimate' rather than no data at all.
    Returns (row, distance_km) or (None, None) if df is empty.
    """
    view = get_view_for_state(state_name)
    if view is None or len(df) == 0:
        return None, None
    best_row, best_dist = None, float("inf")
    for _, row in df.iterrows():
        d = haversine_km(view["center_lat"], view["center_lon"], row["lat"], row["lon"])
        if d < best_dist:
            best_dist = d
            best_row = row
    return best_row, best_dist


def default_india_view():
    """Whole-country center/zoom (used for the 'All India' selector option)."""
    return {"center_lat": 22.5, "center_lon": 80.0, "zoom": 3.5}
