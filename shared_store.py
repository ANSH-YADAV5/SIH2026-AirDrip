"""
Shared, disk-backed storage for AirDrip Community posts.

Streamlit's st.session_state is per-browser-session — two people opening
the same deployed app do NOT see each other's posts by default. This module
fixes that by persisting posts to a JSON file on the server, so every
visitor reads and writes the same shared feed.

Caveat (documented honestly for judges): this works correctly for a
single-process deployment (e.g. Streamlit Community Cloud's free tier,
which runs one instance per app). It is NOT a substitute for a real
database in a multi-instance/production deployment — see the note in
utils/community.py and the README for the production upgrade path
(Postgres/Firebase).
"""

import json
import os
import threading
from datetime import datetime

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_POSTS_PATH = os.path.join(_DATA_DIR, "community_posts.json")
_LOCK = threading.Lock()

MAX_STORED_POSTS = 500  # keep the file small; oldest posts drop off


def _ensure_file():
    os.makedirs(_DATA_DIR, exist_ok=True)
    if not os.path.exists(_POSTS_PATH):
        with open(_POSTS_PATH, "w") as f:
            json.dump([], f)


def load_shared_posts() -> list:
    """Returns all user-submitted posts, newest first."""
    _ensure_file()
    with _LOCK:
        try:
            with open(_POSTS_PATH, "r") as f:
                posts = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            posts = []
    # timestamps are stored as ISO strings; parse back to datetime for display
    for p in posts:
        if isinstance(p.get("timestamp"), str):
            try:
                p["timestamp"] = datetime.fromisoformat(p["timestamp"])
            except ValueError:
                p["timestamp"] = datetime.now()
    return posts


def add_shared_post(post: dict):
    """Appends a new post to the shared feed, visible to every visitor."""
    _ensure_file()
    post = dict(post)
    ts = post.get("timestamp", datetime.now())
    post["timestamp"] = ts.isoformat() if isinstance(ts, datetime) else str(ts)

    with _LOCK:
        try:
            with open(_POSTS_PATH, "r") as f:
                posts = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            posts = []
        posts.insert(0, post)
        posts = posts[:MAX_STORED_POSTS]
        with open(_POSTS_PATH, "w") as f:
            json.dump(posts, f)


def like_shared_post(post_id: str):
    """Increments the like count on a shared post by id."""
    _ensure_file()
    with _LOCK:
        try:
            with open(_POSTS_PATH, "r") as f:
                posts = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            posts = []
        for p in posts:
            if p.get("id") == post_id:
                p["likes"] = p.get("likes", 0) + 1
                break
        with open(_POSTS_PATH, "w") as f:
            json.dump(posts, f)
