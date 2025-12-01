# backend/app.py
import os
import re
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tweepy import Client
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import dateparser
import pandas as pd
import time

# ---------- config ----------
load_dotenv()
TWITTER_BEARER = os.getenv("TWITTER_BEARER_TOKEN")
if not TWITTER_BEARER:
    raise RuntimeError("Set TWITTER_BEARER_TOKEN in .env")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CACHE_DIR = Path(os.getenv("EMBED_CACHE_DIR", "embed_cache"))
CSV_DIR = Path(os.getenv("CSV_DIR", "scraped_csvs"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

# ---------- model & client ----------
print("Loading embedding model:", EMBED_MODEL_NAME)
model = SentenceTransformer(EMBED_MODEL_NAME)
# Enable wait_on_rate_limit to let tweepy handle basic rate limit waits.
twitter_client = Client(bearer_token=TWITTER_BEARER, wait_on_rate_limit=True)

app = Flask(__name__)
CORS(app)  # enable CORS for dev; restrict in production

# ---------- helpers ----------
def _sha256(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.pkl"

def load_cached_embedding(key: str):
    p = _cache_path(key)
    if p.exists():
        try:
            with p.open("rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def save_cached_embedding(key: str, vec):
    p = _cache_path(key)
    try:
        with p.open("wb") as f:
            pickle.dump(vec, f)
    except Exception:
        pass

def embed_texts_local(texts):
    """Embed texts locally with per-text cache."""
    results = [None] * len(texts)
    to_compute = []
    for i, t in enumerate(texts):
        key = _sha256(EMBED_MODEL_NAME + "::" + t)
        cached = load_cached_embedding(key)
        if cached is not None:
            results[i] = cached
        else:
            to_compute.append((i, t, key))
    if to_compute:
        idxs, texts_to_compute, keys = zip(*to_compute)
        embedded = model.encode(list(texts_to_compute), convert_to_numpy=True, show_progress_bar=False)
        for j, vec in enumerate(embedded):
            i = idxs[j]; k = keys[j]
            vec_list = vec.astype(float).tolist()
            results[i] = vec_list
            save_cached_embedding(k, vec_list)
    return results

def cosine_sim(a, b):
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    denom = (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    return 0.0 if denom == 0 else float(np.dot(a_arr, b_arr) / denom)

# ---------- datetime helpers ----------
def to_iso_day_range(d: datetime):
    start = datetime(d.year, d.month, d.day)
    end = start + timedelta(days=1)
    return start.strftime("%Y-%m-%dT00:00:00Z"), end.strftime("%Y-%m-%dT00:00:00Z")

def month_year_to_range(month: int, year: int):
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)
    return start.strftime("%Y-%m-%dT00:00:00Z"), end.strftime("%Y-%m-%dT00:00:00Z")

# ---------- date extraction ----------
def extract_date_range(text):
    text_low = (text or "").lower()

    # explicit: from X to Y or between X and Y
    m = re.search(r"(?:from|between)\s+(.+?)\s+(?:to|and|-)\s+(.+)", text, flags=re.I)
    if m:
        d1 = dateparser.parse(m.group(1))
        d2 = dateparser.parse(m.group(2))
        if d1 and d2:
            start = datetime(d1.year, d1.month, d1.day).strftime("%Y-%m-%dT00:00:00Z")
            end = (datetime(d2.year, d2.month, d2.day) + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
            return start, end

    # on dd-mm-yyyy or dd/mm/yyyy
    m = re.search(r"\bon\s+([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})\b", text, flags=re.I)
    if m:
        d = dateparser.parse(m.group(1), settings={'DATE_ORDER': 'DMY'})
        if d:
            return to_iso_day_range(d)

    # in Month YYYY -> whole month
    m = re.search(r"\bin\s+([A-Za-z]+)\s*,?\s*(\d{4})\b", text, flags=re.I)
    if m:
        month_name = m.group(1)
        year = int(m.group(2))
        dt = dateparser.parse(month_name + " " + str(year))
        if dt:
            return month_year_to_range(dt.month, dt.year)

    # general "on <date words>"
    m = re.search(r"\bon\s+(.+?)(?:\s|$)", text, flags=re.I)
    if m:
        d = dateparser.parse(m.group(1))
        if d:
            return to_iso_day_range(d)

    # relative words
    if "yesterday" in text_low:
        d = datetime.utcnow().date() - timedelta(days=1)
        return to_iso_day_range(datetime(d.year, d.month, d.day))
    if "today" in text_low:
        d = datetime.utcnow().date()
        return to_iso_day_range(datetime(d.year, d.month, d.day))

    # last N days
    m = re.search(r"last\s+(\d{1,3})\s+days?", text_low)
    if m:
        n = int(m.group(1))
        end = datetime.utcnow().date() + timedelta(days=1)
        start = datetime.utcnow().date() - timedelta(days=n-1)
        return datetime(start.year, start.month, start.day).strftime("%Y-%m-%dT00:00:00Z"), datetime(end.year, end.month, end.day).strftime("%Y-%m-%dT00:00:00Z")

    return None, None

# ---------- basic NL parser (tailored to the user's examples) ----------
def parse_basic_nlq(text):
    text_orig = text or ""
    text = text.strip()

    # count
    count = None
    m = re.search(r"(\d{1,5})\s*(?:tweets|tweet)\b", text, flags=re.I)
    if m:
        count = int(m.group(1))

    # username
    username = None
    m = re.search(r"@([A-Za-z0-9_]{1,50})", text)
    if m:
        username = m.group(1)
    if not username:
        m = re.search(r"(?:of|from)\s+@?([A-Za-z0-9_&\.\-]{2,60})(?:\b|account|about)", text, flags=re.I)
        if m:
            username = m.group(1)
    if not username:
        m = re.search(r"\b([A-Za-z][A-Za-z0-9&\.\-]{1,60})'s\b", text)
        if m:
            username = m.group(1)
    if not username:
        m = re.search(r"\b([A-Z][a-zA-Z0-9&\.\-]{2,50})\b", text)
        if m:
            username = m.group(1)

    # topic
    topic = None
    m = re.search(r"(?:about|regarding|related to|on|based on)\s+(.+)$", text, flags=re.I)
    if m:
        topic = m.group(1).strip()
        topic = re.sub(r"\bon\s+([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})\b", "", topic, flags=re.I).strip()
        topic = re.sub(r"\bin\s+[A-Za-z]+\s*,?\s*\d{4}\b", "", topic, flags=re.I).strip()
    else:
        if username:
            parts = re.split(re.escape(username), text, flags=re.I)
            if len(parts) > 1 and parts[1].strip():
                cand = parts[1].strip()
                cand = re.sub(r"^(?:'s|s|of|about|regarding|on|related to|based on)\b", "", cand, flags=re.I).strip()
                cand = re.sub(r"\bon\s+.*$", "", cand, flags=re.I).strip()
                if cand:
                    topic = cand

    # date range
    start_time, end_time = extract_date_range(text)

    return {
        "username": username,
        "count": count,
        "topic": topic,
        "start_time": start_time,
        "end_time": end_time,
        "debug_parse": {"orig": text_orig, "username": username, "count": count, "topic": topic, "start_time": start_time, "end_time": end_time}
    }

# ---------- pagination fetch ----------
def fetch_user_tweets_paged(user_id, max_tweets=10, start_time=None, end_time=None):
    """
    Fetch up to max_tweets tweets for the given user_id.
    This function enforces a hard cap of 10 tweets per call by default.
    """
    HARD_CAP = 10
    collected = []
    next_token = None
    # enforce hard cap
    max_tweets = min(int(max_tweets), HARD_CAP)
    remaining = max_tweets
    tweet_fields = ["created_at", "public_metrics", "lang", "entities", "attachments"]
    while remaining > 0:
        # request at most 10 tweets per page (and never more than remaining)
        to_request = min(10, remaining)
        try:
            resp = twitter_client.get_users_tweets(
                id=user_id,
                max_results=to_request,
                pagination_token=next_token,
                tweet_fields=tweet_fields,
                start_time=start_time,
                end_time=end_time
            )
        except Exception as e:
            return collected, {"error": str(e)}
        if not resp or not resp.data:
            break
        for t in resp.data:
            td = {
                "id": getattr(t, "id", None),
                "text": getattr(t, "text", "") or "",
                "created_at": str(getattr(t, "created_at", "")),
                "public_metrics": getattr(t, "public_metrics", {}) or {},
            }
            try:
                ent = getattr(t, "entities", None)
                if ent: td["entities"] = ent
            except: pass
            try:
                att = getattr(t, "attachments", None)
                if att: td["attachments"] = att
            except: pass
            collected.append(td)
        remaining = max_tweets - len(collected)
        # get next_token from resp.meta (tweepy may expose differently)
        next_token = None
        meta = getattr(resp, "meta", None)
        try:
            if isinstance(meta, dict):
                next_token = meta.get("next_token")
            else:
                next_token = getattr(resp, "meta", None) and resp.meta.get("next_token")
        except:
            next_token = None
        if not next_token:
            break
        # small pause to be polite; tweepy.wait_on_rate_limit is enabled
        time.sleep(0.2)
    return collected, None

# ---------- save CSV ----------
def save_tweets_to_csv(username, topic, tweets):
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_topic = re.sub(r"[^\w\-]+", "_", (topic or "all"))
    filename = f"{username}_{safe_topic}_{now}.csv"
    path = CSV_DIR / filename
    rows = []
    for t in tweets:
        pm = t.get("public_metrics", {}) or {}
        rows.append({
            "id": t.get("id"),
            "text": t.get("text"),
            "created_at": t.get("created_at"),
            "like_count": int(pm.get("like_count", 0)),
            "retweet_count": int(pm.get("retweet_count", 0)),
            "reply_count": int(pm.get("reply_count", 0)),
            "quote_count": int(pm.get("quote_count", 0)),
            "score": t.get("score", None)
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return str(path.resolve())

def _csv_filename_from_path(path_str):
    return Path(path_str).name

# ---------- main route ----------
@app.route("/nl_search_csv", methods=["POST"])
def nl_search_csv():
    payload = request.get_json(force=True) or {}
    # Enforce hard caps here as well: never allow fetch_max or default_count > 10
    fetch_max = min(int(payload.get("fetch_max") or 10), 10)
    default_count = min(int(payload.get("default_count") or 10), 10)

    nlq = (payload.get("nl_query") or "").strip()
    if not nlq:
        return jsonify({"error": "nl_query required"}), 400

    parsed = parse_basic_nlq(nlq)
    username = parsed.get("username")
    count = parsed.get("count") or default_count
    # enforce count cap
    count = min(int(count), 10)
    topic = parsed.get("topic")
    start_time = parsed.get("start_time")
    end_time = parsed.get("end_time")

    if not username:
        return jsonify({"error": "Could not detect username in query", "debug_parse": parsed}), 400
    username = username.lstrip("@")

    try:
        user_resp = twitter_client.get_user(username=username)
    except Exception as e:
        return jsonify({"error": "Twitter API error resolving user", "detail": str(e), "debug_parse": parsed}), 500
    if user_resp.data is None:
        return jsonify({"error": f"User @{username} not found", "debug_parse": parsed}), 404
    user_id = user_resp.data.id

    # fetch tweets with enforced cap
    tweets_raw, fetch_err = fetch_user_tweets_paged(user_id, max_tweets=fetch_max, start_time=start_time, end_time=end_time)
    if fetch_err:
        return jsonify({"error": "Twitter API fetch error", "detail": fetch_err, "debug_parse": parsed}), 500
    if not tweets_raw:
        return jsonify({"username": username, "requested_query": nlq, "count_returned": 0, "tweets": [], "debug_parse": parsed})

    semantic_query = topic if topic else nlq
    texts_to_embed = [semantic_query] + [t["text"] for t in tweets_raw]
    embeddings = embed_texts_local(texts_to_embed)
    query_vec = embeddings[0]
    tweet_vecs = embeddings[1:]

    results = []
    for i, t in enumerate(tweets_raw):
        score = cosine_sim(query_vec, tweet_vecs[i])
        pm = t.get("public_metrics", {}) or {}
        t_out = {
            "id": t.get("id"),
            "text": t.get("text"),
            "created_at": t.get("created_at"),
            "like_count": int(pm.get("like_count", 0)),
            "retweet_count": int(pm.get("retweet_count", 0)),
            "reply_count": int(pm.get("reply_count", 0)),
            "quote_count": int(pm.get("quote_count", 0)),
            "score": score
        }
        results.append(t_out)

    results_sorted = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
    top_k = results_sorted[:count]

    csv_path = save_tweets_to_csv(username, topic or "all", top_k)
    csv_filename = _csv_filename_from_path(csv_path)
    csv_url = f"/download_csv/{csv_filename}"

    return jsonify({
        "username": username,
        "requested_query": nlq,
        "used_topic": semantic_query,
        "count_returned": len(top_k),
        "csv_filename": csv_filename,
        "csv_url": csv_url,
        "tweets": top_k,
        "debug_parse": parsed
    })

# ---------- download endpoint ----------
@app.route("/download_csv/<path:filename>", methods=["GET"])
def download_csv(filename):
    safe_name = Path(filename).name
    full_path = CSV_DIR / safe_name
    if not full_path.exists() or not full_path.is_file():
        return jsonify({"error": "file not found"}), 404
    return send_from_directory(directory=str(CSV_DIR.resolve()), path=safe_name, as_attachment=True)

@app.route("/")
def home():
    return jsonify({"message": "Twitter NLQ CSV Scraper (Flask) - advanced parser + CSV"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=True)
