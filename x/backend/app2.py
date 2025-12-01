# backend/app.py
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tweepy import Client
from dotenv import load_dotenv
import dateparser
import pandas as pd

# ---------- config ----------
load_dotenv()
TWITTER_BEARER = os.getenv("TWITTER_BEARER_TOKEN")
if not TWITTER_BEARER:
    raise RuntimeError("Set TWITTER_BEARER_TOKEN in .env")

CSV_DIR = Path(os.getenv("CSV_DIR", "scraped_csvs"))
CSV_DIR.mkdir(parents=True, exist_ok=True)

twitter_client = Client(bearer_token=TWITTER_BEARER, wait_on_rate_limit=True)

app = Flask(__name__)
CORS(app)


# DATE PARSING HELPERS
def to_iso_day_range(d):
    start = datetime(d.year, d.month, d.day)
    end = start + timedelta(days=1)
    return start.strftime("%Y-%m-%dT00:00:00Z"), end.strftime("%Y-%m-%dT00:00:00Z")

def month_year_to_range(month, year):
    start = datetime(year, month, 1)
    end = datetime(year + (month == 12), (month % 12) + 1, 1)
    return start.strftime("%Y-%m-%dT00:00:00Z"), end.strftime("%Y-%m-%dT00:00:00Z")

def extract_date_range(text):
    text_low = text.lower()

    m = re.search(r"(?:from|between)\s+(.+?)\s+(?:to|and|-)\s+(.+)", text, flags=re.I)
    if m:
        d1, d2 = dateparser.parse(m.group(1)), dateparser.parse(m.group(2))
        if d1 and d2:
            return (
                datetime(d1.year, d1.month, d1.day).strftime("%Y-%m-%dT00:00:00Z"),
                (datetime(d2.year, d2.month, d2.day)+timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
            )

    m = re.search(r"\bon\s+([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})", text, flags=re.I)
    if m:
        d = dateparser.parse(m.group(1), settings={"DATE_ORDER":"DMY"})
        if d: return to_iso_day_range(d)

    m = re.search(r"\bin\s+([A-Za-z]+)\s*,?\s*(\d{4})", text, flags=re.I)
    if m:
        dt = dateparser.parse(f"{m.group(1)} {m.group(2)}")
        if dt: return month_year_to_range(dt.month, dt.year)

    m = re.search(r"\bon\s+(.+?)(?:\s|$)", text, flags=re.I)
    if m:
        d = dateparser.parse(m.group(1))
        if d: return to_iso_day_range(d)

    if "yesterday" in text_low:
        d = datetime.utcnow().date() - timedelta(days=1)
        return to_iso_day_range(datetime(d.year, d.month, d.day))

    if "today" in text_low:
        d = datetime.utcnow().date()
        return to_iso_day_range(datetime(d.year, d.month, d.day))

    m = re.search(r"last\s+(\d+)\s+days?", text_low)
    if m:
        n = int(m.group(1))
        start = datetime.utcnow().date() - timedelta(days=n-1)
        end = datetime.utcnow().date() + timedelta(days=1)
        return (
            datetime(start.year, start.month, start.day).strftime("%Y-%m-%dT00:00:00Z"),
            datetime(end.year, end.month, end.day).strftime("%Y-%m-%dT00:00:00Z"),
        )

    return None, None

# BASIC NL QUERY PARSER
def parse_basic_nlq(text):
    text_orig = text
    text = text.strip()

    count = None
    m = re.search(r"(\d{1,5})\s*tweets?", text, flags=re.I)
    if m:
        count = int(m.group(1))

    username = None
    m = re.search(r"@([A-Za-z0-9_]+)", text)
    if m:
        username = m.group(1)
    else:
        m = re.search(r"(?:of|from)\s+@?([A-Za-z0-9_]+)", text, flags=re.I)
        if m:
            username = m.group(1)

    topic = None
    m = re.search(r"(?:about|on|regarding|related to)\s+(.+)", text, flags=re.I)
    if m:
        topic = m.group(1).strip()

    start_time, end_time = extract_date_range(text)

    return {
        "username": username,
        "count": count,
        "topic": topic,
        "start_time": start_time,
        "end_time": end_time,
        "debug_parse": {
            "orig": text_orig,
            "username": username,
            "count": count,
            "topic": topic
        }
    }


# CSV SAVE
def save_tweets_to_csv(username, topic, tweets):
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_topic = re.sub(r"[^\w\-]+", "_", (topic or "all"))
    filename = f"{username}_{safe_topic}_{now}.csv"
    path = CSV_DIR / filename
    pd.DataFrame(tweets).to_csv(path, index=False)
    return str(path.resolve())

def _csv_filename_from_path(path_str):
    return Path(path_str).name


# MAIN ROUTE — topic used as hashtag
@app.route("/nl_search_csv", methods=["POST"])
def nl_search_csv():
    payload = request.get_json(force=True) or {}

    nlq = (payload.get("nl_query") or "").strip()
    if not nlq:
        return jsonify({"error": "nl_query required"}), 400

    parsed = parse_basic_nlq(nlq)
    username = parsed["username"]
    topic = parsed["topic"]
    count = min(parsed["count"] or 10, 10)
    fetch_max = count
    start_time = parsed["start_time"]
    end_time = parsed["end_time"]

    if not username:
        return jsonify({"error": "Could not detect username", "debug": parsed}), 400

    username = username.lstrip("@")

    # Ensure user exists
    user_resp = twitter_client.get_user(username=username)
    if not user_resp.data:
        return jsonify({"error": f"User @{username} not found"}), 404

    # Convert topic into hashtag
    if not topic:
        return jsonify({"error": "Topic required. Example: '5 tweets from elon about AI'"}), 400

    hashtag = "#" + topic.replace(" ", "")

    # Build final query
    query = f"({hashtag}) (from:{username})"

    # Twitter API call DIRECTLY here
    resp = twitter_client.search_recent_tweets(
        query=query,
        max_results=fetch_max,
        tweet_fields=["created_at", "public_metrics"],
        expansions=["author_id"]
    )

    tweets_raw = resp.data or []

    # Extract metrics
    tweets = []
    for t in tweets_raw:
        pm = getattr(t, "public_metrics", {}) or {}
        tweets.append({
            "id": t.id,
            "text": t.text,
            "created_at": str(t.created_at),
            "like_count": pm.get("like_count", 0),
            "retweet_count": pm.get("retweet_count", 0),
            "reply_count": pm.get("reply_count", 0),
            "quote_count": pm.get("quote_count", 0),
        })

    # Save CSV
    csv_path = save_tweets_to_csv(username, hashtag, tweets)
    csv_filename = _csv_filename_from_path(csv_path)

    return jsonify({
        "username": username,
        "topic_used_as_hashtag": hashtag,
        "count_returned": len(tweets),
        "csv_filename": csv_filename,
        "csv_url": f"/download_csv/{csv_filename}",
        "tweets": tweets,
        "debug_parse": parsed
    })

# --------------------------------------------------
@app.route("/download_csv/<path:filename>")
def download_csv(filename):
    return send_from_directory(CSV_DIR, filename, as_attachment=True)

@app.route("/")
def home():
    return jsonify({"message": "Twitter Scraper"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=True)
