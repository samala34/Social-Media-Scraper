# app_streamlit.py
"""
Streamlit app for LinkedIn scraping pipeline (SerpAPI + ScrapingBee) with optional MongoDB storage.
- Discover LinkedIn post URLs via SerpAPI queries.
- Fetch HTML via ScrapingBee (frontend key supported).
- Parse posts with linkedin_parser (content, likes, comments_list, raw_jsonld).
- When comments_list is empty, try to fetch embedUrl to extract more comments.
- Save combined JSON and update master Excel with two sheets: Posts and Comments.
- Optionally upsert results into MongoDB and browse/download JSON from DB.
"""

import os
import time
import json
import re
from typing import List, Set, Dict, Optional, Any
from datetime import datetime

import streamlit as st
import pandas as pd
import requests

# Mongo client
try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None  # will show instructions in UI if missing

# Try to import client fetch_html (new refactor) or fallback to scraper.fetch_html
client_fetch_html = None
try:
    from scrapingbee_client import fetch_html as client_fetch_html
except Exception:
    try:
        from scraper import fetch_html as client_fetch_html
    except Exception:
        client_fetch_html = None

# Parser import
try:
    from linkedin_parser import parse_linkedin_html, LinkedInParser
except Exception:
    try:
        from parse_linkedin_post import parse_linkedin_html, LinkedInParser  # type: ignore
    except Exception:
        parse_linkedin_html = None
        LinkedInParser = None

# ---------- Utility / Helpers ----------
import urllib.parse


def extract_linkedin_company_slugs_from_url(url: str):
    try:
        parsed = urllib.parse.urlparse(url)
        path = parsed.path
        parts = [p for p in path.split("/") if p]
        if "company" in parts:
            idx = parts.index("company")
            if idx + 1 < len(parts):
                return [parts[idx + 1]]
        if parts:
            return [parts[-1]]
    except Exception:
        pass
    return []


def normalize(s):
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def company_matches_parsed(parsed: Dict, company_names: List[str], company_slugs: List[str] = None) -> bool:
    company_slugs = company_slugs or []
    names_norm = [normalize(x) for x in company_names if x]
    slugs_norm = [normalize(x) for x in company_slugs if x]

    author = normalize(parsed.get("author") or parsed.get("creator") or "")
    for n in names_norm:
        if n and n in author:
            return True

    raw = parsed.get("raw_jsonld") or []
    for obj in raw:
        try:
            auth = obj.get("author") or obj.get("creator") or obj.get("publisher")
            if isinstance(auth, dict):
                aname = normalize(auth.get("name"))
                aurl = normalize(auth.get("url") or auth.get("sameAs") or "")
                for n in names_norm:
                    if n and n in aname:
                        return True
                for s in slugs_norm:
                    if s and s in aurl:
                        return True
            elif isinstance(auth, str):
                for n in names_norm:
                    if n and n in normalize(auth):
                        return True
        except Exception:
            pass

    post_url = normalize(parsed.get("url") or parsed.get("fetched_url") or "")
    for s in slugs_norm:
        if s and s in post_url:
            return True

    content = normalize(parsed.get("content") or parsed.get("description") or "")
    for n in names_norm:
        if n and content.startswith(n):
            return True

    return False


# -------------------- SerpAPI search helper --------------------
def serpapi_search(query: str, top: int = 10, serpapi_key: str = None) -> List[str]:
    key = serpapi_key or os.getenv("SERPAPI_KEY")
    if not key:
        raise RuntimeError("SerpAPI key not provided. Set SERPAPI_KEY env var or provide key in app.")
    endpoint = "https://serpapi.com/search.json"
    params = {"engine": "google", "q": query, "num": top, "api_key": key}
    r = requests.get(endpoint, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    urls = []
    for item in data.get("organic_results", []):
        url = item.get("link")
        if url and "linkedin.com" in url:
            urls.append(url)
    return urls


# -------------------- local fetch wrapper --------------------
def local_fetch_html(url: str, scrapingbee_key: Optional[str] = None, render_js: bool = True, save_path: Optional[str] = None, timeout: int = 60) -> str:
    """
    Prefer a client_fetch_html (from scrapingbee_client or scraper) and pass the frontend key.
    Falls back to a direct requests call to ScrapingBee endpoint if client_fetch_html is not available.
    """
    if client_fetch_html:
        try:
            return client_fetch_html(url, render_js=render_js, save_path=save_path, timeout=timeout, api_key=scrapingbee_key)
        except TypeError:
            return client_fetch_html(url, render_js=render_js, save_path=save_path, timeout=timeout)

    key = scrapingbee_key or os.getenv("SCRAPINGBEE_KEY")
    if not key:
        raise RuntimeError("No ScrapingBee key found. Set SCRAPINGBEE_KEY env var or provide key in app.")
    endpoint = "https://app.scrapingbee.com/api/v1/"
    params = {"api_key": key, "url": url, "render_js": "true" if render_js else "false"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    r = requests.get(endpoint, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    html = r.text
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)
    return html


def local_parse_html(html: str, source_filename: str = None) -> Dict:
    if parse_linkedin_html:
        return parse_linkedin_html(html, source_filename=source_filename)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    out = {"url": None, "title": None, "content": None, "likes": 0, "comments": 0, "reposts": 0,
           "author": None, "date_published": None, "images": [], "raw_jsonld": [], "comments_list": []}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            out["raw_jsonld"].append(data)
            if isinstance(data, dict) and data.get("articleBody"):
                out["content"] = data.get("articleBody")
                out["title"] = data.get("headline") or out["title"]
        except Exception:
            continue
    can = soup.find("link", rel="canonical")
    if can and can.get("href"):
        out["url"] = can["href"]
    return out


# -------------------- Streamlit UI --------------------
st.set_page_config(layout="wide", page_title="LinkedIn Scraper for Marketing Teams")
st.title("LinkedIn Scraper — Search, Parse, Filter (SerpAPI + ScrapingBee + Mongo)")

with st.sidebar:
    st.header("API Keys (optional)")
    serp_key_input = st.text_input("SerpAPI Key", value=os.getenv("SERPAPI_KEY") or "", type="password")
    sb_key_input = st.text_input("ScrapingBee Key", value=os.getenv("SCRAPINGBEE_KEY") or "", type="password")
    st.markdown("---")
    st.header("MongoDB (optional)")
    st.caption("Provide MongoDB connection URI (Atlas or local). Example (Atlas): mongodb+srv://user:pass@cluster0.mongodb.net/dbname")
    mongodb_uri_input = st.text_input("MongoDB URI", value=os.getenv("MONGODB_URI") or "", type="password")
    mongodb_db = st.text_input("Mongo DB name", value=os.getenv("MONGODB_DB") or "linkedin_scraper")
    mongodb_collection = st.text_input("Mongo Collection name", value=os.getenv("MONGODB_COLLECTION") or "posts")
    st.markdown("---")
    st.header("Pipeline options")
    top_n = st.number_input("Top N results per query", min_value=1, max_value=50, value=8, step=1)
    render_js = st.checkbox("Render JS when fetching pages (recommended)", value=True)
    delay_between_fetch = st.number_input("Delay between fetches (seconds)", min_value=0.0, max_value=5.0, value=0.8, step=0.1)
    st.markdown("---")
    st.header("Output")
    master_excel_name = st.text_input("Master Excel filename", value="linkedin_posts_master.xlsx")
    combined_json_name = st.text_input("Combined JSON filename", value="all_posts_combined.json")

mode = st.selectbox("Select mode", ["Event search", "Company search", "Specific post URLs"])

if mode == "Event search":
    st.subheader("Event search")
    event_keywords = st.text_area("Event keywords (one per line). Examples: children's day, #ChildrensDay, mothers day", height=120)
    companies_filter = st.text_area("Optional: restrict to companies (one per line)", height=80)
elif mode == "Company search":
    st.subheader("Company search")
    companies_input = st.text_area("Company names (one per line)", height=200)
    event_filter = st.text_area("Optional: event keywords to filter company posts (one per line)", height=80)
else:
    st.subheader("Specific post URLs")
    urls_input = st.text_area("Paste LinkedIn post URLs (one per line)", height=200)

st.sidebar.header("Company-only filter (optional)")
only_company = st.sidebar.checkbox("Only posts by specified company", value=False)
company_input = st.sidebar.text_input("Company name(s) (comma-separated)", value="")
slug_input = st.sidebar.text_input("Company slug(s) (comma-separated)", value="")

# Date range
st.sidebar.header("Date range filter (optional)")
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start date (inclusive)", value=None)
end_date = col2.date_input("End date (inclusive)", value=None)

# Mongo controls
st.sidebar.markdown("---")
st.sidebar.header("MongoDB actions")
save_to_mongo_checkbox = st.sidebar.checkbox("Auto-save parsed results to Mongo after run", value=False)
mongo_test_connect = st.sidebar.button("Test Mongo Connection")

run = st.button("Run pipeline")
results_placeholder = st.empty()

# ---------- Mongo helpers ----------
def get_mongo_client(uri: str) -> Optional[Any]:
    if not MongoClient:
        return None
    if not uri:
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # quick ping
        client.admin.command('ping')
        return client
    except Exception as e:
        return None

def normalize_for_mongo(parsed: Dict) -> Dict:
    """
    Prepare parsed dict for Mongo insertion:
      - convert datetime-like objects to ISO strings
      - ensure no non-serializable types
    """
    p = dict(parsed)  # shallow copy
    # normalize date_published
    dp = p.get("date_published")
    if dp:
        # convert to ISO str if it's a datetime
        try:
            if isinstance(dp, str):
                # attempt to parse and reformat (leave as-is if can't parse)
                p["date_published"] = dp
            else:
                p["date_published"] = str(dp)
        except Exception:
            p["date_published"] = str(dp)
    # comments_list is list of dicts (ensure it's JSONable)
    cl = p.get("comments_list")
    if cl is None:
        p["comments_list"] = []
    # ensure images is list
    imgs = p.get("images")
    if imgs is None:
        p["images"] = []
    # add fetched_at
    p["fetched_at"] = datetime.utcnow().isoformat() + "Z"
    return p

def upsert_parsed_results_to_mongo(parsed_results: List[Dict], uri: str, dbname: str, collname: str) -> dict:
    """
    Upsert parsed results into Mongo collection by 'url' key.
    Returns summary dict: {inserted: n, updated: m, errors: []}
    """
    res = {"inserted": 0, "updated": 0, "errors": []}
    client = get_mongo_client(uri)
    if not client:
        res["errors"].append("Failed to connect to MongoDB with provided URI.")
        return res
    db = client[dbname]
    coll = db[collname]
    for parsed in parsed_results:
        try:
            doc = normalize_for_mongo(parsed)
            url = doc.get("url") or doc.get("fetched_url")
            if not url:
                # skip documents with no url
                continue
            # use replace_one with upsert to keep the document in sync
            result = coll.replace_one({"url": url}, doc, upsert=True)
            # PyMongo's replace_one returns matched_count / modified_count
            # If upserted_id is present, it was inserted; otherwise it was an update or no-op.
            if getattr(result, "upserted_id", None):
                res["inserted"] += 1
            else:
                res["updated"] += 1
        except Exception as e:
            res["errors"].append(str(e))
    try:
        client.close()
    except Exception:
        pass
    return res

def query_mongo_docs(uri: str, dbname: str, collname: str, search_term: str = "", page: int = 0, page_size: int = 20) -> dict:
    """Query mongo docs by searching title or author (simple text filter). Returns dict with docs and total count."""
    client = get_mongo_client(uri)
    if not client:
        return {"docs": [], "total": 0, "error": "Failed to connect"}
    db = client[dbname]
    coll = db[collname]
    query = {}
    if search_term:
        regex = {"$regex": re.escape(search_term), "$options": "i"}
        query = {"$or": [{"title": regex}, {"author": regex}, {"content": regex}]}
    total = coll.count_documents(query)
    docs_cursor = coll.find(query).sort("fetched_at", -1).skip(page * page_size).limit(page_size)
    docs = []
    for d in docs_cursor:
        # convert ObjectId to str and ensure JSON serializable
        d["_id"] = str(d.get("_id"))
        docs.append(d)
    try:
        client.close()
    except Exception:
        pass
    return {"docs": docs, "total": total}


# ---------- Main run block ----------
def build_queries_from_inputs(company_slugs_for_queries: List[str] = None):
    queries = []
    if mode == "Event search":
        events = [e.strip() for e in event_keywords.splitlines() if e.strip()]
        companies = [c.strip() for c in companies_filter.splitlines() if c.strip()]
        for e in events:
            queries.append(f'site:linkedin.com/posts "{e}"')
            queries.append(f'site:linkedin.com/feed/update "{e}"')
            if company_slugs_for_queries:
                for slug in company_slugs_for_queries:
                    queries.append(f'site:linkedin.com/company/{slug} "{e}"')
                    queries.append(f'site:linkedin.com/posts "linkedin.com/company/{slug}" "{e}"')
            for c in companies:
                queries.append(f'site:linkedin.com/posts "{e}" "{c}"')
                queries.append(f'site:linkedin.com/feed/update "{e}" "{c}"')

    elif mode == "Company search":
        companies = [c.strip() for c in companies_input.splitlines() if c.strip()]
        events = [e.strip() for e in event_filter.splitlines() if e.strip()]
        for c in companies:
            if company_slugs_for_queries:
                for slug in company_slugs_for_queries:
                    queries.append(f'site:linkedin.com/company/{slug}')
                    queries.append(f'site:linkedin.com/company/{slug} "{c}"')
            queries.append(f'site:linkedin.com/posts "{c}"')
            queries.append(f'site:linkedin.com/feed/update "{c}"')
            for e in events:
                if company_slugs_for_queries:
                    for slug in company_slugs_for_queries:
                        queries.append(f'site:linkedin.com/company/{slug} "{e}"')
                        queries.append(f'site:linkedin.com/posts "linkedin.com/company/{slug}" "{e}"')
                queries.append(f'site:linkedin.com/posts "{e}" "{c}"')
                queries.append(f'site:linkedin.com/feed/update "{e}" "{c}"')

    return list(dict.fromkeys(queries))


def within_date_range(parsed_obj, start_date_obj, end_date_obj):
    dp = parsed_obj.get("date_published")
    if not dp:
        return True
    try:
        dt = datetime.fromisoformat(dp.replace("Z", "+00:00"))
        d = dt.date()
        if start_date_obj and d < start_date_obj:
            return False
        if end_date_obj and d > end_date_obj:
            return False
        return True
    except Exception:
        return True


if run:
    serp_key = serp_key_input.strip() or os.getenv("SERPAPI_KEY")
    sb_key = sb_key_input.strip() or os.getenv("SCRAPINGBEE_KEY")
    mongo_uri = mongodb_uri_input.strip() or os.getenv("MONGODB_URI")
    mongo_dbname = mongodb_db.strip() or "linkedin_scraper"
    mongo_collname = mongodb_collection.strip() or "posts"

    candidate_urls: Set[str] = set()
    resolved_slugs: List[str] = []

    if mode in ("Event search", "Company search"):
        company_names = [c.strip() for c in company_input.split(",") if c.strip()]
        company_slugs_ui = [s.strip() for s in slug_input.split(",") if s.strip()]

        if only_company and company_names:
            resolved_slugs.extend(company_slugs_ui)
            for cn in company_names:
                try:
                    # try using serpapi_search helper directly
                    q = f'site:linkedin.com/company "{cn}"'
                    try:
                        urls_found = serpapi_search(q, top=6, serpapi_key=serp_key)
                    except Exception:
                        urls_found = []
                    for link in urls_found:
                        for s in extract_linkedin_company_slugs_from_url(link):
                            if s not in resolved_slugs:
                                resolved_slugs.append(s)
                except Exception:
                    pass
            if resolved_slugs:
                st.info(f"Auto-detected company slug(s): {resolved_slugs}")
                try:
                    st.sidebar.write("Suggested slug(s): " + ", ".join(resolved_slugs))
                except Exception:
                    pass

        queries = build_queries_from_inputs(company_slugs_for_queries=resolved_slugs)
        if not queries:
            st.error("No queries were built — provide keywords or companies.")
        else:
            st.info(f"Built {len(queries)} queries. Running SerpAPI (top {top_n} per query)...")
            progress = st.progress(0)
            total = len(queries)
            for i, q in enumerate(queries):
                try:
                    urls = serpapi_search(q, top=top_n, serpapi_key=serp_key)
                    for u in urls:
                        if "linkedin.com" in u:
                            candidate_urls.add(u)
                except Exception as e:
                    st.warning(f"Search failed for query: {q} — {e}")
                time.sleep(0.5)
                progress.progress(int((i+1)/total * 100))
            st.success(f"Discovered {len(candidate_urls)} candidate LinkedIn URLs.")
    else:
        urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
        candidate_urls.update(urls)
        st.info(f"Added {len(urls)} specific URLs.")

    if not candidate_urls:
        st.warning("No LinkedIn URLs found — nothing to fetch.")
    else:
        st.info(f"Fetching and parsing {len(candidate_urls)} pages (this may take a while).")
        parsed_results = []
        pbar = st.progress(0)
        total_count = len(candidate_urls)
        company_names = [c.strip() for c in company_input.split(",") if c.strip()]
        company_slugs = [s.strip() for s in slug_input.split(",") if s.strip()]
        try:
            if only_company and not company_slugs and resolved_slugs:
                company_slugs = resolved_slugs
        except NameError:
            pass

        for idx, url in enumerate(sorted(candidate_urls)):
            try:
                st.write(f"Fetching: {url}")
                html = local_fetch_html(url, scrapingbee_key=sb_key, render_js=render_js, save_path=None, timeout=60)
                parsed = local_parse_html(html, source_filename=None)
                if not parsed.get("url"):
                    parsed["url"] = url
                parsed["fetched_url"] = url

                # Ensure numeric fields exist
                for k in ("likes", "comments", "reposts"):
                    if k not in parsed or parsed.get(k) is None:
                        parsed[k] = 0

                # If parser didn't obtain comments_list, try embedUrl (from raw_jsonld)
                if not parsed.get("comments_list"):
                    embed_url = None
                    raw = parsed.get("raw_jsonld") or []
                    for obj in raw:
                        if isinstance(obj, dict):
                            ev = obj.get("embedUrl") or obj.get("embed_url") or obj.get("embedURL")
                            if ev and isinstance(ev, str) and ev.startswith("http"):
                                embed_url = ev
                                break
                    if embed_url:
                        try:
                            embed_html = local_fetch_html(embed_url, scrapingbee_key=sb_key, render_js=render_js, save_path=None, timeout=30)
                            if LinkedInParser:
                                parser = LinkedInParser()
                                embed_comments = parser._extract_comments_from_embed_html(embed_html)
                                if embed_comments:
                                    parsed["comments_list"] = embed_comments
                                    parsed["comments"] = parsed.get("comments") or len(embed_comments)
                        except Exception as e:
                            st.write("Embed comments fetch failed:", e)

                # company-only filter
                if only_company and company_names:
                    if not company_matches_parsed(parsed, company_names, company_slugs):
                        st.write(f"Skipping (not company-author): {url}")
                        time.sleep(delay_between_fetch)
                        pbar.progress(int((idx+1)/total_count * 100))
                        continue

                # date filter
                if not within_date_range(parsed, start_date, end_date):
                    st.write(f"Skipping (out of date range): {url}")
                    time.sleep(delay_between_fetch)
                    pbar.progress(int((idx+1)/total_count * 100))
                    continue

                parsed_results.append(parsed)
                st.write("Parsed:", parsed.get("title") or (parsed.get("content") or "")[:120])

            except Exception as e:
                st.error(f"Failed to fetch/parse {url}: {e}")
            time.sleep(delay_between_fetch)
            pbar.progress(int((idx+1)/total_count * 100))

        # Display results and KPIs
        if parsed_results:
            df = pd.DataFrame(parsed_results)
            # normalize lists
            if "images" in df.columns:
                df["images"] = df["images"].apply(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
            # ensure numeric types
            for col in ("likes", "comments", "reposts"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
                else:
                    df[col] = 0
            df["engagement"] = df["likes"] + df["comments"] + df["reposts"]

            # date parsing
            try:
                df["date_published"] = pd.to_datetime(df["date_published"], utc=True)
            except Exception:
                pass

            # Excel cannot handle timezone-aware datetimes → convert to naive datetime
            if "date_published" in df.columns:
                try:
                   df["date_published"] = df["date_published"].dt.tz_localize(None)
                except Exception:
                    pass

            # KPIs at top
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Posts", len(df))
            col2.metric("Total Likes", int(df["likes"].sum()))
            col3.metric("Total Comments", int(df["comments"].sum()))
            col4.metric("Total Reposts", int(df["reposts"].sum()))

            # Averages
            col5, col6, col7 = st.columns(3)
            col5.metric("Avg Likes / Post", round(df["likes"].mean(), 1))
            col6.metric("Avg Comments / Post", round(df["comments"].mean(), 1))
            col7.metric("Avg Engagement / Post", round(df["engagement"].mean(), 1))

            results_placeholder.dataframe(df[["url", "title", "author", "date_published", "likes", "comments", "reposts", "engagement"]], use_container_width=True)

            # Save outputs: combined JSON
            combined_json_path = combined_json_name
            with open(combined_json_path, "w", encoding="utf-8") as f:
                json.dump(parsed_results, f, ensure_ascii=False, indent=4)
            st.success(f"Combined JSON saved to {combined_json_path}")

            # Update master excel and also produce Comments sheet
            try:
                if os.path.exists(master_excel_name):
                    master_df = pd.read_excel(master_excel_name, engine="openpyxl")
                else:
                    master_df = pd.DataFrame()

                for parsed in parsed_results:
                    url_val = parsed.get("url")
                    row = {
                        "url": url_val,
                        "title": parsed.get("title"),
                        "author": parsed.get("author"),
                        "content": parsed.get("content"),
                        "likes": parsed.get("likes"),
                        "comments": parsed.get("comments"),
                        "reposts": parsed.get("reposts"),
                        "date_published": parsed.get("date_published"),
                        "images": json.dumps(parsed.get("images", []), ensure_ascii=False),
                        # serialized comments_list
                        "comments_list": json.dumps(parsed.get("comments_list", []), ensure_ascii=False),
                        "fetched_url": parsed.get("fetched_url")
                    }
                    if master_df is None or master_df.empty:
                        master_df = pd.DataFrame([row])
                    else:
                        exist = master_df.index[master_df["url"] == url_val].tolist()
                        if exist:
                            idx0 = exist[0]
                            for k, v in row.items():
                                master_df.at[idx0, k] = v
                        else:
                            master_df = pd.concat([master_df, pd.DataFrame([row])], ignore_index=True)

                # Prepare comments flattened table
                comments_rows = []
                for parsed in parsed_results:
                    url_val = parsed.get("url")
                    title_val = parsed.get("title")
                    for c in parsed.get("comments_list", []):
                        comments_rows.append({
                            "post_url": url_val,
                            "post_title": title_val,
                            "comment_author": c.get("author"),
                            "comment_text": c.get("text"),
                            "comment_date": c.get("date_published"),
                            "comment_likes": c.get("likes")
                        })

                # Write Posts + Comments sheets
                try:
                    with pd.ExcelWriter(master_excel_name, engine="openpyxl", mode="w") as writer:
                        master_df.to_excel(writer, sheet_name="Posts", index=False)
                        if comments_rows:
                            pd.DataFrame(comments_rows).to_excel(writer, sheet_name="Comments", index=False)
                    st.success(f"Master Excel updated: {master_excel_name}")
                except Exception as e:
                    # fallback single-sheet save
                    master_df.to_excel(master_excel_name, index=False, engine="openpyxl")
                    st.warning("Saved master sheet only (failed to write Comments sheet): " + str(e))

            except Exception as e:
                st.error(f"Failed to update master excel: {e}")

            # Download buttons: combined JSON
            json_bytes = json.dumps(parsed_results, ensure_ascii=False, indent=4).encode("utf-8")
            st.download_button("Download combined JSON", data=json_bytes, file_name=combined_json_name, mime="application/json")

            # Download results as Excel (Posts + Comments)
            try:
                import io
                tosave = io.BytesIO()
                with pd.ExcelWriter(tosave, engine="openpyxl") as writer:
                    # posts sheet
                    df_to_save = df[["url", "title", "author", "date_published", "likes", "comments", "reposts", "engagement"]]
                    df_to_save.to_excel(writer, sheet_name="Posts", index=False)
                    # comments sheet
                    if comments_rows:
                        pd.DataFrame(comments_rows).to_excel(writer, sheet_name="Comments", index=False)
                tosave.seek(0)
                st.download_button("Download results as Excel", data=tosave, file_name="linkedin_results_with_comments.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.warning("Excel download not available: " + str(e))

            # Small UI: show comments under each parsed entry (optional)
            st.markdown("### Parsed posts and comments preview")
            for parsed in parsed_results:
                st.write(parsed.get("title") or (parsed.get("content") or "")[:120])
                st.caption(f'Likes: {parsed.get("likes")}  Comments: {parsed.get("comments")}')
                if parsed.get("comments_list"):
                    with st.expander(f"Show comments ({len(parsed['comments_list'])})"):
                        for c in parsed["comments_list"]:
                            st.markdown(f"**{c.get('author') or 'Unknown'}** — {c.get('date_published') or ''}")
                            st.write((c.get("text") or "")[:1000])

            # ------------------ Mongo saving (optional) ------------------
            if mongo_uri:
                if save_to_mongo_checkbox:
                    st.info("Saving parsed results to MongoDB...")
                    summary = upsert_parsed_results_to_mongo(parsed_results, mongo_uri, mongo_dbname, mongo_collname)
                    st.success(f"Mongo save result: inserted={summary.get('inserted')} updated={summary.get('updated')}")
                    if summary.get("errors"):
                        st.warning("Mongo errors: " + "; ".join(summary.get("errors")))
                else:
                    if st.button("Save parsed results to MongoDB"):
                        if not MongoClient:
                            st.error("pymongo not installed. Run `pip install pymongo dnspython` and restart.")
                        elif not mongo_uri:
                            st.error("Provide MongoDB URI in sidebar.")
                        else:
                            st.info("Saving parsed results to MongoDB...")
                            summary = upsert_parsed_results_to_mongo(parsed_results, mongo_uri, mongo_dbname, mongo_collname)
                            st.success(f"Mongo save result: inserted={summary.get('inserted')} updated={summary.get('updated')}")
                            if summary.get("errors"):
                                st.warning("Mongo errors: " + "; ".join(summary.get("errors")))

            # ------------------ Show Mongo browser (optional) ------------------
            st.markdown("---")
            st.markdown("## MongoDB Viewer (optional)")
            st.write("Use this to browse documents you saved to Mongo and download JSON results.")

            mongo_search = st.text_input("Search term (searches title/author/content)", value="")
            mongo_page_size = st.number_input("Page size", min_value=1, max_value=200, value=20, step=1)
            mongo_page = st.number_input("Page number (0-index)", min_value=0, value=0, step=1)
            if st.button("Load from Mongo"):
                if not MongoClient:
                    st.error("pymongo not installed. Run `pip install pymongo dnspython`")
                elif not mongo_uri:
                    st.error("Provide MongoDB URI in sidebar.")
                else:
                    qres = query_mongo_docs(mongo_uri, mongo_dbname, mongo_collname, search_term=mongo_search, page=mongo_page, page_size=mongo_page_size)
                    if qres.get("error"):
                        st.error(qres.get("error"))
                    else:
                        docs = qres.get("docs", [])
                        total = qres.get("total", 0)
                        st.write(f"Total matching docs: {total}  — Showing page {mongo_page} (size {mongo_page_size})")
                        st.json(docs)
                        # Download
                        if docs:
                            b = json.dumps(docs, ensure_ascii=False, indent=2).encode("utf-8")
                            st.download_button("Download shown Mongo JSON", data=b, file_name="mongo_posts.json", mime="application/json")
        else:
            st.warning("No parsed results after fetching (maybe filtered by company/date).")

# Test mongo connection button (sidebar)
if mongo_test_connect:
    if not MongoClient:
        st.sidebar.error("pymongo not installed. Run: pip install pymongo dnspython")
    elif not (mongodb_uri_input.strip() or os.getenv("MONGODB_URI")):
        st.sidebar.error("Provide a MongoDB URI in the sidebar or set MONGODB_URI env var.")
    else:
        cli = get_mongo_client(mongodb_uri_input.strip() or os.getenv("MONGODB_URI"))
        if cli:
            st.sidebar.success("Connected to MongoDB (ping OK).")
            try:
                cli.close()
            except Exception:
                pass
        else:
            st.sidebar.error("Failed to connect to MongoDB. Check URI and network access.")
