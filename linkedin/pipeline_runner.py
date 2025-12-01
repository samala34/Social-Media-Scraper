# File: pipeline_runner.py  (suggested replacement for run_pipeline.py)
"""
pipeline_runner.py
------------------
Orchestration for discovery (SerpAPI), fetching (ScrapingBee) and parsing pipeline.
Exposes the same run_keywords(...) function and CLI behaviour as original run_pipeline.py.
"""

import os
import json
import time
from typing import List, Set
from urllib.parse import quote_plus

import requests
import pandas as pd

from scrapingbee_client import ScrapingBeeClient, fetch_html  # or from scraper import fetch_html if you keep old name
from linkedin_parser import parse_linkedin_html  # or parse_linkedin_post.parse_linkedin_html

# Config defaults (keep same names for compatibility)
TOP_N_PER_KEYWORD = 10
HTML_TEMP_FOLDER = "html_temp"
PARSED_JSON_FOLDER = "parsed_jsons"
MASTER_EXCEL = "linkedin_posts_master.xlsx"
COMBINED_JSON = "all_posts_combined.json"
BING_DELAY_SEC = 1.0
SCRAPINGBEE_DELAY_SEC = 0.8

os.makedirs(HTML_TEMP_FOLDER, exist_ok=True)
os.makedirs(PARSED_JSON_FOLDER, exist_ok=True)


def serpapi_search(query: str, top: int = 10, serpapi_key: str = None):
    key = serpapi_key or os.getenv("SERPAPI_KEY") or ""
    endpoint = "https://serpapi.com/search.json"
    params = {"engine": "google", "q": query, "num": top, "api_key": key}
    r = requests.get(endpoint, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    urls = []
    for item in data.get("organic_results", []):
        url = item.get("link")
        if url:
            urls.append(url)
    return urls


class PipelineRunner:
    """Encapsulate discovery + fetch + parse flow, but keep run_keywords interface."""

    def __init__(self, scrapingbee_key: Optional[str] = None):
        self.scrapingbee_client = ScrapingBeeClient(api_key=scrapingbee_key)

    def is_linkedin_post_url(self, url: str) -> bool:
        if "linkedin.com" not in url:
            return False
        patterns = ["/posts/", "/feed/update/", "/activity:", "/activity/"]
        return any(p in url for p in patterns)

    def save_json(self, parsed: dict, folder: str, filename_base: str):
        os.makedirs(folder, exist_ok=True)
        fname = f"{filename_base}.json"
        path = os.path.join(folder, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=4)
        return path

    def load_master_df(self, path: str):
        if os.path.exists(path):
            try:
                return pd.read_excel(path, engine="openpyxl")
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def upsert_master(self, df: pd.DataFrame, parsed: dict) -> pd.DataFrame:
        # identical logic as your original helper — kept for compatibility
        row = {
            "url": parsed.get("url"),
            "title": parsed.get("title"),
            "author": parsed.get("author"),
            "content": parsed.get("content"),
            "likes": parsed.get("likes"),
            "comments": parsed.get("comments"),
            "date_published": parsed.get("date_published"),
            "images": json.dumps(parsed.get("images", []), ensure_ascii=False),
            "shared_url": parsed.get("shared_url"),
            "description": parsed.get("description"),
            "source_file": parsed.get("source_file")
        }
        if df is None or df.empty:
            return pd.DataFrame([row])
        if "url" in df.columns:
            idxs = df.index[df["url"] == row["url"]].tolist()
            if idxs:
                idx = idxs[0]
                for k, v in row.items():
                    df.at[idx, k] = v
                return df
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        return df

    def run_keywords(self, keywords: List[str], top_n_per_keyword: int = TOP_N_PER_KEYWORD, serpapi_key: str = None):
        all_urls: Set[str] = set()
        for kw in keywords:
            q = f'site:linkedin.com/posts "{kw}"'
            try:
                urls = serpapi_search(q, top=top_n_per_keyword, serpapi_key=serpapi_key)
            except Exception as e:
                print("SerpAPI search error:", e)
                urls = []
            time.sleep(BING_DELAY_SEC)
            for u in urls:
                if self.is_linkedin_post_url(u):
                    all_urls.add(u)

        print(f"Found {len(all_urls)} LinkedIn candidate URLs from keywords: {keywords}")

        master_df = self.load_master_df(MASTER_EXCEL)
        processed = []
        for url in sorted(all_urls):
            try:
                print("Fetching:", url)
                html = self.scrapingbee_client.fetch_html(url, render_js=True, save_path=None)
                parsed = parse_linkedin_html(html)
                
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
                            embed_html = self.scrapingbee_client.fetch_html(embed_url, render_js=True, timeout=30)
                            # instantiate parser class to use embed extractor
                            from linkedin_parser import LinkedInParser
                            parser = LinkedInParser()
                            embed_comments = parser._extract_comments_from_embed_html(embed_html)
                            if embed_comments:
                                parsed["comments_list"] = embed_comments
                                parsed["comments"] = parsed.get("comments") or len(embed_comments)
                        except Exception as e:
                            print("Embed comments fetch failed:", e)
                
                
                processed.append(parsed)
                base = url.replace("https://", "").replace("/", "_")
                self.save_json(parsed, PARSED_JSON_FOLDER, base)
                master_df = self.upsert_master(master_df, parsed)
                time.sleep(SCRAPINGBEE_DELAY_SEC)
            except Exception as e:
                print("Failed to fetch/parse:", url, e)

        with open(COMBINED_JSON, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=4)
        try:
            master_df.to_excel(MASTER_EXCEL, index=False, engine="openpyxl")
        except Exception:
            master_df.to_csv(MASTER_EXCEL.replace(".xlsx", ".csv"), index=False)
        print("Pipeline finished. Combined JSON and master updated.")


# CLI behaviour preserved:
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipeline_runner.py <keyword1> [keyword2] ...")
        sys.exit(1)
    prs = PipelineRunner()
    prs.run_keywords(sys.argv[1:])
