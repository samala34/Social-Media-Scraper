# File: batch_parser.py  (suggested replacement for linkedin_batch_parse_and_save.py)
"""
batch_parser.py
---------------
Batch parser for offline HTML files. Keeps same behaviour as linkedin_batch_parse_and_save.py
but encapsulates the logic in a BatchParser class for readability and testing.
"""

import os
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from bs4 import BeautifulSoup
import pandas as pd

# Use the parser wrapper so we get class-based implementation
from linkedin_parser import parse_linkedin_html  # keep compatibility (or import parse_linkedin_post.parse_linkedin_html)

HTML_FOLDER = "html_pages"
OUTPUT_JSON_FOLDER = "parsed_jsons"
MASTER_EXCEL = "linkedin_posts_master.xlsx"
COMBINED_JSON = "all_posts_combined.json"

os.makedirs(OUTPUT_JSON_FOLDER, exist_ok=True)


def parse_short_number(s: Union[str, int, None]) -> Optional[int]:
    # Keep the same helper — small duplication ok for clarity
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s).strip().replace(",", "").replace("\u00A0", "")
    if s == "":
        return None
    m = re.match(r"^([\d.]+)([KkMm]?)$", s)
    if m:
        num = float(m.group(1))
        suf = m.group(2).upper()
        if suf == "K":
            return int(num * 1_000)
        if suf == "M":
            return int(num * 1_000_000)
        return int(num)
    m2 = re.search(r"(\d[\d,]*)", s)
    if m2:
        return int(m2.group(1).replace(",", ""))
    return None


class BatchParser:
    """Encapsulates offline parsing/upsert behaviour for saved LinkedIn HTML files."""

    def __init__(self, html_folder: str = HTML_FOLDER, out_json_folder: str = OUTPUT_JSON_FOLDER,
                 master_excel: str = MASTER_EXCEL, combined_json: str = COMBINED_JSON):
        self.html_folder = html_folder
        self.out_json_folder = out_json_folder
        self.master_excel = master_excel
        self.combined_json = combined_json
        os.makedirs(self.out_json_folder, exist_ok=True)

    def _list_html_files(self) -> List[str]:
        if not os.path.exists(self.html_folder):
            return []
        return [f for f in os.listdir(self.html_folder) if f.lower().endswith(".html")]

    def _save_parsed(self, parsed: Dict[str, Any], filename_base: str):
        fname = f"{filename_base}.json"
        path = os.path.join(self.out_json_folder, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=4)
        return path

    def _load_master(self) -> pd.DataFrame:
        if os.path.exists(self.master_excel):
            try:
                return pd.read_excel(self.master_excel, engine="openpyxl")
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def _upsert_master(self, df: pd.DataFrame, parsed: Dict[str, Any]) -> pd.DataFrame:
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

    def run(self):
        files = self._list_html_files()
        combined = []
        master_df = self._load_master()

        for fname in files:
            path = os.path.join(self.html_folder, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    html = f.read()
                parsed = parse_linkedin_html(html, source_filename=fname)
                combined.append(parsed)
                base = os.path.splitext(fname)[0]
                self._save_parsed(parsed, base)
                master_df = self._upsert_master(master_df, parsed)
            except Exception as e:
                print(f"Failed to parse {fname}: {e}")

        # write combined json
        with open(self.combined_json, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=4)
        # write master
        try:
            master_df.to_excel(self.master_excel, index=False, engine="openpyxl")
        except Exception:
            master_df.to_csv(self.master_excel.replace(".xlsx", ".csv"), index=False)
        print(f"Parsed {len(combined)} files. Combined: {self.combined_json}. Master: {self.master_excel}")


# CLI-friendly behaviour: keep script runnable
if __name__ == "__main__":
    bp = BatchParser()
    bp.run()
