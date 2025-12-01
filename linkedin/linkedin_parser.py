"""
linkedin_parser.py
------------------
LinkedIn HTML -> structured dict parser encapsulated in a class.
Provides parse_linkedin_html(html, source_filename=None) wrapper for compatibility.
"""

from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

class LinkedInParser:
    """Parser that extracts structured metadata from a LinkedIn post HTML document."""

    def __init__(self):
        pass

    @staticmethod
    def parse_short_number(s: Union[str, int, None]) -> Optional[int]:
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

    @staticmethod
    def _extract_jsonld(soup: BeautifulSoup) -> List[Dict[str, Any]]:
        results = []
        for script in soup.find_all("script", {"type": "application/ld+json"}):
            txt = script.string or script.get_text() or ""
            txt = txt.strip()
            if not txt:
                continue
            try:
                parsed = json.loads(txt)
                if isinstance(parsed, list):
                    results.extend(parsed)
                else:
                    results.append(parsed)
            except Exception:
                # attempt to split concatenated JSON objects
                parts = re.split(r"\}\s*\{", txt)
                if len(parts) > 1:
                    for i, p in enumerate(parts):
                        if i == 0:
                            candidate = p + "}"
                        elif i == len(parts) - 1:
                            candidate = "{" + p
                        else:
                            candidate = "{" + p + "}"
                        try:
                            results.append(json.loads(candidate))
                        except Exception:
                            continue
        return results
    
        # ------------------ comments extraction helpers ------------------
    def _extract_comments_from_jsonld(self, posting_obj):
        """
        Extract comment list if present in JSON-LD (often under 'comment' or 'commentList').
        Returns list of dicts: [{author, text, date_published, likes}, ...]
        """
        comments_out = []
        if not posting_obj:
            return comments_out

        raw_comments = posting_obj.get("comment") or posting_obj.get("comments") or posting_obj.get("commentList")
        if isinstance(raw_comments, dict):
            raw_comments = [raw_comments]
        if isinstance(raw_comments, list):
            for c in raw_comments:
                try:
                    author = None
                    auth = c.get("author") or c.get("creator")
                    if isinstance(auth, dict):
                        author = auth.get("name")
                    elif isinstance(auth, str):
                        author = auth
                    text = c.get("text") or c.get("description") or c.get("commentText") or ""
                    date = c.get("datePublished") or c.get("dateCreated") or c.get("uploadDate")
                    likes = None
                    stats = c.get("interactionStatistic")
                    if isinstance(stats, dict):
                        likes = stats.get("userInteractionCount")
                    comments_out.append({
                        "author": author,
                        "text": text,
                        "date_published": date,
                        "likes": int(likes) if likes is not None else None
                    })
                except Exception:
                    continue
        return comments_out


    def _extract_comments_from_dom(self, soup):
        """
        Heuristic DOM fallback to extract visible comments from the rendered HTML.
        Conservative selectors to avoid false positives.
        """
        comments = []
        try:
            selectors = [
                '[data-test-id="comments-list"]',
                '[data-test-comments-list]',
                '.comments-comment-item',
                '.feed-shared-comments-list__comment-item',
                '.comments-comment',
                '.feed-shared-comment'  # additional
            ]
            for sel in selectors:
                nodes = soup.select(sel)
                if nodes:
                    for node in nodes:
                        author = None
                        text = None
                        date = None
                        a = node.select_one('span > span[dir="ltr"], .feed-shared-actor__name, .comment-author, .feed-shared-comment__name')
                        if a:
                            author = a.get_text(strip=True)
                        t = node.select_one('span[dir="ltr"], p, .comment-text, .feed-shared-text__text-view')
                        if t:
                            text = t.get_text(separator=" ", strip=True)
                        tm = node.select_one('time, span.feed-shared-actor__sub-description, .comment-date, .visually-hidden')
                        if tm:
                            date = tm.get('datetime') or tm.get_text(strip=True)
                        if text or author:
                            comments.append({"author": author, "text": text, "date_published": date, "likes": None})
                    if comments:
                        return comments
        except Exception:
            pass
        return comments


    def _extract_comments_from_embed_html(self, embed_html: str):
        """
        Parse an embed HTML for comments. Re-uses the DOM heuristic.
        """
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(embed_html, "html.parser")
        return self._extract_comments_from_dom(soup)


    def parse(self, html: str, source_filename: Optional[str] = None) -> Dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        out = {
            "url": None,
            "title": None,
            "description": None,
            "content": None,
            "likes": None,
            "comments": None,
            "author": None,
            "date_published": None,
            "images": [],
            "shared_url": None,
            "raw_jsonld": [],
            "source_file": source_filename
        }

        jsonlds = LinkedInParser._extract_jsonld(soup)
        out["raw_jsonld"] = jsonlds

        posting = None
        for obj in jsonlds:
            if not isinstance(obj, dict):
                continue
            typ = obj.get("@type") or obj.get("type") or ""
            if isinstance(typ, list):
                typ = " ".join(typ)
            if "SocialMediaPosting" in str(typ) or "VideoObject" in str(typ) or "SocialMediaPosting" == typ:
                posting = obj
                break
            if "articleBody" in obj or "interactionStatistic" in obj:
                posting = obj
                break

        if posting:
            out["content"] = posting.get("articleBody") or posting.get("headline") or out["content"]
            out["title"] = posting.get("headline") or posting.get("name") or out["title"]
            out["description"] = posting.get("description") or out["description"]

            auth = posting.get("author") or posting.get("creator") or posting.get("publisher")
            if isinstance(auth, dict):
                out["author"] = auth.get("name") or auth.get("url")
            elif isinstance(auth, str):
                out["author"] = auth
            elif isinstance(auth, list) and auth:
                a0 = auth[0]
                if isinstance(a0, dict):
                    out["author"] = a0.get("name")

            out["date_published"] = posting.get("datePublished") or posting.get("uploadDate") or out["date_published"]

            for key in ("thumbnailUrl", "image", "thumbnail", "thumbnailUrl", "thumbnailImage"):
                v = posting.get(key)
                if isinstance(v, str) and v not in out["images"]:
                    out["images"].append(v)
                elif isinstance(v, dict):
                    url = v.get("url")
                    if url and url not in out["images"]:
                        out["images"].append(url)

            shared = posting.get("sharedContent")
            if isinstance(shared, dict):
                out["shared_url"] = shared.get("url") or out["shared_url"]
                si = shared.get("image") or shared.get("thumbnail")
                if isinstance(si, dict):
                    url = si.get("url")
                    if url and url not in out["images"]:
                        out["images"].append(url)

            stats = posting.get("interactionStatistic") or posting.get("interactionStatistics")
            if isinstance(stats, dict):
                stats = [stats]
            if isinstance(stats, list):
                for s in stats:
                    it = str(s.get("interactionType", "")).lower()
                    cnt = s.get("userInteractionCount")
                    try:
                        cntv = int(cnt) if cnt is not None else None
                    except Exception:
                        cntv = LinkedInParser.parse_short_number(cnt)
                    if "like" in it:
                        out["likes"] = cntv
                    if "comment" in it:
                        out["comments"] = cntv
                    if "share" in it or "resha" in it:
                        out["reposts"] = cntv

        # fallback meta tags and canonical link
        can = soup.find("link", rel="canonical")
        if can and can.get("href"):
            out["url"] = out.get("url") or can["href"]

        # og/meta fallback
        og_title = None
        try:
            og_title = soup.find("meta", property="og:title")
        except Exception:
            og_title = None
        if og_title and og_title.get("content"):
            out["title"] = out["title"] or og_title["content"]
        meta_desc = None
        try:
            meta_desc = soup.find("meta", attrs={"name": "description"})
        except Exception:
            meta_desc = None
        if meta_desc and meta_desc.get("content"):
            out["description"] = out["description"] or meta_desc["content"]

        # normalize numeric fields
        for key in ("likes", "comments", "reposts"):
            if key in out and out[key] is None:
                out[key] = 0
                
                # -----------------------------------------------------------------
        # COMMENTS: attempt JSON-LD -> DOM fallback -> embedUrl (only flagged)
        # -----------------------------------------------------------------
        comments_list = []

        # 1) try JSON-LD-based comments
        if posting:
            try:
                comments_list = self._extract_comments_from_jsonld(posting) or []
            except Exception:
                comments_list = []

        # 2) if not found, try DOM heuristics
        if not comments_list:
            try:
                comments_list = self._extract_comments_from_dom(soup) or []
            except Exception:
                comments_list = []

        # 3) attach to output (numeric 'comments' stays for compatibility)
        out["comments_list"] = comments_list
        if out.get("comments") is None:
            out["comments"] = len(comments_list) if comments_list else 0


        return out


# Backwards-compatible function for existing imports:
_default_parser = LinkedInParser()

def parse_linkedin_html(html: str, source_filename: Optional[str] = None) -> Dict[str, Any]:
    """Compatibility wrapper to preserve original function name used by other scripts."""
    return _default_parser.parse(html, source_filename=source_filename)
