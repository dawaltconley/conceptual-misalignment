"""
Download Mengzi Book 6A (告子上, Gaozi I) from the ctext.org free API
and save it as a plain-text file suitable for parse_classical_chinese.py.
"""

from pathlib import Path

import requests

API_BASE = "https://api.ctext.org"
URN = "ctp:mengzi/gaozi-i"
OUT_PATH = Path("mengzi_6a.txt")


def fetch_text(urn: str) -> dict:
    r = requests.get(f"{API_BASE}/gettext", params={"urn": urn}, timeout=30)
    r.raise_for_status()
    return r.json()


data = fetch_text(URN)

title = data.get("title", "")
paragraphs = data.get("fulltext", [])

if not paragraphs:
    raise RuntimeError(f"No text returned. Full response: {data}")

# join paragraphs with a sentence-final marker so parse_classical_chinese.py
# can split on 。 for sentence segmentation
full_text = "\n".join(p.strip() for p in paragraphs if p.strip())

OUT_PATH.write_text(full_text, encoding="utf-8")

print(f"Title : {title}")
print(f"Paragraphs : {len(paragraphs)}")
print(f"Characters : {len(full_text)}")
print(f"Saved to   : {OUT_PATH}")
