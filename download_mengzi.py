"""
Download the entire Mengzi from the ctext.org free API and save each
book as a plain-text file under text/ctext/.

Books and their ctext URNs:
  1A  梁惠王上  ctp:mengzi/liang-hui-wang-i
  1B  梁惠王下  ctp:mengzi/liang-hui-wang-ii
  2A  公孫丑上  ctp:mengzi/gongsun-chou-i
  2B  公孫丑下  ctp:mengzi/gongsun-chou-ii
  3A  滕文公上  ctp:mengzi/teng-wen-gong-i
  3B  滕文公下  ctp:mengzi/teng-wen-gong-ii
  4A  離婁上    ctp:mengzi/li-lou-i
  4B  離婁下    ctp:mengzi/li-lou-ii
  5A  萬章上    ctp:mengzi/wan-zhang-i
  5B  萬章下    ctp:mengzi/wan-zhang-ii
  6A  告子上    ctp:mengzi/gaozi-i
  6B  告子下    ctp:mengzi/gaozi-ii
  7A  盡心上    ctp:mengzi/jin-xin-i
  7B  盡心下    ctp:mengzi/jin-xin-ii
"""

import time

import requests

from config import CTEXT

API_BASE = "https://api.ctext.org"

BOOKS = [
    ("mengzi_1a", "ctp:mengzi/liang-hui-wang-i"),
    ("mengzi_1b", "ctp:mengzi/liang-hui-wang-ii"),
    ("mengzi_2a", "ctp:mengzi/gong-sun-chou-i"),
    ("mengzi_2b", "ctp:mengzi/gong-sun-chou-ii"),
    ("mengzi_3a", "ctp:mengzi/teng-wen-gong-i"),
    ("mengzi_3b", "ctp:mengzi/teng-wen-gong-ii"),
    ("mengzi_4a", "ctp:mengzi/li-lou-i"),
    ("mengzi_4b", "ctp:mengzi/li-lou-ii"),
    ("mengzi_5a", "ctp:mengzi/wan-zhang-i"),
    ("mengzi_5b", "ctp:mengzi/wan-zhang-ii"),
    ("mengzi_6a", "ctp:mengzi/gaozi-i"),
    ("mengzi_6b", "ctp:mengzi/gaozi-ii"),
    ("mengzi_7a", "ctp:mengzi/jin-xin-i"),
    ("mengzi_7b", "ctp:mengzi/jin-xin-ii"),
]


def fetch_text(urn: str) -> dict:
    r = requests.get(f"{API_BASE}/gettext", params={"urn": urn}, timeout=30)
    r.raise_for_status()
    return r.json()


for filename, urn in BOOKS:
    out_path = CTEXT / f"{filename}.txt"
    data = fetch_text(urn)

    title = data.get("title", "")
    paragraphs = data.get("fulltext", [])

    if not paragraphs:
        raise RuntimeError(f"No text returned for {urn}. Full response: {data}")

    full_text = "\n".join(p.strip() for p in paragraphs if p.strip())
    out_path.write_text(full_text, encoding="utf-8")

    print(f"{filename}  {title:8s}  paragraphs={len(paragraphs):3d}  chars={len(full_text):5d}  → {out_path}")

    # be polite to the free API
    time.sleep(1)

print("\nDone.")
