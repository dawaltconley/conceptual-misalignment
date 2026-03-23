from collections import Counter
from pathlib import Path

import spacy
import textacy.extract
import textacy.extract.keyterms as kt
import textacy.extract.triples as triples
from bs4 import BeautifulSoup
from spacy_html_tokenizer import create_html_tokenizer

# parse the main text of an SEP article

html_path = Path("sep/jonathan-edwards.html")
html = html_path.read_text(encoding="utf-8")

soup = BeautifulSoup(html, "html.parser")
article = {
    "preamble": soup.find("div", id="preamble"),
    "toc": soup.find("div", id="toc"),
    "main-text": soup.find("div", id="main-text"),
}
article_html = str(article["preamble"]) + str(article["toc"])+ str(article["main-text"])

nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = create_html_tokenizer()(nlp)

doc = nlp(article_html)

term = "benevolence"

# --- noun chunks ---

print(f"{'='*60}")
print(f"NOUN CHUNKS containing '{term}'")
print(f"{'='*60}\n")

chunks = [chunk.text.lower() for chunk in doc.noun_chunks if term in chunk.text.lower()]
counts = Counter(chunks)

print(f"{sum(counts.values())} total occurrences, {len(counts)} distinct chunks\n")
for chunk, count in counts.most_common():
    print(f"  {count:3d}x  {chunk}")

# --- KWIC ---

print(f"\n{'='*60}")
print(f"KEYWORD IN CONTEXT: '{term}'")
print(f"{'='*60}\n")

matches = list(textacy.extract.kwic.keyword_in_context(doc, term, window_width=60, pad_context=True))
for pre, kw, post in matches:
    print(f"...{pre.strip()} [{kw}] {post.strip()}...")

# --- key terms (TextRank) ---

print(f"\n{'='*60}")
print(f"TOP KEY TERMS (TextRank)")
print(f"{'='*60}\n")

keyterms = kt.textrank(doc, normalize="lower", topn=0.1)
for phrase, score in filter(lambda kt : term in kt[0], keyterms):
    print(f"  {score:.4f}  {phrase}")

# --- subject/verb/object triples ---

print(f"\n{'='*60}")
print(f"SVO TRIPLES containing '{term}'")
print(f"{'='*60}\n")

svos = list(triples.subject_verb_object_triples(doc))

def spans_text(spans):
    return " ".join(s.text for s in spans)

hits = [
    t for t in svos
    if term in spans_text(t.subject).lower() or term in spans_text(t.object).lower()
]
if hits:
    for t in hits:
        print(f"  SUBJ: {spans_text(t.subject)}")
        print(f"  VERB: {spans_text(t.verb)}")
        print(f"   OBJ: {spans_text(t.object)}")
        print()
else:
    print("  (no SVO triples found containing this term)")
