from collections import Counter
from pathlib import Path

import spacy
from spacy_html_tokenizer import create_html_tokenizer
from bs4 import BeautifulSoup

# parse the main text of an SEP article

html_path = Path("sep/jonathan-edwards.html")
html = html_path.read_text(encoding="utf-8")

soup = BeautifulSoup(html, "html.parser")
article = soup.find("div", id="article-content")
article_html = str(article)

nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = create_html_tokenizer()(nlp)

doc = nlp(article_html)

# analyze noun chunks for a term

term = "benevolence"

chunks = [chunk.text.lower() for chunk in doc.noun_chunks if term in chunk.text.lower()]
counts = Counter(chunks)

print(f"Noun chunks containing '{term}' ({sum(counts.values())} total):\n")
for chunk, count in counts.most_common():
    print(f"  {count:3d}x  {chunk}")
