from collections import Counter

from config import DIST, SEP
from network_utils import draw_term_network, save_graph_json

import spacy
import matplotlib.pyplot as plt
import networkx as nx
import textacy.extract
import textacy.extract.keyterms as kt
import textacy.extract.triples as triples
import textacy.representations.network as tnet
import textacy.viz.network as tviz
from bs4 import BeautifulSoup
from spacy_html_tokenizer import create_html_tokenizer

# parse the main text of an SEP article

html_path = SEP / "jonathan-edwards.html"
html = html_path.read_text(encoding="utf-8")

soup = BeautifulSoup(html, "html.parser")
article = {
    "preamble": soup.find("div", id="preamble"),
    "toc": soup.find("div", id="toc"),
    "main-text": soup.find("div", id="main-text"),
}
article_html = str(article["preamble"]) + \
    str(article["toc"]) + str(article["main-text"])

nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = create_html_tokenizer()(nlp)

doc = nlp(article_html)

term = "benevolence"

# --- noun chunks ---

print(f"{'='*60}")
print(f"NOUN CHUNKS containing '{term}'")
print(f"{'='*60}\n")

chunks = [chunk.text.lower()
          for chunk in doc.noun_chunks if term in chunk.text.lower()]
counts = Counter(chunks)

print(f"{sum(counts.values())} total occurrences, {len(counts)} distinct chunks\n")
for chunk, count in counts.most_common():
    print(f"  {count:3d}x  {chunk}")

# --- KWIC ---

print(f"\n{'='*60}")
print(f"KEYWORD IN CONTEXT: '{term}'")
print(f"{'='*60}\n")

matches = list(textacy.extract.kwic.keyword_in_context(
    doc, term, ignore_case=True, window_width=60, pad_context=True))
for pre, kw, post in matches:
    print(f"...{pre.strip()} [{kw}] {post.strip()}...")

# --- key terms (TextRank) ---

print(f"\n{'='*60}")
print(f"TOP KEY TERMS (TextRank)")
print(f"{'='*60}\n")

keyterms = kt.textrank(doc, normalize="lower", topn=0.1)
for phrase, score in filter(lambda kt: term in kt[0], keyterms):
    print(f"  {score:.4f}  {phrase}")

# --- keyterm co-occurrence in KWIC windows ---

print(f"\n{'='*60}")
print(f"KEYTERMS CO-OCCURRING WITH '{term}' IN CONTEXT WINDOWS")
print(f"{'='*60}\n")

all_keyterms = kt.textrank(doc, normalize="lower", topn=1.0)
keyterm_phrases = [phrase for phrase, _ in all_keyterms]

cooccurrences = Counter()
for pre, kw, post in matches:
    window = (pre + post).lower()
    for phrase in keyterm_phrases:
        if phrase != term and phrase in window:
            cooccurrences[phrase] += 1

for phrase, count in cooccurrences.most_common(20):
    if count > 1:
        print(f"  {count:3d}x  {phrase}")

# --- co-occurrence network (whole document, target term highlighted) ---

print(f"\n{'='*60}")
print(f"CO-OCCURRENCE NETWORK (whole document)")
print(f"{'='*60}\n")

# build per-sentence keyterm lists as input to the network builder
top_keyterms = {phrase for phrase, _ in kt.textrank(
    doc, normalize="lower", topn=50)}
top_keyterms.add(term)

sent_keyterm_lists = [
    [p for p in top_keyterms if p in sent.text.lower()]
    for sent in doc.sents
]
sent_keyterm_lists = [s for s in sent_keyterm_lists if len(s) >= 2]

G = tnet.build_cooccurrence_network(sent_keyterm_lists, window_size=100)
G.remove_nodes_from(list(nx.isolates(G)))

node_weights = tnet.rank_nodes_by_pagerank(G)

ax = tviz.draw_semantic_network(
    G,
    node_weights=node_weights,
    spread=5.0,
    draw_nodes=True,
    base_node_size=500,
    node_alpha=0.2,
    line_width=1.0,
    line_alpha=0.25,
    base_font_size=10,
)

ax.set_title(
    f"Keyterm co-occurrence network — '{term}' highlighted", fontsize=14)
out_path = DIST / f"cooccurrence_{term}.png"
plt.savefig(out_path, bbox_inches="tight", dpi=150)
print(f"Network saved to {out_path}")

json_path = DIST / f"cooccurrence_{term}.json"
save_graph_json(G, json_path)
print(f"Network JSON saved to {json_path}")

# --- semantic similarity network (whole document, target term highlighted) ---

print(f"\n{'='*60}")
print(f"SEMANTIC SIMILARITY NETWORK (whole document)")
print(f"{'='*60}\n")

# tokenize each keyterm phrase for jaccard similarity
tokenized_keyterms = [tuple(phrase.split()) for phrase in sorted(top_keyterms)]

S = tnet.build_similarity_network(tokenized_keyterms, edge_weighting="jaccard")

# prune very-low-similarity edges to keep graph readable
low_edges = [(u, v) for u, v, d in S.edges(data=True) if d["weight"] < 0.2]
S.remove_edges_from(low_edges)
S.remove_nodes_from(list(nx.isolates(S)))

# use phrase string as node label (nodes are tuples after build_similarity_network)
label_map = {node: " ".join(node) for node in S.nodes()}
S = nx.relabel_nodes(S, label_map)

out_path = DIST / f"similarity_{term}.png"
draw_term_network(
    S, term,
    title=f"Keyterm similarity network (Jaccard) — '{term}' highlighted",
    out_path=out_path,
)
print(f"Network saved to {out_path}")

json_path = DIST / f"similarity_{term}.json"
save_graph_json(S, json_path)
print(f"Network JSON saved to {json_path}")

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
