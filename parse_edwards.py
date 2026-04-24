from collections import Counter

from config import DIST, SEP
from network_utils import (
    build_cosine_similarity_graph,
    build_pmi_graph,
    draw_term_network,
    prune_to_neighborhood,
    save_graph_json,
)

import spacy
import networkx as nx
import textacy.extract
import textacy.extract.keyterms as kt
import textacy.extract.triples as triples
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
MAX_NODES = 15
MIN_FREQ = 22
SIM_THRESHOLD = 0.7

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

# --- co-occurrence network (individual words, PMI-weighted, pruned to neighborhood) ---

print(f"\n{'='*60}")
print(f"CO-OCCURRENCE NETWORK (PMI-weighted)")
print(f"{'='*60}\n")

# lemmatize and filter: content words only, no stopwords, no punctuation
CONTENT_POS = {"NOUN", "VERB", "ADJ", "PROPN"}
sent_token_lists = [
    [
        token.lemma_.lower()
        for token in sent
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and token.is_alpha
        and token.pos_ in CONTENT_POS
    ]
    for sent in doc.sents
]

freq = Counter(tok for sent in sent_token_lists for tok in sent)
nodes: set[str] = {t for t, c in freq.items() if c >= MIN_FREQ}
nodes.add(term)
node_list = sorted(nodes)

print(f"Nodes: {len(nodes)}  (min_freq={MIN_FREQ}, after spaCy stopword filter)")
print(
    f"Top nodes by freq: {[t for t, _ in freq.most_common(20) if t in nodes][:15]}")

sent_node_lists = [
    [t for t in sent if t in nodes]
    for sent in sent_token_lists
]
sent_node_lists = [s for s in sent_node_lists if len(s) >= 2]
print(f"Sentences with ≥2 content nodes: {len(sent_node_lists)}")

G, _, _, _ = build_pmi_graph(sent_node_lists, nodes)
G.remove_nodes_from(list(nx.isolates(G)))
G = prune_to_neighborhood(G, term, MAX_NODES)

print(
    f"Co-occurrence — nodes: {G.number_of_nodes()}  edges: {G.number_of_edges()}")

out_path = DIST / f"cooccurrence_{term}.png"
draw_term_network(
    G, term,
    title=f"Word co-occurrence network (PMI) — '{term}' highlighted",
    out_path=out_path,
)
print(f"Network saved to {out_path}")

json_path = DIST / f"cooccurrence_{term}.json"
save_graph_json(G, json_path)
print(f"Network JSON saved to {json_path}")

# --- semantic similarity network (cosine, pruned to neighborhood) ---

print(f"\n{'='*60}")
print(f"SEMANTIC SIMILARITY NETWORK (cosine)")
print(f"{'='*60}\n")

S = build_cosine_similarity_graph(sent_node_lists, node_list, SIM_THRESHOLD)
S.remove_nodes_from(list(nx.isolates(S)))
S = prune_to_neighborhood(S, term, MAX_NODES)

print(
    f"Similarity — nodes: {S.number_of_nodes()}  edges: {S.number_of_edges()}")

out_path = DIST / f"similarity_{term}.png"
draw_term_network(
    S, term,
    title=f"Word similarity network (cosine) — '{term}' highlighted",
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
