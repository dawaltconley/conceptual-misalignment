"""
Co-occurrence and semantic similarity networks for a target term in classical Chinese.
Nodes: content tokens from CLTK's lzh (Classical Chinese) model, frequency-filtered.
Co-occurrence edges: positive PMI (terms that appear together more than chance predicts).
Similarity edges: cosine similarity of sentence co-occurrence context vectors.
"""

from collections import Counter, defaultdict

from config import DIST, CTEXT
from network_utils import (
    build_cosine_similarity_graph,
    build_pmi_graph,
    draw_term_network,
    is_cjk,
    pmi,
    prune_to_neighborhood,
    save_graph_json,
)

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from cltk import NLP
from matplotlib import font_manager
import textacy.representations.network as tnet
import textacy.viz.network as tviz

# --- font ---

font_manager.fontManager.addfont(
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP"]
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "sans-serif"]

tviz.RC_PARAMS['font.family'] = ["Noto Sans CJK JP"]
tviz.RC_PARAMS["font.sans-serif"] = ["Noto Sans CJK JP", "sans-serif"]

# --- config ---

TEXT_PATH = CTEXT / "mengzi_6a.txt"
TERM = "仁"
MIN_FREQ = 3        # minimum total occurrences to be a node
SIM_THRESHOLD = 0.5  # minimum cosine similarity for a similarity edge
MAX_NODES = 15      # max non-TERM nodes to keep (1-hop first, then 2-hop)

STOPWORDS: set[str] = {
    # sentence-final and modal particles
    "之", "也", "乎", "矣", "焉", "哉", "邪", "耳", "已",
    # connectives and adverbs
    "而", "則", "以", "且", "雖", "若", "如", "猶", "亦", "故", "乃", "夫",
    # pronouns and demonstratives
    "我", "吾", "汝", "其", "此", "彼", "是",
    # auxiliaries, negation, prepositions
    "有", "無", "為", "曰", "謂", "不", "非", "所", "者", "於", "豈",
    # other high-frequency function words
    "然", "得", "能", "可", "將", "及", "皆", "未", "與",
}

# --- load and tokenize ---

raw_text = TEXT_PATH.read_text(encoding="utf-8")

cltk_nlp = NLP(language_code="lzh")
doc = cltk_nlp.analyze(text=raw_text)

# Group CJK tokens by sentence using CLTK's index_sentence attribute
sent_tokens: dict[int, list[str]] = defaultdict(list)
for w in doc.words:
    if w.string and w.index_sentence and is_cjk(w.string):
        sent_tokens[w.index_sentence].append(w.string)

tokens_per_sent: list[list[str]] = [
    sent_tokens[i] for i in sorted(sent_tokens.keys())
]

# --- build node set ---

freq = Counter(tok for sent in tokens_per_sent for tok in sent)
nodes: set[str] = {
    t for t, c in freq.items()
    if c >= MIN_FREQ and t not in STOPWORDS
}
nodes.add(TERM)

node_list = sorted(nodes)

print(f"Nodes: {len(nodes)}  (min_freq={MIN_FREQ}, after stopword filter)")
print(f"Top nodes by freq: {[t for t, _ in freq.most_common(20) if t in nodes][:15]}")

# --- filter sentences to node-containing lists ---

sent_node_lists: list[list[str]] = [
    [t for t in sent if t in nodes]
    for sent in tokens_per_sent
]
sent_node_lists = [s for s in sent_node_lists if len(s) >= 2]
print(f"Sentences with ≥2 content nodes: {len(sent_node_lists)}")

# --- co-occurrence network (PMI-weighted) ---

G, pair_cooc, sent_freq, N_sents = build_pmi_graph(sent_node_lists, nodes)
G.remove_nodes_from(list(nx.isolates(G)))
G = prune_to_neighborhood(G, TERM, MAX_NODES)

# --- PMI of top-20 most frequent characters relative to TERM ---

print(f"\n{'='*60}")
print(f"PMI of 20 most-frequent characters relative to '{TERM}'")
print(f"{'='*60}")
print(f"  {'char':>4}  {'freq':>5}  {'pmi':>8}")
print(f"  {'-'*4}  {'-'*5}  {'-'*8}")
for tok, tok_freq in [(t, c) for t, c in freq.most_common() if t not in STOPWORDS][:20]:
    pmi_val = pmi(pair_cooc, sent_freq, N_sents, tok, TERM)
    pmi_str = "   -inf" if pmi_val == float("-inf") else f"{pmi_val:8.3f}"
    print(f"  {tok:>4}  {tok_freq:>5}  {pmi_str}")

print(f"\nCo-occurrence — nodes: {G.number_of_nodes()}  edges: {G.number_of_edges()}")

cooc_json = DIST / f"cooccurrence_{TERM}.json"
save_graph_json(G, cooc_json)
print(f"Saved {cooc_json}")

if G.number_of_nodes() >= 2:
    node_weights = tnet.rank_nodes_by_pagerank(G)
    ax = tviz.draw_semantic_network(
        G, node_weights=node_weights, spread=5.0,
        draw_nodes=True, base_node_size=500,
        node_alpha=0.2, line_width=1.0, line_alpha=0.25, base_font_size=10,
    )
    ax.set_title(f"Co-occurrence network — '{TERM}' highlighted", fontsize=14)
    plt.savefig(DIST / f"cooccurrence_{TERM}.png", bbox_inches="tight", dpi=150)
    plt.close()

# --- semantic similarity network ---

S = build_cosine_similarity_graph(sent_node_lists, node_list, SIM_THRESHOLD)
S.remove_nodes_from(list(nx.isolates(S)))
S = prune_to_neighborhood(S, TERM, MAX_NODES)

print(f"\nSimilarity — nodes: {S.number_of_nodes()}  edges: {S.number_of_edges()}")

sim_json = DIST / f"similarity_{TERM}.json"
save_graph_json(S, sim_json)
print(f"Saved {sim_json}")

draw_term_network(
    S, TERM,
    title=f"Semantic similarity network (cosine) — '{TERM}' highlighted",
    out_path=DIST / f"similarity_{TERM}.png",
)
