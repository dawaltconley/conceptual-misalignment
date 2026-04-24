"""
Reproduces the co-occurrence and similarity networks from parse_edwards.py and
networks_mengzi_ren.py using the shared functions in cooccurrence_html.py.
Outputs are saved with a '_new' suffix so they can be diffed against the originals.
"""

from config import DIST, SEP, CTEXT
from network_utils import save_graph_json
from cooccurrence_html import (
    build_cooccurrence_network,
    build_similarity_network,
    filter_to_sent_node_lists,
    parse_sep_article,
    tokenize_classical_chinese,
    tokenize_english_html,
)

# ---------------------------------------------------------------------------
# Edwards — "benevolence" in the SEP article on Jonathan Edwards
# ---------------------------------------------------------------------------

EDWARDS_TERM = "benevolence"
EDWARDS_MIN_FREQ = 22
EDWARDS_MAX_NODES = 15
EDWARDS_SIM_THRESHOLD = 0.7

print("=== Edwards ===")

edwards_html = parse_sep_article(
    (SEP / "jonathan-edwards.html").read_text(encoding="utf-8")
)
edwards_tokens = tokenize_english_html(edwards_html)
edwards_sent_node_lists, edwards_nodes = filter_to_sent_node_lists(
    edwards_tokens, EDWARDS_TERM, EDWARDS_MIN_FREQ
)

print(f"Nodes: {len(edwards_nodes)}  sentences with ≥2 nodes: {len(edwards_sent_node_lists)}")

G_edwards = build_cooccurrence_network(
    edwards_sent_node_lists, edwards_nodes, EDWARDS_TERM, EDWARDS_MAX_NODES
)
print(f"Co-occurrence — nodes: {G_edwards.number_of_nodes()}  edges: {G_edwards.number_of_edges()}")
save_graph_json(G_edwards, DIST / f"cooccurrence_{EDWARDS_TERM}_new.json")

S_edwards = build_similarity_network(
    edwards_sent_node_lists, edwards_nodes, EDWARDS_TERM, EDWARDS_MAX_NODES, EDWARDS_SIM_THRESHOLD
)
print(f"Similarity    — nodes: {S_edwards.number_of_nodes()}  edges: {S_edwards.number_of_edges()}")
save_graph_json(S_edwards, DIST / f"similarity_{EDWARDS_TERM}_new.json")

# ---------------------------------------------------------------------------
# Mengzi — 仁 (rén) in Mengzi 6A
# ---------------------------------------------------------------------------

MENGZI_TERM = "仁"
MENGZI_MIN_FREQ = 3
MENGZI_MAX_NODES = 15
MENGZI_SIM_THRESHOLD = 0.5

MENGZI_STOPWORDS: set[str] = {
    "之", "也", "乎", "矣", "焉", "哉", "邪", "耳", "已",
    "而", "則", "以", "且", "雖", "若", "如", "猶", "亦", "故", "乃", "夫",
    "我", "吾", "汝", "其", "此", "彼", "是",
    "有", "無", "為", "曰", "謂", "不", "非", "所", "者", "於", "豈",
    "然", "得", "能", "可", "將", "及", "皆", "未", "與",
}

print("\n=== Mengzi ===")

mengzi_tokens = tokenize_classical_chinese(
    (CTEXT / "mengzi_6a.txt").read_text(encoding="utf-8")
)
mengzi_sent_node_lists, mengzi_nodes = filter_to_sent_node_lists(
    mengzi_tokens, MENGZI_TERM, MENGZI_MIN_FREQ, MENGZI_STOPWORDS
)

print(f"Nodes: {len(mengzi_nodes)}  sentences with ≥2 nodes: {len(mengzi_sent_node_lists)}")

G_mengzi = build_cooccurrence_network(
    mengzi_sent_node_lists, mengzi_nodes, MENGZI_TERM, MENGZI_MAX_NODES
)
print(f"Co-occurrence — nodes: {G_mengzi.number_of_nodes()}  edges: {G_mengzi.number_of_edges()}")
save_graph_json(G_mengzi, DIST / f"cooccurrence_{MENGZI_TERM}_new.json")

S_mengzi = build_similarity_network(
    mengzi_sent_node_lists, mengzi_nodes, MENGZI_TERM, MENGZI_MAX_NODES, MENGZI_SIM_THRESHOLD
)
print(f"Similarity    — nodes: {S_mengzi.number_of_nodes()}  edges: {S_mengzi.number_of_edges()}")
save_graph_json(S_mengzi, DIST / f"similarity_{MENGZI_TERM}_new.json")
