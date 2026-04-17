"""
Co-occurrence and semantic similarity networks for a target term in classical Chinese.
Nodes: content tokens from CLTK's lzh (Classical Chinese) model, frequency-filtered.
Co-occurrence edges: sentence-level co-occurrence counts.
Similarity edges: cosine similarity of sentence co-occurrence context vectors.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from cltk import NLP
from matplotlib import font_manager
from sklearn.metrics.pairwise import cosine_similarity
import textacy.representations.network as tnet
import textacy.viz.network as tviz

# --- font ---

font_manager.fontManager.addfont(
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP"]
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "sans-serif"]
# print('font: ' + matplotlib.rcParams["font.family"])

tviz.RC_PARAMS['font.family'] = ["Noto Sans CJK JP"]
tviz.RC_PARAMS["font.sans-serif"] = ["Noto Sans CJK JP", "sans-serif"]


# --- config ---

TEXT_PATH = Path("mengzi_6a.txt")
TERM = "仁"
MIN_FREQ = 3        # minimum total occurrences to be a node
SIM_THRESHOLD = 0.15  # minimum cosine similarity for a similarity edge
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

# --- helpers ---


def is_cjk(s: str) -> bool:
    return bool(s) and all("\u4e00" <= c <= "\u9fff" for c in s)


def proximity_score(G: nx.Graph, term: str, node: str) -> float:
    """Score a node by its weighted proximity to term (1-hop or best 2-hop path)."""
    if G.has_edge(term, node):
        return float(G[term][node]["weight"])
    return max(
        (G[term][mid]["weight"] * G[mid][node]["weight"]
         for mid in nx.common_neighbors(G, term, node)
         if G.has_edge(term, mid)),
        default=0.0,
    )


def prune_to_neighborhood(G: nx.Graph, term: str, max_nodes: int) -> nx.Graph:
    """Return a subgraph of G containing term and its top max_nodes neighbours."""
    if term not in G:
        return G
    ego = nx.ego_graph(G, term, radius=2)
    scored = sorted(
        (n for n in ego.nodes() if n != term),
        key=lambda n: proximity_score(ego, term, n),
        reverse=True,
    )
    keep = {term} | set(scored[:max_nodes])
    return G.subgraph(keep).copy()

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
nodes.add(TERM)  # always include target term

node_list = sorted(nodes)
idx: dict[str, int] = {n: i for i, n in enumerate(node_list)}
N = len(node_list)

print(f"Nodes: {N}  (min_freq={MIN_FREQ}, after stopword filter)")
print(
    f"Top nodes by freq: {[t for t, _ in freq.most_common(20) if t in nodes][:15]}")

# --- filter sentences to node-containing lists ---

sent_node_lists: list[list[str]] = [
    [t for t in sent if t in nodes]
    for sent in tokens_per_sent
]
sent_node_lists = [s for s in sent_node_lists if len(s) >= 2]
print(f"Sentences with ≥2 content nodes: {len(sent_node_lists)}")

# --- co-occurrence network ---

G = tnet.build_cooccurrence_network(sent_node_lists, window_size=100)
G.remove_nodes_from(list(nx.isolates(G)))
G = prune_to_neighborhood(G, TERM, MAX_NODES)

print(
    f"\nCo-occurrence — nodes: {G.number_of_nodes()}  edges: {G.number_of_edges()}")

cooc_json = Path(f"cooccurrence_{TERM}.json")
cooc_json.write_text(json.dumps(
    nx.node_link_data(G), indent=2), encoding="utf-8")
print(f"Saved {cooc_json}")

if G.number_of_nodes() >= 2:
    node_weights = tnet.rank_nodes_by_pagerank(G)
    ax = tviz.draw_semantic_network(
        G, node_weights=node_weights, spread=5.0,
        draw_nodes=True, base_node_size=500,
        node_alpha=0.2, line_width=1.0, line_alpha=0.25, base_font_size=10,
    )
    ax.set_title(f"Co-occurrence network — '{TERM}' highlighted", fontsize=14)
    plt.savefig(f"cooccurrence_{TERM}.png", bbox_inches="tight", dpi=150)
    plt.close()

# --- semantic similarity network ---
# Context vector of token X: how many sentences it shares with each other node.
# Edge weight = cosine similarity of context vectors.

M = np.zeros((N, N), dtype=float)
for sent in sent_node_lists:
    sent_nodes_in = [t for t in sent if t in idx]
    for a in sent_nodes_in:
        for b in sent_nodes_in:
            if a != b:
                M[idx[a], idx[b]] += 1.0

# cosine_similarity handles zero vectors (returns 0 similarity)
sim_matrix = cosine_similarity(M)

S = nx.Graph()
S.add_nodes_from(node_list)
for i, a in enumerate(node_list):
    for j, b in enumerate(node_list):
        if i < j and sim_matrix[i, j] >= SIM_THRESHOLD:
            S.add_edge(a, b, weight=float(sim_matrix[i, j]))

S.remove_nodes_from(list(nx.isolates(S)))
S = prune_to_neighborhood(S, TERM, MAX_NODES)

print(
    f"\nSimilarity — nodes: {S.number_of_nodes()}  edges: {S.number_of_edges()}")

sim_json = Path(f"similarity_{TERM}.json")
sim_json.write_text(json.dumps(
    nx.node_link_data(S), indent=2), encoding="utf-8")
print(f"Saved {sim_json}")

if S.number_of_nodes() >= 2:
    node_weights = tnet.rank_nodes_by_pagerank(S)
    max_weight = max(node_weights.values())
    edge_weights = [S[u][v]["weight"] for u, v in S.edges()]
    max_ew = max(edge_weights)

    pos = nx.spring_layout(
        S, seed=42, k=3.0 / (S.number_of_nodes() ** 0.5), weight="weight"
    )

    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    node_sizes = [
        3000 * (node_weights.get(n, 0) / max_weight) ** 0.5
        if n != TERM else 3000
        for n in S.nodes()
    ]
    node_colors = ["#e07b39" if n == TERM else "#6aaed6" for n in S.nodes()]

    nx.draw_networkx_nodes(S, pos, node_size=node_sizes,
                           node_color=node_colors, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(S, pos,
                           width=[3 * w / max_ew for w in edge_weights],
                           alpha=0.4, edge_color="#888888", ax=ax)
    for node, (x, y) in pos.items():
        ax.text(x, y, node,
                fontsize=11 if node == TERM else 8,
                fontweight="bold" if node == TERM else "normal",
                color="#c0440a" if node == TERM else "#333333",
                ha="center", va="center")

    ax.set_title(
        f"Semantic similarity network (cosine) — '{TERM}' highlighted", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"similarity_{TERM}.png", bbox_inches="tight", dpi=150)
    plt.close()
