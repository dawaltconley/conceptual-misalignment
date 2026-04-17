import json
import re
from collections import Counter

import matplotlib
from config import DIST, CTEXT
import matplotlib.pyplot as plt
import networkx as nx
import textacy.representations.network as tnet
import textacy.viz.network as tviz
from cltk import NLP
from matplotlib import font_manager

font_manager.fontManager.addfont(
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
_cjk_prop = font_manager.FontProperties(
    fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
matplotlib.rcParams["font.family"] = _cjk_prop.get_name()
print('font: ' + matplotlib.rcParams["font.family"])

# --- config ---

text_path = CTEXT / "mengzi_6a.txt"
term = "仁"
top_n = 50          # keyterms to include in networks
yake_total = 200    # how many keyterms YAKE extracts before we take top_n

# --- load text and segment sentences ---

raw_text = text_path.read_text(encoding="utf-8")

sentences = re.split(r"[。；！？\n]", raw_text)
sentences = [s.strip() for s in sentences if s.strip()]

# --- CLTK tokenization ---
# Use stanza's lzh (Classical Chinese) model so YAKE gets linguistically
# meaningful token boundaries rather than a naive character-by-character split.

cltk_nlp = NLP(language_code="lzh")
doc = cltk_nlp.analyze(text=raw_text)

# Keep only tokens that are entirely CJK characters — stanza also produces
# punctuation tokens (:, 「, 。, ,) which corrupt YAKE's statistics.


def is_cjk(s: str) -> bool:
    return bool(s) and all("\u4e00" <= c <= "\u9fff" for c in s)


tokens = [w.string for w in doc.words if is_cjk(w.string)]

# --- frequency-based keyterm extraction ---
# YAKE's statistical model doesn't transfer to classical Chinese, so we rank
# by token frequency instead, excluding grammatical function words.

# Classical Chinese particles, pronouns, connectives, and auxiliaries that
# carry little conceptual content on their own.
STOPWORDS: set[str] = {
    # particles
    "之", "也", "乎", "矣", "焉", "哉", "邪", "耳", "已", "與",
    # connectives / adverbs
    "而", "則", "以", "且", "雖", "若", "如", "猶", "亦", "故", "乃", "夫",
    # pronouns / demonstratives
    "我", "吾", "汝", "其", "此", "彼", "是", "之",
    # auxiliaries / copulas
    "有", "無", "為", "曰", "謂", "不", "非", "所", "者", "於", "豈",
    # other high-frequency function words
    "然", "得", "能", "可", "將", "及", "皆", "未",
}

freq = Counter(tokens)
keyterm_phrases = [
    tok for tok, _ in freq.most_common(yake_total)
    if tok not in STOPWORDS and len(tok) >= 1
]

# always include the target term
top_keyterms = set(keyterm_phrases[:top_n])
top_keyterms.add(term)

print(f"Extracted {len(top_keyterms)} keyterms (including '{term}')")
print(f"Target term in keyterms: {term in top_keyterms}")

# --- co-occurrence network ---

sent_keyterm_lists = [
    [p for p in top_keyterms if p in sent]
    for sent in sentences
]
sent_keyterm_lists = [s for s in sent_keyterm_lists if len(s) >= 2]

print(f"\nSentences with ≥2 co-occurring keyterms: {len(sent_keyterm_lists)}")

if not sent_keyterm_lists:
    print("  Co-occurrence network is empty — no sentences share 2+ keyterms.")
else:
    G = tnet.build_cooccurrence_network(sent_keyterm_lists, window_size=100)
    G.remove_nodes_from(list(nx.isolates(G)))

    print(f"  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

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
    out = DIST / f"cooccurrence_{term}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved {out}")

    (DIST / f"cooccurrence_{term}.json").write_text(
        json.dumps(nx.node_link_data(G), indent=2), encoding="utf-8"
    )

# --- semantic similarity network ---

tokenized_keyterms = [tuple(phrase) for phrase in sorted(top_keyterms)]

S = tnet.build_similarity_network(tokenized_keyterms, edge_weighting="jaccard")

low_edges = [(u, v) for u, v, d in S.edges(data=True) if d["weight"] < 0.2]
S.remove_edges_from(low_edges)
S.remove_nodes_from(list(nx.isolates(S)))

label_map = {node: "".join(node) for node in S.nodes()}
S = nx.relabel_nodes(S, label_map)

print(
    f"\nSimilarity network — Nodes: {S.number_of_nodes()}  Edges: {S.number_of_edges()}")

if S.number_of_nodes() == 0:
    print("  Similarity network is empty — no keyterm pairs above Jaccard threshold.")
else:
    node_weights = tnet.rank_nodes_by_pagerank(S)
    max_weight = max(node_weights.values())

    pos = nx.spring_layout(
        S, seed=42, k=3.0 / (S.number_of_nodes() ** 0.5), weight="weight"
    )

    edge_weights = [S[u][v]["weight"] for u, v in S.edges()]
    max_ew = max(edge_weights)

    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    node_sizes = [
        3000 * (node_weights.get(n, 0) /
                max_weight) ** 0.5 if n != term else 3000
        for n in S.nodes()
    ]
    node_colors = ["#e07b39" if n == term else "#6aaed6" for n in S.nodes()]
    edge_widths = [3 * w / max_ew for w in edge_weights]

    nx.draw_networkx_nodes(S, pos, node_size=node_sizes, node_color=node_colors,
                           alpha=0.85, ax=ax)
    nx.draw_networkx_edges(S, pos, width=edge_widths, alpha=0.4,
                           edge_color="#888888", ax=ax)
    for node, (x, y) in pos.items():
        ax.text(x, y, node,
                fontsize=11 if node == term else 8,
                fontweight="bold" if node == term else "normal",
                color="#c0440a" if node == term else "#333333",
                ha="center", va="center")

    ax.set_title(f"Keyterm similarity network (Jaccard) — '{term}' highlighted",
                 fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    out = DIST / f"similarity_{term}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved {out}")

    (DIST / f"similarity_{term}.json").write_text(
        json.dumps(nx.node_link_data(S), indent=2), encoding="utf-8"
    )
