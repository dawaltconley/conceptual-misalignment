import json
import re
from pathlib import Path

import yake
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import textacy.representations.network as tnet
import textacy.viz.network as tviz
from matplotlib import font_manager

font_manager.fontManager.addfont(
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
_cjk_prop = font_manager.FontProperties(
    fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
matplotlib.rcParams["font.family"] = _cjk_prop.get_name()

# --- config ---

text_path = Path("mengzi_6a.txt")
term = "仁"
top_n = 50          # keyterms to include in networks
yake_total = 200    # how many keyterms YAKE extracts before we take top_n

# --- load text and segment sentences ---

raw_text = text_path.read_text(encoding="utf-8")

sentences = re.split(r"[。；！？\n]", raw_text)
sentences = [s.strip() for s in sentences if s.strip()]

# --- YAKE keyterm extraction ---
# Classical Chinese has no spaces, so we insert a space between every CJK
# character before passing to YAKE and use lan="en" to force whitespace
# tokenization. After extraction we strip the spaces back out.

def cjk_spaced(text: str) -> str:
    chars = [ch if "\u4e00" <= ch <= "\u9fff" else " " for ch in text]
    return " ".join("".join(chars).split())


spaced = cjk_spaced(raw_text)
kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=yake_total)
raw_keyterms = kw_extractor.extract_keywords(spaced)
raw_keyterms.sort(key=lambda x: x[1])  # lower score = more salient

# strip spaces back out so keyterms are contiguous Chinese characters
keyterm_phrases = [kw.replace(" ", "") for kw, _ in raw_keyterms]

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
    ax.set_title(f"Keyterm co-occurrence network — '{term}' highlighted", fontsize=14)
    out = f"cooccurrence_{term}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved {out}")

    Path(f"cooccurrence_{term}.json").write_text(
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

print(f"\nSimilarity network — Nodes: {S.number_of_nodes()}  Edges: {S.number_of_edges()}")

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
        3000 * (node_weights.get(n, 0) / max_weight) ** 0.5 if n != term else 3000
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
    out = f"similarity_{term}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved {out}")

    Path(f"similarity_{term}.json").write_text(
        json.dumps(nx.node_link_data(S), indent=2), encoding="utf-8"
    )
