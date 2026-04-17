import json
import re
from collections import Counter
from pathlib import Path

import yake
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import textacy.representations.network as tnet
import textacy.viz.network as tviz
from cltk import NLP

# load CJK font directly by path so labels render correctly
from matplotlib import font_manager
font_manager.fontManager.addfont(
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
_cjk_prop = font_manager.FontProperties(
    fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
matplotlib.rcParams["font.family"] = _cjk_prop.get_name()

# --- load text ---

text_path = Path("mengzi_6a.txt")   # adjust to your file
raw_text = text_path.read_text(encoding="utf-8")

# --- CLTK analysis ---

cltk_nlp = NLP(language_code="lzh")
doc = cltk_nlp.analyze(text=raw_text)

term = "仁"   # adjust to your target concept

# --- noun tokens ---

print(f"{'='*60}")
print(f"NOUN TOKENS containing '{term}'")
print(f"{'='*60}\n")

noun_upos = {"NOUN", "PROPN"}
nouns = [w.string for w in doc.words
         if getattr(getattr(w, "upos", None), "name", None) in noun_upos]
noun_counts = Counter(nouns)

term_nouns = [(n, c) for n, c in noun_counts.most_common() if n and term in n]
print(f"{sum(c for _, c in term_nouns)} occurrences, {len(term_nouns)} distinct noun tokens\n")
for noun, count in term_nouns:
    print(f"  {count:3d}x  {noun}")

# --- KWIC ---


def keyword_in_context(text, term, window=30):
    results = []
    idx = 0
    while True:
        pos = text.find(term, idx)
        if pos == -1:
            break
        pre = text[max(0, pos - window):pos]
        post = text[pos + len(term): pos + len(term) + window]
        results.append((pre, term, post))
        idx = pos + 1
    return results


print(f"\n{'='*60}")
print(f"KEYWORD IN CONTEXT: '{term}'")
print(f"{'='*60}\n")

matches = keyword_in_context(raw_text, term, window=30)
for pre, kw, post in matches:
    print(f"...{pre} [{kw}] {post}...")

# --- YAKE keyterms ---
# YAKE tokenizes on whitespace, so we space-separate every CJK character first.
# This lets YAKE treat individual characters as tokens and extract uni/bigrams.
# Lower score = more salient keyword.


def cjk_spaced(text: str) -> str:
    """Return text with a space between every CJK character; drop other chars."""
    chars = [ch if "\u4e00" <= ch <= "\u9fff" else " " for ch in text]
    return " ".join("".join(chars).split())


print(f"\n{'='*60}")
print(f"TOP KEY TERMS (YAKE)")
print(f"{'='*60}\n")

spaced_text = cjk_spaced(raw_text)
kw_extractor = yake.KeywordExtractor(lan="zh", n=2, dedupLim=0.9, top=100)
yake_keyterms = kw_extractor.extract_keywords(spaced_text)
# ascending: lower score = more important
yake_keyterms.sort(key=lambda x: x[1])

# strip the spaces YAKE inserts between characters so phrases read naturally
yake_keyterms = [(kw.replace(" ", ""), score) for kw, score in yake_keyterms]
keyterm_phrases = [kw for kw, _ in yake_keyterms]

for kw, score in filter(lambda kt: term in kt[0], yake_keyterms):
    print(f"  {score:.4f}  {kw}")

# --- keyterm co-occurrence in KWIC windows ---

print(f"\n{'='*60}")
print(f"KEYTERMS CO-OCCURRING WITH '{term}' IN CONTEXT WINDOWS")
print(f"{'='*60}\n")

cooccurrences = Counter()
for pre, kw, post in matches:
    window_text = pre + post
    for phrase in keyterm_phrases:
        if phrase != term and phrase in window_text:
            cooccurrences[phrase] += 1

for phrase, count in cooccurrences.most_common(20):
    if count > 1:
        print(f"  {count:3d}x  {phrase}")

# --- co-occurrence network ---

print(f"\n{'='*60}")
print(f"CO-OCCURRENCE NETWORK (whole document)")
print(f"{'='*60}\n")

sentences = re.split(r"[。；！？\n]", raw_text)
sentences = [s.strip() for s in sentences if s.strip()]

top_keyterms = set(keyterm_phrases[:50])
top_keyterms.add(term)

sent_keyterm_lists = [
    [p for p in top_keyterms if p in sent]
    for sent in sentences
]
sent_keyterm_lists = [s for s in sent_keyterm_lists if len(s) >= 2]

G = tnet.build_cooccurrence_network(sent_keyterm_lists, window_size=100)
G.remove_nodes_from(list(nx.isolates(G)))

if len(G.nodes()) == 0:
    print("  (no co-occurring keyterm pairs found — graph is empty)")
else:
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
    out_path = f"cooccurrence_{term}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Network saved to {out_path}")

    json_path = f"cooccurrence_{term}.json"
    Path(json_path).write_text(
        json.dumps(nx.node_link_data(G), indent=2), encoding="utf-8"
    )
    print(f"Network JSON saved to {json_path}")

# --- semantic similarity network ---

print(f"\n{'='*60}")
print(f"SEMANTIC SIMILARITY NETWORK (whole document)")
print(f"{'='*60}\n")

# tokenize each keyterm as a tuple of characters for Jaccard comparison
tokenized_keyterms = [tuple(phrase) for phrase in sorted(top_keyterms)]

S = tnet.build_similarity_network(tokenized_keyterms, edge_weighting="jaccard")

low_edges = [(u, v) for u, v, d in S.edges(data=True) if d["weight"] < 0.2]
S.remove_edges_from(low_edges)
S.remove_nodes_from(list(nx.isolates(S)))

label_map = {node: "".join(node) for node in S.nodes()}
S = nx.relabel_nodes(S, label_map)

if len(S.nodes()) == 0:
    print("  (no similar keyterm pairs found above threshold — graph is empty)")
else:
    node_weights = tnet.rank_nodes_by_pagerank(S)
    max_weight = max(node_weights.values()) if node_weights else 1

    pos = nx.spring_layout(S, seed=42, k=3.0 /
                           (len(S.nodes()) ** 0.5), weight="weight")

    edge_weights = [S[u][v]["weight"] for u, v in S.edges()]
    max_ew = max(edge_weights) if edge_weights else 1

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

    ax.set_title(
        f"Keyterm similarity network (Jaccard) — '{term}' highlighted", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    out_path = f"similarity_{term}.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Network saved to {out_path}")

    json_path = f"similarity_{term}.json"
    Path(json_path).write_text(
        json.dumps(nx.node_link_data(S), indent=2), encoding="utf-8"
    )
    print(f"Network JSON saved to {json_path}")

# --- SVO triples via Udkanbun (optional) ---
# Udkanbun uses UDPipe trained on Classical Chinese Universal Dependencies.
# Install with: pip install udkanbun

print(f"\n{'='*60}")
print(f"SVO TRIPLES containing '{term}' (Udkanbun)")
print(f"{'='*60}\n")

try:
    import udkanbun
    ud_nlp = udkanbun.load()

    for sent_text in sentences:
        if term not in sent_text:
            continue
        parsed = ud_nlp(sent_text)
        # parsed is a list of token dicts with CoNLL-U fields
        by_id = {t["id"]: t for t in parsed if isinstance(t.get("id"), int)}
        for tok in by_id.values():
            if tok.get("deprel") == "nsubj" and term in tok.get("form", ""):
                head = by_id.get(tok["head"])
                if not head:
                    continue
                objs = [t for t in by_id.values()
                        if t.get("head") == head["id"]
                        and t.get("deprel") in ("obj", "obl")]
                for obj_tok in objs:
                    print(f"  SUBJ: {tok['form']}")
                    print(f"  VERB: {head['form']}")
                    print(f"   OBJ: {obj_tok['form']}")
                    print()

except ImportError:
    print("  (udkanbun not installed — run: pip install udkanbun)")
