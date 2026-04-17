# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Digital humanities research tool for analyzing philosophical texts using NLP. Currently studies conceptual associations of key terms (e.g. 仁 *rén* in the Mengzi, "benevolence" in Jonathan Edwards) by building co-occurrence and semantic similarity networks, then visualizing and exporting them as JSON for further use.

## Commands

All commands assume the `.venv` virtual environment is active:

```bash
source .venv/bin/activate
```

**Run tests:**
```bash
python -m pytest tests/          # full suite
python -m pytest tests/test_network_utils.py::test_pmi_positive  # single test
```

**Run analysis scripts:**
```bash
python parse_edwards.py          # English: Jonathan Edwards SEP article
python parse_mengzi.py           # Classical Chinese: Mengzi 6A (spaCy zh model)
python networks_mengzi_ren.py    # Classical Chinese: PMI + cosine networks
```

**Download source texts:**
```bash
python download_mengzi_6a.py     # fetches from ctext.org API → text/ctext/
```

**Type-check:**
```bash
pyright <file>.py
```

## Directory structure

```
text/sep/        # Stanford Encyclopedia of Philosophy HTML source files
text/ctext/      # Classical Chinese plain-text sources (downloaded via ctext.org API)
dist/            # All outputs: JSON network files and PNG visualizations (git-ignored)
tests/           # pytest suite
```

## Path constants (`config.py`)

All scripts import path constants from `config.py` rather than hardcoding paths:
- `DIST` — output directory (`dist/`)
- `SEP` — SEP HTML inputs (`text/sep/`)
- `CTEXT` — classical Chinese text inputs (`text/ctext/`)

`config.py` also calls `mkdir(exist_ok=True)` on all three, so they are created on import.

## Shared utilities (`network_utils.py`)

Pure functions used by multiple scripts. All are unit-tested:

| Function | Purpose |
|---|---|
| `pmi(pair_cooc, sent_freq, n_sents, a, b)` | Pointwise mutual information between two terms |
| `count_pair_cooccurrences(sent_token_lists, vocab)` | Sentence-level co-occurrence counts, sorted `(min, max)` keys |
| `build_pmi_graph(sent_node_lists, nodes)` | Full PMI graph; returns `(G, pair_cooc, sent_freq, n_sents)` |
| `build_cosine_similarity_graph(sent_node_lists, node_list, threshold)` | Context-vector cosine similarity graph |
| `proximity_score(G, term, node)` | 1-hop edge weight or best 2-hop product |
| `prune_to_neighborhood(G, term, max_nodes)` | Ego-subgraph pruned to top N nodes; 1-hop neighbors prioritized |
| `save_graph_json(G, path)` | Serialize to `nx.node_link_data()` JSON |
| `draw_term_network(G, term, title, out_path)` | Matplotlib spring-layout visualization with target term highlighted |

## Network output format

JSON files use NetworkX's `node_link_data()` schema:
```json
{ "directed": false, "multigraph": false, "graph": {},
  "nodes": [{"id": "仁"}, ...],
  "edges": [{"weight": 1.969, "source": "仁", "target": "義"}, ...] }
```

## Classical Chinese pipeline (`networks_mengzi_ren.py`)

Key design decisions:
- **Tokenizer**: CLTK `NLP(language_code="lzh")` (stanza `kyoto` model) groups tokens by `w.index_sentence`; punctuation tokens are filtered with `is_cjk()`
- **Node selection**: frequency ≥ `MIN_FREQ` after applying `STOPWORDS` (classical Chinese particles, connectives, pronouns); target `TERM` always included
- **Co-occurrence edges**: positive PMI only — terms that appear together *more* than chance predicts
- **Similarity edges**: cosine similarity of sentence co-occurrence context vectors, pruned at `SIM_THRESHOLD`
- **Graph pruning**: 1-hop neighbors of `TERM` filled first by direct edge weight; remaining `MAX_NODES` slots filled by 2-hop score (product of two edge weights)

## English pipeline (`parse_edwards.py`)

Uses spaCy `en_core_web_sm` with `spacy_html_tokenizer` to parse SEP HTML directly. Co-occurrence network uses textacy's `build_cooccurrence_network`; similarity network uses textacy's `build_similarity_network` with Jaccard on token-set overlap of TextRank keyphrases.

## Known limitations

- `zh_core_web_sm` (modern Mandarin) is used in `parse_mengzi.py` as a simpler alternative; `networks_mengzi_ren.py` uses CLTK's classical Chinese model instead
- textacy's `noun_chunks` is not implemented for `zh`, so `parse_mengzi.py` falls back to POS-filtered tokens (`NOUN`, `PROPN`)
- CJK font rendering in matplotlib requires `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`; the font registers as `"Noto Sans CJK JP"` under matplotlib's font manager
