from collections import Counter, defaultdict

import networkx as nx
import spacy
from bs4 import BeautifulSoup
from spacy_html_tokenizer import create_html_tokenizer

from network_utils import build_pmi_graph, build_cosine_similarity_graph, is_cjk, prune_to_neighborhood

CONTENT_POS = {"NOUN", "VERB", "ADJ", "PROPN"}

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
        _nlp.tokenizer = create_html_tokenizer()(_nlp)
    return _nlp


def tokenize_english_html(html: str) -> list[list[str]]:
    """Per-sentence lemma lists from an HTML string via spaCy (content words only)."""
    nlp = _get_nlp()
    doc = nlp(html)
    return [
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


def tokenize_classical_chinese(text: str) -> list[list[str]]:
    """Per-sentence CJK token lists via CLTK's classical Chinese (lzh) model."""
    from cltk import NLP as CLTK_NLP

    cltk_nlp = CLTK_NLP(language_code="lzh")
    doc = cltk_nlp.analyze(text=text)

    sent_tokens: dict[int, list[str]] = defaultdict(list)
    for w in doc.words:
        if w.string and w.index_sentence and is_cjk(w.string):
            sent_tokens[w.index_sentence].append(w.string)

    return [sent_tokens[i] for i in sorted(sent_tokens.keys())]


def filter_to_sent_node_lists(
    sent_token_lists: list[list[str]],
    term: str,
    min_freq: int,
    stopwords: set[str] | frozenset[str] = frozenset(),
) -> tuple[list[list[str]], set[str]]:
    """Frequency-filter a token vocabulary, then return sentences restricted to that vocab."""
    freq = Counter(tok for sent in sent_token_lists for tok in sent)
    nodes: set[str] = {
        t for t, c in freq.items()
        if c >= min_freq and t not in stopwords
    }
    nodes.add(term)

    sent_node_lists = [
        [t for t in sent if t in nodes]
        for sent in sent_token_lists
    ]
    return [s for s in sent_node_lists if len(s) >= 2], nodes


def build_cooccurrence_network(
    sent_node_lists: list[list[str]],
    nodes: set[str],
    term: str,
    max_nodes: int = 15,
) -> nx.Graph:
    G, _, _, _ = build_pmi_graph(sent_node_lists, nodes)
    G.remove_nodes_from(list(nx.isolates(G)))
    return prune_to_neighborhood(G, term, max_nodes)


def build_similarity_network(
    sent_node_lists: list[list[str]],
    nodes: set[str],
    term: str,
    max_nodes: int = 15,
    sim_threshold: float = 0.7,
) -> nx.Graph:
    S = build_cosine_similarity_graph(sent_node_lists, sorted(nodes), sim_threshold)
    S.remove_nodes_from(list(nx.isolates(S)))
    return prune_to_neighborhood(S, term, max_nodes)


def parse_sep_article(html: str) -> str:
    """Extract preamble, TOC, and main-text divs from an SEP article page."""
    soup = BeautifulSoup(html, "html.parser")
    parts = [soup.find("div", id=div_id) for div_id in ("preamble", "toc", "main-text")]
    return "".join(str(p) for p in parts if p is not None)
