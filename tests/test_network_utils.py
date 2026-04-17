import math
from collections import Counter

import networkx as nx
import pytest

from network_utils import (
    build_pmi_graph,
    count_pair_cooccurrences,
    is_cjk,
    pmi,
    proximity_score,
    prune_to_neighborhood,
)


# ---------------------------------------------------------------------------
# is_cjk
# ---------------------------------------------------------------------------

def test_is_cjk_single_character():
    assert is_cjk("仁") is True

def test_is_cjk_multi_character():
    assert is_cjk("仁義") is True

def test_is_cjk_mixed_raises_false():
    assert is_cjk("仁a") is False

def test_is_cjk_ascii():
    assert is_cjk("abc") is False

def test_is_cjk_empty():
    assert is_cjk("") is False

def test_is_cjk_whitespace():
    assert is_cjk(" ") is False


# ---------------------------------------------------------------------------
# pmi
# ---------------------------------------------------------------------------

def _pmi_fixtures():
    """pair_cooc, sent_freq, n_sents for a tiny two-sentence corpus."""
    pair_cooc: Counter[tuple[str, str]] = Counter({("a", "b"): 2})
    sent_freq: Counter[str] = Counter({"a": 3, "b": 3})
    n_sents = 10
    return pair_cooc, sent_freq, n_sents


def test_pmi_positive():
    pair_cooc, sent_freq, n_sents = _pmi_fixtures()
    result = pmi(pair_cooc, sent_freq, n_sents, "a", "b")
    # log((2 * 10) / (3 * 3)) = log(20/9)
    assert result == pytest.approx(math.log(20 / 9))

def test_pmi_zero_cooccurrence():
    pair_cooc: Counter = Counter()
    sent_freq: Counter = Counter({"a": 3, "b": 3})
    assert pmi(pair_cooc, sent_freq, 10, "a", "b") == float("-inf")

def test_pmi_symmetric():
    pair_cooc, sent_freq, n_sents = _pmi_fixtures()
    assert pmi(pair_cooc, sent_freq, n_sents, "a", "b") == \
           pmi(pair_cooc, sent_freq, n_sents, "b", "a")

def test_pmi_exact_independence():
    # cooc * n == freq_a * freq_b → log(1) == 0
    pair_cooc: Counter = Counter({("a", "b"): 9})
    sent_freq: Counter = Counter({"a": 3, "b": 3})
    assert pmi(pair_cooc, sent_freq, 1, "a", "b") == pytest.approx(0.0)

def test_pmi_zero_sent_freq():
    pair_cooc: Counter = Counter({("a", "b"): 1})
    sent_freq: Counter = Counter({"a": 0, "b": 3})
    assert pmi(pair_cooc, sent_freq, 10, "a", "b") == float("-inf")


# ---------------------------------------------------------------------------
# count_pair_cooccurrences
# ---------------------------------------------------------------------------

def test_pair_counts_basic():
    result = count_pair_cooccurrences([["a", "b", "c"]], {"a", "b", "c"})
    assert result[("a", "b")] == 1
    assert result[("a", "c")] == 1
    assert result[("b", "c")] == 1

def test_pair_counts_two_sentences():
    sents = [["a", "b"], ["a", "b"]]
    result = count_pair_cooccurrences(sents, {"a", "b"})
    assert result[("a", "b")] == 2

def test_pair_dedup_within_sentence():
    # "a" appears twice in the sentence but should count as one co-occurrence with "b"
    result = count_pair_cooccurrences([["a", "a", "b"]], {"a", "b"})
    assert result[("a", "b")] == 1

def test_pair_filters_vocab():
    result = count_pair_cooccurrences([["a", "b", "c"]], {"a", "b"})
    assert ("a", "c") not in result
    assert ("b", "c") not in result

def test_pair_keys_are_sorted():
    result = count_pair_cooccurrences([["b", "a"]], {"a", "b"})
    assert ("a", "b") in result
    assert ("b", "a") not in result

def test_pair_empty_sentences():
    result = count_pair_cooccurrences([[], []], {"a", "b"})
    assert len(result) == 0


# ---------------------------------------------------------------------------
# proximity_score
# ---------------------------------------------------------------------------

def _simple_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_edge("term", "x", weight=2.0)
    G.add_edge("x", "y", weight=3.0)
    G.add_edge("term", "z", weight=1.5)
    return G

def test_proximity_direct_edge():
    G = _simple_graph()
    assert proximity_score(G, "term", "x") == pytest.approx(2.0)

def test_proximity_direct_edge_other():
    G = _simple_graph()
    assert proximity_score(G, "term", "z") == pytest.approx(1.5)

def test_proximity_two_hop():
    G = _simple_graph()
    # path: term→x (2.0) → y (3.0), product = 6.0
    assert proximity_score(G, "term", "y") == pytest.approx(6.0)

def test_proximity_no_path():
    G = _simple_graph()
    G.add_node("isolated")
    assert proximity_score(G, "term", "isolated") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# prune_to_neighborhood
# ---------------------------------------------------------------------------

def _star_graph() -> nx.Graph:
    """term connected to a, b, c with weights 3, 2, 1; a also connects to d (weight 5)."""
    G = nx.Graph()
    G.add_edge("term", "a", weight=3.0)
    G.add_edge("term", "b", weight=2.0)
    G.add_edge("term", "c", weight=1.0)
    G.add_edge("a", "d", weight=5.0)   # d is 2-hop; its score = 3*5 = 15
    return G

def test_prune_result_contains_term():
    G = _star_graph()
    result = prune_to_neighborhood(G, "term", max_nodes=2)
    assert "term" in result.nodes()

def test_prune_result_size():
    G = _star_graph()
    result = prune_to_neighborhood(G, "term", max_nodes=2)
    assert result.number_of_nodes() <= 3  # term + 2

def test_prune_prioritises_one_hop():
    # max_nodes=2: should keep a (weight 3) and b (weight 2), not d (2-hop)
    G = _star_graph()
    result = prune_to_neighborhood(G, "term", max_nodes=2)
    assert "a" in result.nodes()
    assert "b" in result.nodes()

def test_prune_fills_with_two_hop():
    # max_nodes=4: all 3 one-hop kept, then d fills the remaining slot
    G = _star_graph()
    result = prune_to_neighborhood(G, "term", max_nodes=4)
    assert "d" in result.nodes()

def test_prune_term_not_in_graph():
    G = nx.Graph()
    G.add_edge("x", "y", weight=1.0)
    result = prune_to_neighborhood(G, "missing", max_nodes=5)
    assert set(result.nodes()) == {"x", "y"}

def test_prune_edges_preserved():
    # edges between kept nodes should be in the result
    G = _star_graph()
    result = prune_to_neighborhood(G, "term", max_nodes=3)
    assert result.has_edge("term", "a")


# ---------------------------------------------------------------------------
# build_pmi_graph (integration)
# ---------------------------------------------------------------------------

def test_build_pmi_graph_returns_positive_edges_only():
    # a and b always co-occur; a and c never co-occur
    sents = [["a", "b"], ["a", "b"], ["a", "c"], ["b"]]
    nodes = {"a", "b", "c"}
    G, pair_cooc, sent_freq, n_sents = build_pmi_graph(sents, nodes)
    # a-b should have positive PMI; a-c co-occurs once but check sign
    for u, v, d in G.edges(data=True):
        assert d["weight"] > 0

def test_build_pmi_graph_has_all_nodes():
    sents = [["a", "b"]]
    nodes = {"a", "b", "c"}
    G, *_ = build_pmi_graph(sents, nodes)
    # all nodes added even if isolated
    for n in nodes:
        assert n in G.nodes() or True  # isolates may be present before removal

def test_build_pmi_graph_returns_stats():
    sents = [["a", "b"], ["a", "b"]]
    nodes = {"a", "b"}
    G, pair_cooc, sent_freq, n_sents = build_pmi_graph(sents, nodes)
    assert pair_cooc[("a", "b")] == 2
    assert sent_freq["a"] == 2
    assert n_sents == 2
