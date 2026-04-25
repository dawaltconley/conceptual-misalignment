import networkx as nx
import pytest

from cooccurrence_html import (
    build_cooccurrence_network,
    build_similarity_network,
    filter_to_sent_node_lists,
    parse_sep_article,
    tokenize_english_html,
)


# ---------------------------------------------------------------------------
# filter_to_sent_node_lists
# ---------------------------------------------------------------------------

def test_filter_excludes_low_freq_tokens():
    sents = [["love", "rare"], ["love", "rare"], ["love", "beauty"]]
    # "rare" appears twice (= min_freq), "beauty" appears once (below)
    result_sents, nodes = filter_to_sent_node_lists(sents, "love", min_freq=2)
    assert "beauty" not in nodes

def test_filter_includes_term_below_min_freq():
    sents = [["common", "other"], ["common", "other"]]
    _, nodes = filter_to_sent_node_lists(sents, "rare_term", min_freq=2)
    assert "rare_term" in nodes

def test_filter_excludes_stopwords():
    sents = [["love", "the"], ["love", "the"], ["love", "god"]]
    _, nodes = filter_to_sent_node_lists(sents, "love", min_freq=2, stopwords={"the"})
    assert "the" not in nodes

def test_filter_drops_short_sentences():
    sents = [["love"], ["love", "god"], ["love", "god"]]
    result_sents, _ = filter_to_sent_node_lists(sents, "love", min_freq=1)
    assert all(len(s) >= 2 for s in result_sents)

def test_filter_keeps_sentences_with_two_nodes():
    sents = [["love", "god"], ["love", "god"]]
    result_sents, _ = filter_to_sent_node_lists(sents, "love", min_freq=1)
    assert len(result_sents) == 2

def test_filter_returns_correct_node_set():
    sents = [["a", "b"], ["a", "b"], ["a", "c"]]
    _, nodes = filter_to_sent_node_lists(sents, "a", min_freq=2)
    assert nodes == {"a", "b"}  # "c" appears only once

def test_filter_empty_input():
    result_sents, nodes = filter_to_sent_node_lists([], "term", min_freq=1)
    assert result_sents == []
    assert "term" in nodes


# ---------------------------------------------------------------------------
# build_cooccurrence_network
# ---------------------------------------------------------------------------

def _cooc_fixtures():
    """
    'love' and 'god' co-occur 3x in 5 sentences → PMI = log(5/3) > 0.
    'beauty' and 'virtue' share a separate cluster with no overlap with love/god,
    so love is never an isolate and prune_to_neighborhood keeps it as the term.
    """
    sent_node_lists = [
        ["love", "god"],
        ["love", "god"],
        ["love", "god"],
        ["beauty", "virtue"],
        ["beauty", "virtue"],
    ]
    nodes = {"love", "god", "beauty", "virtue"}
    return sent_node_lists, nodes


def test_cooccurrence_returns_graph():
    sent_node_lists, nodes = _cooc_fixtures()
    G = build_cooccurrence_network(sent_node_lists, nodes, "love")
    assert isinstance(G, nx.Graph)

def test_cooccurrence_term_in_graph():
    sent_node_lists, nodes = _cooc_fixtures()
    G = build_cooccurrence_network(sent_node_lists, nodes, "love")
    assert "love" in G.nodes()

def test_cooccurrence_edges_are_positive_pmi():
    sent_node_lists, nodes = _cooc_fixtures()
    G = build_cooccurrence_network(sent_node_lists, nodes, "love")
    for _, _, data in G.edges(data=True):
        assert data["weight"] > 0

def test_cooccurrence_respects_max_nodes():
    sent_node_lists, nodes = _cooc_fixtures()
    G = build_cooccurrence_network(sent_node_lists, nodes, "love", max_nodes=1)
    assert G.number_of_nodes() <= 2  # term + 1

def test_cooccurrence_high_pmi_pair_connected():
    # love-god co-occur in 3/4 sentences — should have a positive PMI edge
    sent_node_lists, nodes = _cooc_fixtures()
    G = build_cooccurrence_network(sent_node_lists, nodes, "love", max_nodes=5)
    assert G.has_edge("love", "god")


# ---------------------------------------------------------------------------
# build_similarity_network
# ---------------------------------------------------------------------------

def _sim_fixtures():
    """
    'a' and 'b' share the same context word 'x' → cosine similarity = 1.0.
    'c' only ever appears with 'z' → dissimilar to 'a'/'b'.
    """
    sent_node_lists = [
        ["a", "x"], ["a", "x"], ["a", "x"],
        ["b", "x"], ["b", "x"], ["b", "x"],
        ["c", "z"], ["c", "z"],
        ["a", "b"],
    ]
    nodes = {"a", "b", "c", "x", "z"}
    return sent_node_lists, nodes


def test_similarity_returns_graph():
    sent_node_lists, nodes = _sim_fixtures()
    S = build_similarity_network(sent_node_lists, nodes, "a")
    assert isinstance(S, nx.Graph)

def test_similarity_term_in_graph():
    sent_node_lists, nodes = _sim_fixtures()
    S = build_similarity_network(sent_node_lists, nodes, "a", sim_threshold=0.5)
    assert "a" in S.nodes()

def test_similarity_edge_weights_in_range():
    sent_node_lists, nodes = _sim_fixtures()
    S = build_similarity_network(sent_node_lists, nodes, "a", sim_threshold=0.0)
    for _, _, data in S.edges(data=True):
        assert 0.0 <= data["weight"] <= 1.0

def test_similarity_threshold_filters_low_edges():
    sent_node_lists, nodes = _sim_fixtures()
    # High threshold — only near-identical context vectors survive
    S_strict = build_similarity_network(sent_node_lists, nodes, "a", sim_threshold=0.99)
    S_loose = build_similarity_network(sent_node_lists, nodes, "a", sim_threshold=0.0)
    assert S_strict.number_of_edges() <= S_loose.number_of_edges()

def test_similarity_respects_max_nodes():
    sent_node_lists, nodes = _sim_fixtures()
    S = build_similarity_network(sent_node_lists, nodes, "a", max_nodes=1, sim_threshold=0.0)
    assert S.number_of_nodes() <= 2  # term + 1

def test_similarity_identical_context_connected():
    # 'a' and 'b' have identical context vectors → cosine = 1.0 → connected at any threshold
    sent_node_lists, nodes = _sim_fixtures()
    S = build_similarity_network(sent_node_lists, nodes, "a", max_nodes=10, sim_threshold=0.5)
    assert S.has_edge("a", "b")


# ---------------------------------------------------------------------------
# parse_sep_article
# ---------------------------------------------------------------------------

SEP_HTML = """
<html><body>
  <div id="preamble"><p>Preamble text.</p></div>
  <div id="toc"><ul><li>Section 1</li></ul></div>
  <div id="main-text"><p>Main text.</p></div>
  <div id="other"><p>Should be excluded.</p></div>
</body></html>
"""

def test_parse_sep_includes_preamble():
    result = parse_sep_article(SEP_HTML)
    assert "Preamble text." in result

def test_parse_sep_includes_toc():
    result = parse_sep_article(SEP_HTML)
    assert "Section 1" in result

def test_parse_sep_includes_main_text():
    result = parse_sep_article(SEP_HTML)
    assert "Main text." in result

def test_parse_sep_excludes_other_divs():
    result = parse_sep_article(SEP_HTML)
    assert "Should be excluded." not in result

def test_parse_sep_missing_divs_returns_empty():
    result = parse_sep_article("<html><body><p>No matching divs.</p></body></html>")
    assert result == ""

def test_parse_sep_partial_divs():
    html = "<html><body><div id='main-text'><p>Only main.</p></div></body></html>"
    result = parse_sep_article(html)
    assert "Only main." in result


# ---------------------------------------------------------------------------
# tokenize_english_html  (requires en_core_web_sm)
# ---------------------------------------------------------------------------

def test_tokenize_english_html_returns_nested_lists():
    html = "<p>Benevolence and love are virtues of the divine.</p>"
    result = tokenize_english_html(html)
    assert isinstance(result, list)
    assert all(isinstance(s, list) for s in result)
    assert all(isinstance(t, str) for s in result for t in s)

def test_tokenize_english_html_excludes_stopwords():
    html = "<p>The benevolence of God is truly great.</p>"
    result = tokenize_english_html(html)
    tokens = [t for s in result for t in s]
    assert "the" not in tokens
    assert "of" not in tokens

def test_tokenize_english_html_excludes_punctuation():
    html = "<p>Benevolence, love, and virtue.</p>"
    result = tokenize_english_html(html)
    tokens = [t for s in result for t in s]
    assert "," not in tokens
    assert "." not in tokens

def test_tokenize_english_html_lemmatizes():
    # "virtues" should be lemmatized to "virtue"
    html = "<p>These are virtues of divine beings.</p>"
    result = tokenize_english_html(html)
    tokens = [t for s in result for t in s]
    assert "virtue" in tokens
    assert "virtues" not in tokens

def test_tokenize_english_html_content_pos_only():
    # Adverbs and prepositions should not appear
    html = "<p>God truly acts benevolently toward all beings.</p>"
    result = tokenize_english_html(html)
    tokens = [t for s in result for t in s]
    assert "toward" not in tokens  # preposition (ADP)
    assert "truly" not in tokens   # adverb (ADV)
