import pytest
from scholaragent.utils import (check_citations, chunk_text, clean_text, hash_split, heading_of, parse_json, rrf, slugify, split_sections,tokenize)

def test_chunk_text_overlap_and_coverage():
    words = [f"w{i}" for i in range(500)]
    chunks = chunk_text(" ".join(words), size=200, overlap=50)
    assert len(chunks) == 3
    assert all(len(c.split()) <= 200 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert a.split()[-50:] == b.split()[:50]
    covered = {w for c in chunks for w in c.split()}
    assert covered == set(words)

def test_chunk_text_edge_cases():
    assert chunk_text("") == []
    assert chunk_text("a b c", size=10, overlap=2) == ["a b c"]
    with pytest.raises(ValueError):
        chunk_text("a b c", size=5, overlap=5)

def test_clean_text_dehyphenates_and_joins_lines():
    assert clean_text("gener-\nation is\nuseful   here") == "generation is useful here"
    assert clean_text("retrieval-\naugmented") == "retrievalaugmented"

def test_heading_detection():
    assert heading_of("Abstract") == "Abstract"
    assert heading_of("ABSTRACT") == "Abstract"
    assert heading_of("1 Introduction") == "1 Introduction"
    assert heading_of("3.2 Retrieval Module") == "3.2 Retrieval Module"
    assert heading_of("II. RELATED WORK") == "Related Work"
    assert heading_of("We propose a new method for retrieval.") is None
    assert heading_of("Figure 2") is None

def test_split_sections_drops_references():
    text = (
        "Some Title\nAuthors\nAbstract\nWe study things.\n1 Introduction\nIntro text here.\n"
        "2 Related Work\nPrior work.\nReferences\n[1] Foo et al."
    )
    names = [n for n, _ in split_sections(text)]
    assert names == ["Front matter", "Abstract", "1 Introduction", "2 Related Work"]
    assert "Foo" not in " ".join(b for _, b in split_sections(text))

def test_rrf_prefers_items_ranked_high_in_both_lists():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert [i for i, _ in fused][:2] == ["a", "b"] or [i for i, _ in fused][:2] == ["b", "a"]
    assert {i for i, _ in fused} == {"a", "b", "c", "d"}
    assert fused[0][1] >= fused[-1][1]

def test_parse_json_handles_fences_and_chatter():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here you go: {"queries": ["x", "y"]} hope it helps') == {"queries": ["x", "y"]}
    with pytest.raises(ValueError):
        parse_json("no json here")

def test_check_citations_removes_invalid_markers():
    answer, used = check_citations("Agents plan [1] and act [2, 7]. Nonsense [9].", n_docs=3)
    assert used == [1, 2]
    assert "[7]" not in answer and "[9]" not in answer
    assert "[1]" in answer and "[2]" in answer

def test_hash_split_is_deterministic_and_roughly_20_percent():
    assert hash_split("2401.00001") == hash_split("2401.00001")
    share = sum(hash_split(f"2401.{i:05d}") == "test" for i in range(2000)) / 2000
    assert 0.15 < share < 0.25

def test_slugify_and_tokenize():
    assert slugify("BAAI/bge-small-en-v1.5") == "BAAI_bge-small-en-v1_5"
    assert tokenize("Chain-of-Thought, 2022!") == ["chain", "of", "thought", "2022"]