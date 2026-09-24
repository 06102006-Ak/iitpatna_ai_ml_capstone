from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_files_exist():
    required = [
        "README.md",
        "requirements.txt",
        "data_pipeline/scrape_books.py",
        "data_pipeline/load_and_query.py",
        "analytics/01_eda.py",
        "analytics/02_modeling.py",
        "analytics/reload_check.py",
        "support_assistant/ingest.py",
        "support_assistant/graph.py",
        "support_assistant/prompt.py",
        "support_assistant/main.py",
        "support_assistant/Dockerfile",
    ]
    for rel in required:
        assert (ROOT / rel).exists(), rel


def test_support_corpus_has_exactly_eight_docs():
    docs = sorted((ROOT / "support_assistant/docs").glob("doc_*.txt"))
    assert len(docs) == 8


def test_required_graph_terms_are_visible_in_source():
    source = (ROOT / "support_assistant/graph.py").read_text(encoding="utf-8")
    for term in ["StateGraph", "TypedDict", "classify_intent", "retrieve_and_answer", "direct_answer", "add_conditional_edges", "MOCK_LLM"]:
        assert term in source
