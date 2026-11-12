import itertools
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx
from openai import AuthenticationError, RateLimitError

from finrag.documents import Document, load_demo, parse_bytes
from finrag.engine import PipelineConfig, RAGIndex
from finrag.generation import generate, make_client


@pytest.fixture(scope="module")
def index():
    return RAGIndex(load_demo())


@pytest.mark.parametrize("expansion,routing", itertools.product(["none", "prf", "rocchio"], ["none", "centroid", "knn"]))
def test_retrieval_keeps_cyber_incident_evidence(index, expansion, routing):
    result = index.retrieve("When must a material cybersecurity incident be reported under Form 8-K?", PipelineConfig(expansion=expansion, routing=routing))
    assert result["hits"][0]["document_id"] == "std_sec_cyber_01"
    assert "four business days" in " ".join(h["text"] for h in result["hits"]).lower()


def test_unrelated_query_abstains(index):
    result = generate(index.retrieve("photosynthesis chlorophyll botany", PipelineConfig()))
    assert result["hits"] == []
    assert "could not find" in result["answer"]


def test_specific_query_does_not_add_weak_material_keyword_matches(index):
    result = index.retrieve("When must a material cybersecurity incident be disclosed under Form 8-K?", PipelineConfig())
    assert all(h["document_id"] == "std_sec_cyber_01" for h in result["hits"])


@pytest.mark.parametrize("method", ["multi_query", "hyde"])
def test_llm_methods_report_actual_fallback_without_key(index, method):
    result = index.retrieve("Scope 3 emissions", PipelineConfig(expansion=method))
    assert result["effective_expansion"] == "prf"
    assert result["warnings"]
    assert result["hits"]


def test_hypothetical_passage_is_not_evidence(index):
    marker = "HYPOTHETICAL_SENTINEL"
    result = index.retrieve("Scope 3 emissions", PipelineConfig(expansion="hyde"), lambda s, u: marker + " Scope 3 value chain emissions")
    assert marker in result["search_queries"][1]
    assert all(marker not in hit["text"] for hit in result["hits"])


def test_valid_multi_query_uses_separate_searches(index):
    result = index.retrieve("Scope 3 emissions", PipelineConfig(expansion="multi_query"), lambda s, u: json.dumps(["indirect value chain emissions", "financed emissions"]))
    assert len(result["search_queries"]) == 3
    assert result["effective_expansion"] == "multi_query"


def test_invalid_multi_query_keeps_original_query(index):
    result = index.retrieve("Scope 3 emissions", PipelineConfig(expansion="multi_query"), lambda s, u: "not valid JSON")
    assert result["effective_expansion"] == "none"
    assert result["warnings"]
    assert len(result["search_queries"]) == 1


@pytest.mark.parametrize("answer", ["The deadline is tomorrow [S999].", "The deadline is tomorrow 【S999】.", "The deadline is tomorrow, trust me.", "A long paragraph with more than twelve words that contains no citation and cannot be checked.\n\nOther text [S1]."])
def test_bad_citations_force_excerpts(index, answer):
    result = generate(index.retrieve("material cybersecurity incident deadline", PipelineConfig()), lambda s, u: answer)
    assert result["answer_mode"] == "Source excerpts"
    assert result["generation_warning"]
    assert "tomorrow" not in result["answer"]


def test_valid_citations_allow_synthesis(index):
    answer = "The source describes a four-business-day disclosure period after materiality determination. [S1]"
    result = generate(index.retrieve("material cybersecurity incident deadline", PipelineConfig()), lambda s, u: answer)
    assert result["answer"] == answer
    assert result["answer_mode"] == "Groq synthesis"


@pytest.mark.parametrize("marker", ["【S1】", "［S1］"])
def test_unicode_source_markers_are_normalized_without_losing_source_checks(index, marker):
    answer = "The source gives a four-business-day deadline after the incident materiality determination. " + marker
    result = generate(index.retrieve("material cybersecurity incident deadline", PipelineConfig()), lambda s, u: answer)
    assert result["answer_mode"] == "Groq synthesis"
    assert "[S1]" in result["answer"]
    assert marker not in result["answer"]


def test_provider_errors_are_not_exposed(index):
    def unavailable(s, u):
        raise RuntimeError("private-secret-value")
    result = generate(index.retrieve("material cybersecurity incident", PipelineConfig()), unavailable)
    assert result["answer_mode"] == "Source excerpts"
    assert "private-secret-value" not in json.dumps(result)


@pytest.mark.parametrize("status,error_type,category", [(401, AuthenticationError, "authentication"), (429, RateLimitError, "rate_limit")])
def test_provider_diagnostics_distinguish_key_and_quota_without_leaking_messages(index, status, error_type, category):
    def unavailable(s, u):
        response = httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
        raise error_type("private-secret-value", response=response, body={"code": "private-secret-value"})
    result = generate(index.retrieve("material cybersecurity incident", PipelineConfig()), unavailable)
    assert result["generation_error"]["category"] == category
    assert result["generation_error"]["http_status"] == status
    assert "private-secret-value" not in json.dumps(result)


@pytest.mark.parametrize("retired_model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])
def test_old_deployment_settings_send_supported_model_parameters(monkeypatch, retired_model):
    captured = {}
    def create(**parameters):
        captured.update(parameters)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Cited answer [S1]"))])
    monkeypatch.setattr("finrag.generation.OpenAI", lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    client = make_client("test-key", retired_model)
    assert client("System", "Question") == "Cited answer [S1]"
    assert captured["model"] == "openai/gpt-oss-20b"
    assert captured["reasoning_effort"] == "low"
    assert captured["max_completion_tokens"] == 4096


@pytest.mark.parametrize("retrieval", ["bm25", "lsa", "hybrid"])
def test_single_document_upload_is_searchable(retrieval):
    documents = parse_bytes("manual.txt", b"The special reimbursement policy permits expenses up to 200 dollars. Receipts are required.")
    index = RAGIndex(documents)
    result = index.retrieve("What receipts are required for reimbursement?", PipelineConfig(retrieval=retrieval))
    assert result["hits"]
    assert result["hits"][0]["source"] == "manual.txt"


def test_uploaded_collections_are_isolated():
    first = RAGIndex(parse_bytes("first.txt", b"The reimbursement limit is 200 dollars."))
    second = RAGIndex(parse_bytes("second.txt", b"The reimbursement limit is 900 dollars."))
    a = first.retrieve("reimbursement limit", PipelineConfig())
    b = second.retrieve("reimbursement limit", PipelineConfig())
    assert "200" in a["hits"][0]["text"] and "900" not in a["hits"][0]["text"]
    assert "900" in b["hits"][0]["text"] and "200" not in b["hits"][0]["text"]


@pytest.mark.parametrize("chunking", ["sentence_window", "semantic"])
def test_evidence_is_taken_from_original_document(chunking):
    documents = load_demo()
    result = RAGIndex(documents, chunking).retrieve("Scope 3 emissions", PipelineConfig(chunking=chunking))
    for hit in result["hits"]:
        source = next(d for d in documents if d.id == hit["document_id"])
        assert hit["text"] in " ".join(source.text.split())


def test_evaluation_labels_exist_in_actual_corpus():
    cases = json.loads((Path(__file__).resolve().parents[1] / "data" / "benchmark.json").read_text())
    available = {d.id for d in load_demo()}
    assert len(cases) == 14
    assert all(case["expected_document"] in available for case in cases)


def test_bundled_demo_has_real_record_count():
    assert len(load_demo()) == 29


def test_source_code_has_no_hash_characters():
    root = Path(__file__).resolve().parents[1]
    for path in root.rglob("*.py"):
        assert chr(35) not in path.read_text(), str(path)
