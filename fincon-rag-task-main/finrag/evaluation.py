import json
from pathlib import Path

import numpy as np

from finrag.documents import load_demo
from finrag.engine import PipelineConfig, RAGIndex


def presets() -> dict[str, PipelineConfig]:
    return {
        "BM25L baseline": PipelineConfig(expansion="none", routing="none", retrieval="bm25", reranking="relevance", context="window"),
        "PRF + centroid + MMR": PipelineConfig(),
        "Rocchio + KNN + MMR": PipelineConfig(expansion="rocchio", routing="knn"),
        "Semantic chunks + PRF": PipelineConfig(chunking="semantic"),
    }


def evaluate() -> dict:
    root = Path(__file__).resolve().parent.parent
    cases = json.loads((root / "data" / "benchmark.json").read_text())
    documents = load_demo()
    indexes = {method: RAGIndex(documents, method) for method in ("sentence_window", "semantic")}
    rows, summary = [], []
    for name, config in presets().items():
        values = []
        for case in cases:
            result = indexes[config.chunking].retrieve(case["query"], config)
            doc_ids = list(dict.fromkeys(h["document_id"] for h in result["hits"]))
            rank = doc_ids.index(case["expected_document"]) + 1 if case["expected_document"] in doc_ids else None
            context = " ".join(h["text"] for h in result["hits"]).lower()
            coverage = sum(term.lower() in context for term in case["evidence_terms"]) / len(case["evidence_terms"])
            row = {"pipeline": name, "query": case["query"], "expected_document": case["expected_document"], "retrieved_documents": doc_ids, "hit_at_4": int(rank is not None), "reciprocal_rank": 1 / rank if rank else 0, "evidence_term_coverage": coverage, "latency_ms": result["retrieval_ms"]}
            rows.append(row)
            values.append(row)
        summary.append({"pipeline": name, "queries": len(values), "hit_at_4": round(float(np.mean([r["hit_at_4"] for r in values])), 4), "mrr_at_4": round(float(np.mean([r["reciprocal_rank"] for r in values])), 4), "evidence_term_coverage": round(float(np.mean([r["evidence_term_coverage"] for r in values])), 4), "mean_retrieval_ms": round(float(np.mean([r["latency_ms"] for r in values])), 2)})
    return {"description": "Small in-domain, authored smoke benchmark; one labeled document per query. It is not an unbiased estimate of generalization or an LLM faithfulness evaluation.", "summary": summary, "details": rows}


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
