import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    source: str
    domain: str
    page: int | None = None


@dataclass(frozen=True)
class Chunk:
    id: str
    document: Document
    text: str
    position: int


def normalize(text: str) -> str:
    text = re.sub(r"[\u064b-\u065f\u0670ـ]", "", text.lower())
    return re.sub("[أإآ]", "ا", text).replace("ى", "ي")


def tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:[.-][^\W_]+)*", normalize(text), re.UNICODE)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?؟])\s+|\n+", text) if s.strip()]


def infer_domain(title: str, text: str = "") -> str:
    value = normalize(title + " " + text[:180])
    if any(w in value for w in ("scope", "climate", "sustainab", "csrd", "esrs", "ghg")):
        return "Sustainability"
    if any(w in value for w in ("basel", "frtb", "risk", "hedg", "swap", "cyber", "option", "duration")):
        return "Risk & regulation"
    return "Finance & accounting"


def parse_bytes(name: str, content: bytes) -> list[Document]:
    extension = Path(name).suffix.lower()
    digest = hashlib.sha256(content).hexdigest()[:12]
    if len(content) > 10 * 1024 * 1024:
        raise ValueError("Each file must be smaller than 10 MB.")
    if extension == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        if len(reader.pages) > 150:
            raise ValueError("PDF files must have at most 150 pages.")
        result = []
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                result.append(Document(f"{digest}-p{number}", name, text, name, infer_domain(name, text), number))
        if not result:
            raise ValueError("No text found. Scanned PDFs need OCR before upload.")
        return result
    if extension not in (".json", ".txt", ".md"):
        raise ValueError("Supported files: PDF, TXT, MD and JSON.")
    text = content.decode("utf-8-sig")
    if extension != ".json":
        return [Document(digest, Path(name).stem, text, name, infer_domain(name, text))] if text.strip() else []
    rows = json.loads(text)
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("JSON must contain an object or an array of objects.")
    result = []
    for i, row in enumerate(rows):
        body = row.get("content") or row.get("text") or row.get("response")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("Every JSON record needs a nonempty content, text or response string.")
        title = str(row.get("title") or row.get("topic") or row.get("instruction") or f"Record {i + 1}")
        identifier = str(row.get("id") or f"{digest}-{i}")
        result.append(Document(identifier, title, body, name, infer_domain(title, body)))
    return result


def load_demo() -> list[Document]:
    root = Path(__file__).resolve().parent.parent / "data"
    return [doc for path in sorted(root.glob("*.json")) if path.name != "benchmark.json" for doc in parse_bytes(path.name, path.read_bytes())]


def chunk_documents(documents: list[Document], method: str = "sentence_window", size: int = 80) -> list[Chunk]:
    result = []
    for doc in documents:
        parts = sentences(doc.text)
        if method == "semantic" and len(parts) > 1:
            matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(parts)
            similarity = np.asarray(matrix[:-1].multiply(matrix[1:]).sum(axis=1)).ravel()
        else:
            similarity = np.ones(max(0, len(parts) - 1))
        groups, current, count = [], [], 0
        for i, sentence in enumerate(parts):
            words = sentence.split()
            if len(words) > size:
                if current:
                    groups.append(" ".join(current))
                    current, count = [], 0
                groups.extend(" ".join(words[j:j + size]) for j in range(0, len(words), max(1, size - 20)))
                continue
            boundary = method == "semantic" and i > 0 and similarity[i - 1] < 0.12 and count > 35
            if current and (count + len(words) > size or boundary):
                groups.append(" ".join(current))
                current = current[-1:] if method == "sentence_window" else []
                count = sum(len(s.split()) for s in current)
                if count + len(words) > size:
                    current, count = [], 0
            current.append(sentence)
            count += len(words)
        if current:
            groups.append(" ".join(current))
        for i, group in enumerate(groups):
            result.append(Chunk(f"{doc.id}:{i}", doc, group, i))
    return result
