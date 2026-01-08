"""
Helper functions for the support agent.
"""
import math
import re
import urllib.request
from collections import Counter
from pathlib import Path
from firecrawl import Firecrawl
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

load_dotenv(".env", override=True)

app = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def tfidf_vector(tokens: list[str], idf: dict[str, float]) -> Counter[str]:
    tf = Counter(tokens)
    if not tf:
        return Counter()
    return Counter({term: count * idf.get(term, 0.0) for term, count in tf.items()})


def cosine_similarity(vec_a: Counter[str], vec_b: Counter[str]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    shared = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_snippet(text: str, max_chars: int = 800) -> str:
    trimmed = text.strip()
    if len(trimmed) <= max_chars:
        return trimmed
    truncated = trimmed[:max_chars]
    if "\n" in truncated:
        truncated = truncated.rsplit("\n", 1)[0]
    return truncated.strip() + "\n..."


def strip_html(text: str) -> str:
    cleaned = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> list[str]:
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = max(0, end - overlap)
    return chunks


def compute_idf(docs_tokens: list[list[str]]) -> dict[str, float]:
    total_docs = len(docs_tokens)
    if total_docs == 0:
        return {}
    doc_counts: Counter[str] = Counter()
    for tokens in docs_tokens:
        doc_counts.update(set(tokens))
    return {term: math.log((1 + total_docs) / (1 + count)) + 1.0 for term, count in doc_counts.items()}


def read_local_text(path: Path) -> str:
    data = path.read_bytes()
    return data.decode("utf-8", errors="ignore")


def fetch_url_text(url: str, timeout: int = 8) -> str:
    try:
        result = app.scrape(url)

        if hasattr(result, "markdown") and result.markdown: 
            return result.markdown
    except Exception as exc:
            logger.error("Failed scraping doc source %s: %s", url, str(exc), exc_info=True)        
