"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import CrossEncoder

from embed import embed_queries
from index import load_index
from utils import ARTIFACTS_DIR, K_EVAL

_reranker = None
_faiss_index = None
_texts_map = None


def _load_artifacts(root: Path):
    global _faiss_index, _texts_map
    if _faiss_index is None:
        _faiss_index = faiss.read_index(str(root / "faiss.index"))
    if _texts_map is None:
        _texts_map = json.loads((root / "corpus_texts.json").read_text(encoding="utf-8"))


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    root = artifacts_dir or ARTIFACTS_DIR
    _, page_ids = load_index(artifacts_dir)
    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    _load_artifacts(root)
    page_ids_arr = np.array(page_ids)

    fetch_k = min(15, len(page_ids))
    faiss_scores, faiss_indices = _faiss_index.search(query_vectors, fetch_k)

    reranker = get_reranker()
    all_candidates = []
    for i in range(len(queries)):
        seen = set()
        candidates = []
        for idx in faiss_indices[i]:
            if idx < 0:
                continue
            pid = int(page_ids_arr[idx])
            if pid not in seen:
                seen.add(pid)
                candidates.append(pid)
        all_candidates.append(candidates)

    all_pairs = [(queries[i], _texts_map.get(str(pid), ""))
                 for i in range(len(queries)) for pid in all_candidates[i]]
    all_scores = reranker.predict(all_pairs, batch_size=64, show_progress_bar=False)

    ranked: List[List[int]] = []
    offset = 0
    for i in range(len(queries)):
        n = len(all_candidates[i])
        q_scores = all_scores[offset:offset + n]
        offset += n
        reranked = sorted(zip(all_candidates[i], q_scores), key=lambda x: -x[1])
        ranked.append([p for p, _ in reranked[:top_k]])

    return ranked