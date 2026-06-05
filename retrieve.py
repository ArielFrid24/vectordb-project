"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

import json
import pickle
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
_bm25 = None
_bm25_pids = None


def _load_artifacts(root: Path):
    global _faiss_index, _texts_map, _bm25, _bm25_pids
    
    if _faiss_index is None:
        _faiss_index = faiss.read_index(str(root / "faiss.index"))
        
    if _texts_map is None:
        _texts_map = json.loads((root / "corpus_texts.json").read_text(encoding="utf-8"))
        
    # Load the BM25 index and corresponding page IDs for Hybrid Search
    if _bm25 is None:
        with open(root / "bm25.pkl", "rb") as f:
            _bm25, _bm25_pids = pickle.load(f)


def get_reranker():
    global _reranker
    if _reranker is None:
        # 1. Back to the smarter 6-layer model to restore NDCG@10
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
    reranker = get_reranker()

    # 2. Widen the funnel slightly: 10 from FAISS, 10 from BM25
    fetch_k = min(10, len(page_ids))
    
    # --- STAGE 1a: Dense Retrieval (FAISS) ---
    faiss_scores, faiss_indices = _faiss_index.search(query_vectors, fetch_k)

    all_candidates = []
    for i, query in enumerate(queries):
        seen = set()
        candidates = []
        
        # Add FAISS candidates
        for idx in faiss_indices[i]:
            if idx < 0:
                continue
            pid = int(page_ids_arr[idx])
            if pid not in seen:
                seen.add(pid)
                candidates.append(pid)
        
        # --- STAGE 1b: Sparse Retrieval (BM25) ---
        tokenized_query = query.lower().split()
        bm25_scores = _bm25.get_scores(tokenized_query)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:fetch_k]
        
        # Add BM25 candidates (ignoring duplicates already found by FAISS)
        for idx in top_bm25_idx:
            pid = _bm25_pids[idx]
            if pid not in seen:
                seen.add(pid)
                candidates.append(pid)
                
        all_candidates.append(candidates)

    # --- STAGE 2: Cross-Encoder Reranking ---
    # 3. Expand context window to 1500 characters
    all_pairs = [(queries[i], _texts_map.get(str(pid), "")[:1500])
                 for i in range(len(queries)) for pid in all_candidates[i]]
                 
    # Keep batch size at 32 for CPU memory efficiency
    all_scores = reranker.predict(all_pairs, batch_size=32, show_progress_bar=False)

    ranked: List[List[int]] = []
    offset = 0
    for i in range(len(queries)):
        n = len(all_candidates[i])
        q_scores = all_scores[offset:offset + n]
        offset += n
        
        # Sort by the Cross-Encoder score descending
        reranked = sorted(zip(all_candidates[i], q_scores), key=lambda x: -x[1])
        ranked.append([p for p, _ in reranked[:top_k]])

    return ranked