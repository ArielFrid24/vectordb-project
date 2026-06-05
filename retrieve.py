"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import torch
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
        # Keep FAISS on the CPU for architectural compatibility
        _faiss_index = faiss.read_index(str(root / "faiss.index"))
        print("[Hardware Check] FAISS Index loaded onto CPU.")
        
    if _texts_map is None:
        _texts_map = json.loads((root / "corpus_texts.json").read_text(encoding="utf-8"))
        
    if _bm25 is None:
        with open(root / "bm25.pkl", "rb") as f:
            _bm25, _bm25_pids = pickle.load(f)

def get_reranker():
    global _reranker
    if _reranker is None:
        # Explicitly define device="cuda" if available
        device_target = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Original 6-layer model for maximum NDCG
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device_target)
        
        print(f"[Hardware Check] CrossEncoder loaded on: {_reranker.model.device.type.upper()}")
            
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

    # REVERTED: Back to the original wide funnel of 15 candidates from each source
    fetch_k = min(15, len(page_ids))
    
    # --- STAGE 1a: Dense Retrieval (FAISS) ---
    faiss_scores, faiss_indices = _faiss_index.search(query_vectors, fetch_k)

    all_candidates = []
    for i, query in enumerate(queries):
        seen = set()
        candidates = []
        
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
        
        for idx in top_bm25_idx:
            pid = _bm25_pids[idx]
            if pid not in seen:
                seen.add(pid)
                candidates.append(pid)
                
        all_candidates.append(candidates)

    # --- STAGE 2: Cross-Encoder Reranking ---
    # REVERTED: No string slicing. Passing the full 400-word blocks to preserve context and accuracy.
    all_pairs = [(queries[i], _texts_map.get(str(pid), ""))
                 for i in range(len(queries)) for pid in all_candidates[i]]
                 
    # Increased batch_size to feed the GPU more data at once
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