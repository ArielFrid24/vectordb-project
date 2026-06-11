"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

import json
import pickle
import re
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
_corpus_vectors = None
_corpus_page_ids_arr = None

_MULTI_ANSWER_RE = re.compile(
    r'\b(what links|what can be learned|how do .{0,40} and|'
    r'what connects|what ties|how does .{0,40} relate)\b',
    re.IGNORECASE
)


def _is_multi_answer_query(query: str) -> bool:
    return bool(_MULTI_ANSWER_RE.search(query))


def _load_artifacts(root: Path):
    global _faiss_index, _texts_map, _bm25, _bm25_pids, _corpus_vectors, _corpus_page_ids_arr

    if _faiss_index is None:
        _faiss_index = faiss.read_index(str(root / "faiss.index"))

    if _texts_map is None:
        _texts_map = json.loads((root / "corpus_texts.json").read_text(encoding="utf-8"))

    if _bm25 is None:
        with open(root / "bm25.pkl", "rb") as f:
            _bm25, _bm25_pids = pickle.load(f)

    if _corpus_vectors is None:
        _corpus_vectors = np.load(root / "index_vectors.npy")
        meta = json.loads((root / "index_meta.json").read_text(encoding="utf-8"))
        _corpus_page_ids_arr = np.array([int(x) for x in meta["page_ids"]])


def get_reranker():
    global _reranker
    if _reranker is None:
        device_target = "cuda" if torch.cuda.is_available() else "cpu"
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device_target)
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

    BASE_FETCH_K = 15
    MULTI_FETCH_K = 30
    CLUSTER_SIM_THRESH = 0.85
    CLUSTER_MAX_EXPAND = 4

    max_fetch_k = min(MULTI_FETCH_K, len(page_ids))
    faiss_scores_all, faiss_indices_all = _faiss_index.search(query_vectors, max_fetch_k)

    all_candidates = []
    all_faiss_scores = []

    for i, query in enumerate(queries):
        fetch_k = MULTI_FETCH_K if _is_multi_answer_query(query) else BASE_FETCH_K
        fetch_k = min(fetch_k, len(page_ids))

        seen = {}
        candidates_ordered = []

        for rank_idx in range(fetch_k):
            idx = faiss_indices_all[i][rank_idx]
            if idx < 0:
                continue
            pid = int(page_ids_arr[idx])
            score = float(faiss_scores_all[i][rank_idx])
            if pid not in seen:
                seen[pid] = score
                candidates_ordered.append(pid)

        tokenized_query = query.lower().split()
        bm25_scores = _bm25.get_scores(tokenized_query)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:fetch_k]

        for idx in top_bm25_idx:
            pid = _bm25_pids[idx]
            if pid not in seen:
                seen[pid] = 0.0
                candidates_ordered.append(pid)

        all_candidates.append(candidates_ordered)
        all_faiss_scores.append(seen)

    # --- CLUSTER EXPANSION (multi-answer queries only) ---
    for i in range(len(queries)):
        if not _is_multi_answer_query(queries[i]):
            continue
        seen = set(all_candidates[i])
        faiss_map = all_faiss_scores[i]

        for pid in list(all_candidates[i]):
            pid_idx = np.where(_corpus_page_ids_arr == pid)[0]
            if len(pid_idx) == 0:
                continue
            sims = _corpus_vectors[pid_idx[0]] @ _corpus_vectors.T
            neighbor_idxs = np.where(sims >= CLUSTER_SIM_THRESH)[0]
            neighbor_idxs = sorted(neighbor_idxs, key=lambda x: -float(sims[x]))
            added = 0
            for nidx in neighbor_idxs:
                npid = int(_corpus_page_ids_arr[nidx])
                if npid != pid and npid not in seen:
                    seen.add(npid)
                    all_candidates[i].append(npid)
                    faiss_map[npid] = float(sims[nidx]) * 0.9
                    added += 1
                    if added >= CLUSTER_MAX_EXPAND:
                        break
        all_faiss_scores[i] = faiss_map

    # --- CROSS-ENCODER RERANKING ---
    all_pairs = [
        (queries[i], _texts_map.get(str(pid), ""))
        for i in range(len(queries))
        for pid in all_candidates[i]
    ]

    all_ce_scores = reranker.predict(all_pairs, batch_size=64, show_progress_bar=False)

    ranked: List[List[int]] = []
    offset = 0
    for i in range(len(queries)):
        n = len(all_candidates[i])
        q_ce_scores = all_ce_scores[offset:offset + n]
        offset += n

        ce_min, ce_max = float(q_ce_scores.min()), float(q_ce_scores.max())
        ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
        ce_norm = (q_ce_scores - ce_min) / ce_range

        faiss_map = all_faiss_scores[i]
        raw_faiss = np.array([faiss_map.get(pid, 0.0) for pid in all_candidates[i]], dtype=np.float32)
        f_min, f_max = float(raw_faiss.min()), float(raw_faiss.max())
        f_range = f_max - f_min if f_max > f_min else 1.0
        faiss_norm = (raw_faiss - f_min) / f_range

        if _is_multi_answer_query(queries[i]):
            blended = 0.40 * ce_norm + 0.60 * faiss_norm
        else:
            blended = 0.70 * ce_norm + 0.30 * faiss_norm

        reranked = sorted(
            zip(all_candidates[i], blended),
            key=lambda x: -x[1]
        )
        ranked.append([p for p, _ in reranked[:top_k]])

    return ranked