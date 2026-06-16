"""Offline index build and load (not timed at grading)."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from chunk import Chunk, chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries

INDEX_VECTORS_NAME = "index_vectors.npy"
INDEX_META_NAME = "index_meta.json"


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))
    chunks: List[Chunk] = chunk_corpus(records)
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)
    page_ids = [c.page_id for c in chunks]

    np.save(out_dir / INDEX_VECTORS_NAME, vectors)
    meta = {
        "page_ids": page_ids,
        "chunk_ids": [c.chunk_id for c in chunks],
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_vectors": len(page_ids),
    }
    (out_dir / INDEX_META_NAME).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    # save FAISS index
    dim = vectors.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(vectors)
    faiss.write_index(faiss_index, str(out_dir / "faiss.index"))

    # corpus texts for cross-encoder reranking
    records_map = {int(r["page_id"]): r for r in records}
    texts_map = {}
    for pid in set(page_ids):
        r = records_map.get(pid, {})
        title = r.get("title", "")
        content = r.get("content", "")
        words = content.split()
        if title:
            full_text = f"{title}\n\n{' '.join(words[:600])}"
        else:
            full_text = " ".join(words[:600])
        texts_map[str(pid)] = full_text
    (out_dir / "corpus_texts.json").write_text(
        json.dumps(texts_map), encoding="utf-8"
    )

    # BM25 on full page text
    page_texts = {}
    for chunk, text in zip(chunks, texts):
        if chunk.page_id not in page_texts:
            page_texts[chunk.page_id] = text
    sorted_pids = sorted(page_texts.keys())
    bm25_corpus = [page_texts[pid].lower().split() for pid in sorted_pids]
    bm25 = BM25Okapi(bm25_corpus)
    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump((bm25, sorted_pids), f)

    # BM25 on titles only
    title_corpus = [records_map[pid].get("title", "").lower().split() for pid in sorted_pids]
    bm25_title = BM25Okapi(title_corpus)
    with open(out_dir / "bm25_title.pkl", "wb") as f:
        pickle.dump((bm25_title, sorted_pids), f)

    return vectors, page_ids


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    root = artifacts_dir or ARTIFACTS_DIR
    vectors = np.load(root / INDEX_VECTORS_NAME)
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    page_ids = [int(x) for x in meta["page_ids"]]
    return vectors, page_ids