# Section B — Retrieval pipeline

## Video presentation

https://drive.google.com/file/d/1u40ATUBDfWVMbTgefIYu4yXnNkAqVZEa/view?usp=drive_link

## Setup

```bash
pip install -r requirements.txt
```

Corpus lives at **`data/Wikipedia Entries/`** (included in the handout).

## Build index

```bash
python scripts/build_index.py
```

This generates the following files under `artifacts/`:

| File | Contents |
|---|---|
| `index_vectors.npy` | Page embeddings, shape (27074, 384), MiniLM-L6-v2 |
| `index_meta.json` | `page_ids` list and chunk metadata |
| `faiss.index` | FAISS `IndexFlatIP` index for dense retrieval |
| `corpus_texts.json` | Title + up to 600 words of content per page, used for cross-encoder reranking |
| `bm25.pkl` | Serialized BM25Okapi model (full page text) + sorted page ID list |
| `bm25_title.pkl` | Serialized BM25Okapi model (titles only) + sorted page ID list |

## Pipeline overview

`run(queries)` in `main.py` calls `search_batch()` in `retrieve.py`, which performs:

1. **Embed** all queries in one batch using `all-MiniLM-L6-v2`.
2. **FAISS dense retrieval** — top-15 candidates per query.
3. **BM25 sparse retrieval** — top-15 candidates per query, unioned with FAISS results.
4. **Cluster expansion** — for each candidate, add corpus neighbors with cosine similarity ? 0.85, up to 4 per candidate.
5. **Cross-encoder reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` scores all (query, candidate) pairs; final ranking blends normalized cross-encoder and FAISS scores (70/30 for factual queries, 40/60 for detected multi-answer queries).