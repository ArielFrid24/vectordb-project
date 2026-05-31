# Section B — Retrieval Pipeline

## Setup
pip install -r requirements.txt

## Build Index (offline, run once)
cd SectionB
python scripts/build_index.py

## Evaluate
python scripts/eval_public.py

## Artifacts
- artifacts/index_vectors.npy — MiniLM embeddings for all 27,074 pages (384-dim, float32)
- artifacts/index_meta.json — page_id mapping and metadata
- artifacts/corpus_texts.json — page texts used for cross-encoder reranking

## Pipeline Description
1. Chunk: single chunk per page (title + full content via entry_text)
2. Embed: sentence-transformers/all-MiniLM-L6-v2 (L2-normalized, 384-dim)
3. Index: FAISS IndexFlatIP built offline, loaded at query time
4. Retrieve: FAISS top-15 candidates per query
5. Rerank: cross-encoder/ms-marco-MiniLM-L-6-v2 reranks candidates
6. Return: top-10 page_ids per query sorted by cross-encoder score

## Results
- Public NDCG@10: 0.2969
- Query phase time: ~30s for 50 queries
