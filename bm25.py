"""
Minimal from-scratch BM25Okapi implementation using only numpy and the
Python standard library. Mirrors rank_bm25.BM25Okapi exactly (k1=1.5,
b=0.75, epsilon=0.25 defaults) so retrieval behavior is unchanged.
Used in place of the third-party rank_bm25 package, which is outside the
assignment's allowed import list (numpy, sentence-transformers, faiss-cpu).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np


class SimpleBM25Okapi:
    def __init__(self, corpus: Sequence[Sequence[str]], k1: float = 1.5,
                 b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []

        nd = self._initialize(corpus)
        self._calc_idf(nd)
        self.doc_len_arr = np.array(self.doc_len, dtype=np.float64)

    def _initialize(self, corpus):
        nd: Dict[str, int] = {}
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies: Dict[str, int] = {}
            for word in document:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)

            for word in frequencies:
                nd[word] = nd.get(word, 0) + 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size if self.corpus_size else 0.0
        return nd

    def _calc_idf(self, nd):
        idf_sum = 0.0
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf) if self.idf else 0.0

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query: Sequence[str]) -> np.ndarray:
        score = np.zeros(self.corpus_size, dtype=np.float64)
        doc_len = self.doc_len_arr
        for q in query:
            idf_q = self.idf.get(q)
            if not idf_q:
                continue
            q_freq = np.array([doc.get(q, 0) for doc in self.doc_freqs], dtype=np.float64)
            score += idf_q * (q_freq * (self.k1 + 1) /
                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score
