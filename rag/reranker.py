"""
精排模块：CrossEncoder Reranker
"""
from __future__ import annotations

import time
from typing import List, Tuple

from loguru import logger

from rag.document_loader import Document


class CrossEncoderReranker:
    """
    本地 Cross-encoder Reranker
    ms-marco-MiniLM-L-6-v2，80MB，M系列Mac 上 10条约 0.3s
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        logger.info(f"加载 Reranker: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        logger.info("Reranker 加载完成")

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Document, float]],
        top_k: int = 3
    ) -> List[Tuple[Document, float]]:
        if not candidates:
            return []

        t0 = time.time()
        docs = [doc for doc, _ in candidates]
        pairs = [(query, doc.content) for doc in docs]
        scores = self.model.predict(pairs, show_progress_bar=False)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        results = [(doc, float(score)) for doc, score in ranked[:top_k]]
        elapsed = time.time() - t0

        logger.info(
            f"精排: {len(candidates)} → {len(results)} "
            f"(top={results[0][1]:.4f}, {elapsed:.2f}s)"
        )
        return results
