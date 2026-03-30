"""
双路召回模块：BM25（关键词）+ 向量（语义）+ RRF 融合
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from rag.document_loader import Document


def _tokenize(text: str) -> List[str]:
    """简单中英文分词：英文按空格，中文按字"""
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())
    return tokens


class BM25Retriever:
    """BM25 稀疏检索：基于词频统计"""

    def __init__(self, documents: List[Document]):
        self.documents = documents
        tokenized = [_tokenize(doc.content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 索引构建完成，文档数: {len(documents)}")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.documents[i], float(scores[i])) for i in top_indices if scores[i] > 0]
        logger.debug(f"BM25 召回: '{query[:30]}', 返回 {len(results)} 条")
        return results


class VectorRetriever:
    """向量语义检索：FAISS + 索引持久化"""

    def __init__(self, documents: List[Document], model_name: str = None,
                 persist_dir: str = None):
        import faiss
        import json
        from openai import OpenAI
        from config.settings import (
            OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
            EMBEDDING_MODEL, VECTOR_DB_DIR,
        )

        self.documents = documents
        self.embed_model = model_name or EMBEDDING_MODEL
        self.client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

        save_dir = Path(persist_dir) if persist_dir else VECTOR_DB_DIR
        index_path = save_dir / "faiss.index"
        docs_path = save_dir / "documents.json"

        if index_path.exists() and docs_path.exists():
            logger.info(f"加载本地索引: {save_dir}")
            self.index = faiss.read_index(str(index_path))
            with open(docs_path, "r", encoding="utf-8") as f:
                saved_docs = json.load(f)
            self.documents = [
                Document(content=d["content"], metadata=d["metadata"])
                for d in saved_docs
            ]
            logger.info(f"索引加载完成，文档数: {len(self.documents)}")
            return

        logger.info(f"首次构建向量索引，{len(documents)} 个文档...")
        texts = [doc.content for doc in documents]
        embeddings = self._embed_batch(texts)
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        save_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"content": d.content, "metadata": d.metadata} for d in documents],
                f, ensure_ascii=False
            )
        logger.info(f"向量索引已保存: dim={dim}, docs={len(documents)}")

    def _embed_batch(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """批量 Embedding，带指数退避重试"""
        import time
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for retry in range(3):
                try:
                    resp = self.client.embeddings.create(model=self.embed_model, input=batch)
                    all_embeddings.extend([item.embedding for item in resp.data])
                    break
                except Exception as e:
                    if retry < 2:
                        wait = 2 ** retry  # 1s, 2s
                        logger.warning(f"Embedding 失败，{wait}s 后重试: {e}")
                        time.sleep(wait)
                    else:
                        raise RuntimeError(f"Embedding 3次重试后仍失败: {e}")
            logger.debug(f"Embedding: {min(i+batch_size, len(texts))}/{len(texts)}")
        return all_embeddings

    def _embed_query(self, query: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.embed_model, input=[query])
        emb = np.array([resp.data[0].embedding], dtype=np.float32)
        import faiss
        faiss.normalize_L2(emb)
        return emb

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        query_emb = self._embed_query(query)
        scores, indices = self.index.search(query_emb, top_k)
        results = [
            (self.documents[idx], float(scores[0][i]))
            for i, idx in enumerate(indices[0])
            if idx >= 0
        ]
        logger.debug(f"向量召回: '{query[:30]}', 返回 {len(results)} 条")
        return results


class HybridRetriever:
    """混合检索器：BM25 + 向量 + RRF 融合"""

    def __init__(self, bm25: BM25Retriever, vector: VectorRetriever):
        self.bm25 = bm25
        self.vector = vector

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """
        生成文档的唯一标识

        用 source + page + chunk_id 组合，确保 BM25 和 Vector
        返回的同一文档能被正确识别和去重。
        """
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        chunk_id = doc.metadata.get("chunk_id", "")
        return f"{source}::{page}::{chunk_id}"

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        bm25_results = self.bm25.retrieve(query, top_k=top_k)
        vector_results = self.vector.retrieve(query, top_k=top_k)

        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        k = 60

        for rank, (doc, _) in enumerate(bm25_results):
            doc_id = self._doc_key(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
            doc_map[doc_id] = doc

        for rank, (doc, _) in enumerate(vector_results):
            doc_id = self._doc_key(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)
            doc_map[doc_id] = doc

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = [(doc_map[doc_id], score) for doc_id, score in sorted_items[:top_k]]

        # 计算真正的去重数量（两路都返回的文档）
        bm25_keys = {self._doc_key(d) for d, _ in bm25_results}
        vector_keys = {self._doc_key(d) for d, _ in vector_results}
        overlap = len(bm25_keys & vector_keys)

        logger.info(
            f"混合召回: BM25={len(bm25_results)} + Vector={len(vector_results)} "
            f"→ 重叠={overlap}, 融合后={len(results)}"
        )
        return results
