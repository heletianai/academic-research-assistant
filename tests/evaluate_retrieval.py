"""
RAG 召回评估模块
"""
import json
import sys
import time
from pathlib import Path
from typing import List

# 项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger


def load_test_cases(path: str = None) -> List[dict]:
    """加载测试用例"""
    if path is None:
        path = ROOT / "tests" / "test_cases.json"

    path = Path(path)
    if not path.exists():
        logger.warning(f"测试文件不存在: {path}，使用默认测试用例")
        return _default_test_cases()

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _default_test_cases() -> List[dict]:
    """默认的学术领域测试用例"""
    return [
        {
            "query": "What is self-attention mechanism?",
            "expected_keywords": ["attention", "query", "key", "value", "transformer"],
        },
        {
            "query": "How does RAG work?",
            "expected_keywords": ["retrieval", "generation", "augmented", "document"],
        },
        {
            "query": "What is the difference between BERT and GPT?",
            "expected_keywords": ["bidirectional", "autoregressive", "encoder", "decoder"],
        },
        {
            "query": "Explain the transformer architecture",
            "expected_keywords": ["attention", "encoder", "decoder", "layer", "multi-head"],
        },
        {
            "query": "What is prompt engineering?",
            "expected_keywords": ["prompt", "instruction", "few-shot", "chain-of-thought"],
        },
    ]


def hit_rate(retrieved_docs: list, expected_keywords: list, k: int = 3) -> float:
    """
    计算 Hit Rate@K

    如果 top-K 检索结果中有任何文档包含至少一个期望关键词，则为命中
    """
    for doc in retrieved_docs[:k]:
        content = doc.content.lower() if hasattr(doc, "content") else str(doc).lower()
        for kw in expected_keywords:
            if kw.lower() in content:
                return 1.0
    return 0.0


def mrr(retrieved_docs: list, expected_keywords: list) -> float:
    """
    计算 MRR (Mean Reciprocal Rank)

    第一个命中文档的排名倒数
    """
    for i, doc in enumerate(retrieved_docs):
        content = doc.content.lower() if hasattr(doc, "content") else str(doc).lower()
        for kw in expected_keywords:
            if kw.lower() in content:
                return 1.0 / (i + 1)
    return 0.0


def run_comparison_experiment(chunks):
    """
    运行4组对比实验

    对比：BM25 only / Vector only / Hybrid RRF / Hybrid + Reranker
    """
    from rag.retriever import BM25Retriever, VectorRetriever, HybridRetriever
    from rag.reranker import CrossEncoderReranker

    test_cases = load_test_cases()

    logger.info(f"开始对比实验: {len(test_cases)} 个测试用例, {len(chunks)} 个文档块")

    # 构建检索器
    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(chunks)
    hybrid = HybridRetriever(bm25, vector)
    reranker = CrossEncoderReranker()

    results = {
        "BM25 Only": {"hit@3": [], "hit@5": [], "mrr": [], "latency": []},
        "Vector Only": {"hit@3": [], "hit@5": [], "mrr": [], "latency": []},
        "Hybrid (RRF)": {"hit@3": [], "hit@5": [], "mrr": [], "latency": []},
        "Hybrid + Reranker": {"hit@3": [], "hit@5": [], "mrr": [], "latency": []},
    }

    for case in test_cases:
        query = case["query"]
        keywords = case["expected_keywords"]
        logger.info(f"测试: {query[:40]}")

        # BM25
        t0 = time.time()
        bm25_docs = bm25.retrieve(query, top_k=10)
        results["BM25 Only"]["latency"].append(time.time() - t0)
        docs = [d for d, _ in bm25_docs]
        results["BM25 Only"]["hit@3"].append(hit_rate(docs, keywords, 3))
        results["BM25 Only"]["hit@5"].append(hit_rate(docs, keywords, 5))
        results["BM25 Only"]["mrr"].append(mrr(docs, keywords))

        # Vector
        t0 = time.time()
        vec_docs = vector.retrieve(query, top_k=10)
        results["Vector Only"]["latency"].append(time.time() - t0)
        docs = [d for d, _ in vec_docs]
        results["Vector Only"]["hit@3"].append(hit_rate(docs, keywords, 3))
        results["Vector Only"]["hit@5"].append(hit_rate(docs, keywords, 5))
        results["Vector Only"]["mrr"].append(mrr(docs, keywords))

        # Hybrid
        t0 = time.time()
        hyb_docs = hybrid.retrieve(query, top_k=10)
        results["Hybrid (RRF)"]["latency"].append(time.time() - t0)
        docs = [d for d, _ in hyb_docs]
        results["Hybrid (RRF)"]["hit@3"].append(hit_rate(docs, keywords, 3))
        results["Hybrid (RRF)"]["hit@5"].append(hit_rate(docs, keywords, 5))
        results["Hybrid (RRF)"]["mrr"].append(mrr(docs, keywords))

        # Hybrid + Reranker
        t0 = time.time()
        hyb_docs = hybrid.retrieve(query, top_k=10)
        reranked = reranker.rerank(query, hyb_docs, top_k=5)
        results["Hybrid + Reranker"]["latency"].append(time.time() - t0)
        docs = [d for d, _ in reranked]
        results["Hybrid + Reranker"]["hit@3"].append(hit_rate(docs, keywords, 3))
        results["Hybrid + Reranker"]["hit@5"].append(hit_rate(docs, keywords, 5))
        results["Hybrid + Reranker"]["mrr"].append(mrr(docs, keywords))

    # 汇总结果
    print("\n" + "=" * 70)
    print("RAG 召回对比实验结果")
    print("=" * 70)
    print(f"{'方法':<25} {'Hit@3':>8} {'Hit@5':>8} {'MRR':>8} {'延迟(ms)':>10}")
    print("-" * 70)

    for method, metrics in results.items():
        avg_hit3 = sum(metrics["hit@3"]) / len(metrics["hit@3"]) if metrics["hit@3"] else 0
        avg_hit5 = sum(metrics["hit@5"]) / len(metrics["hit@5"]) if metrics["hit@5"] else 0
        avg_mrr = sum(metrics["mrr"]) / len(metrics["mrr"]) if metrics["mrr"] else 0
        avg_lat = sum(metrics["latency"]) / len(metrics["latency"]) * 1000 if metrics["latency"] else 0

        print(f"{method:<25} {avg_hit3:>7.1%} {avg_hit5:>7.1%} {avg_mrr:>7.3f} {avg_lat:>9.1f}")

    print("=" * 70)

    return results


if __name__ == "__main__":
    from config.settings import PAPERS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    from rag.document_loader import DocumentLoader, TextSplitter

    papers_dir = PAPERS_DIR
    if len(sys.argv) > 1:
        papers_dir = Path(sys.argv[1])

    # 加载论文
    loader = DocumentLoader()
    raw_docs = loader.load_directory(papers_dir)

    if not raw_docs:
        print(f"错误: 未找到论文，请将 PDF 放到 {papers_dir}")
        sys.exit(1)

    # 切分
    splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split(raw_docs)

    # 运行实验
    run_comparison_experiment(chunks)
