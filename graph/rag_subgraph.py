"""
CRAG 子图：检索 → 评分 → 条件路由（生成/重检索/Web兜底）
"""
from __future__ import annotations

from typing import Annotated, List

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger
from openai import OpenAI
from typing_extensions import TypedDict

from config.prompts import CRAG_QUERY_REWRITE_PROMPT, RAG_GENERATE_PROMPT
from config.settings import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL,
    TOP_K_RETRIEVAL, TOP_K_RERANK,
)


class RAGSubState(TypedDict):
    """RAG 子图的状态"""
    query: str
    rewritten_query: str
    retrieved_docs: List[dict]
    retrieval_grade: str
    rag_context: str
    retrieval_attempts: int
    final_answer: str
    messages: Annotated[list, add_messages]


# ── 全局单例（延迟初始化）──────────────────────────────────
_retriever = None
_reranker = None
_grader = None
_llm_client = None


def _get_retriever():
    """获取检索器，如果未初始化则返回 None（降级到 Web Search）"""
    global _retriever
    return _retriever


def _get_reranker():
    global _reranker
    if _reranker is None:
        from rag.reranker import CrossEncoderReranker
        _reranker = CrossEncoderReranker()
    return _reranker


def _get_grader():
    global _grader
    if _grader is None:
        from rag.grader import DocumentGrader
        _grader = DocumentGrader()
    return _grader


def _get_llm():
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
    return _llm_client


def init_rag_subgraph(retriever):
    """初始化 RAG 子图的检索器（在应用启动时调用）"""
    global _retriever
    _retriever = retriever
    logger.info("RAG 子图初始化完成")


# ── 子图节点 ─────────────────────────────────────────────

def retrieve_node(state: RAGSubState) -> dict:
    """检索节点：执行混合召回。如果没有论文(retriever=None)，返回空结果→CRAG自动走Web Search"""
    query = state.get("rewritten_query") or state["query"]
    retriever = _get_retriever()

    if retriever is None:
        logger.warning("未加载论文，检索返回空结果（将自动降级到 Web Search）")
        return {"retrieved_docs": [], "rewritten_query": query}

    results = retriever.retrieve(query, top_k=TOP_K_RETRIEVAL)

    docs = [
        {"content": doc.content, "metadata": doc.metadata, "score": score}
        for doc, score in results
    ]
    attempts = state.get("retrieval_attempts", 0) + 1

    logger.info(f"检索完成: query='{query[:30]}', 返回 {len(docs)} 条, 第{attempts}次")
    return {"retrieved_docs": docs, "retrieval_attempts": attempts}


def rerank_node(state: RAGSubState) -> dict:
    """精排节点：CrossEncoder 重排序"""
    from rag.document_loader import Document

    query = state.get("rewritten_query") or state["query"]
    docs = state["retrieved_docs"]

    if not docs:
        return {"retrieved_docs": []}

    reranker = _get_reranker()
    candidates = [
        (Document(content=d["content"], metadata=d["metadata"]), d["score"])
        for d in docs
    ]
    reranked = reranker.rerank(query, candidates, top_k=TOP_K_RERANK)

    result_docs = [
        {"content": doc.content, "metadata": doc.metadata, "score": score}
        for doc, score in reranked
    ]
    return {"retrieved_docs": result_docs}


def grade_documents_node(state: RAGSubState) -> dict:
    """CRAG 评分节点：评估检索文档质量"""
    from rag.document_loader import Document

    query = state.get("rewritten_query") or state["query"]
    docs = state["retrieved_docs"]

    if not docs:
        return {"retrieval_grade": "irrelevant", "retrieved_docs": []}

    grader = _get_grader()
    candidates = [
        (Document(content=d["content"], metadata=d["metadata"]), d["score"])
        for d in docs
    ]

    overall_grade, filtered = grader.grade_documents(query, candidates)

    filtered_docs = [
        {"content": doc.content, "metadata": doc.metadata, "score": score}
        for doc, score in filtered
    ]

    return {"retrieval_grade": overall_grade, "retrieved_docs": filtered_docs}


def generate_node(state: RAGSubState) -> dict:
    """生成节点：基于检索到的文档生成回答"""
    query = state.get("rewritten_query") or state["query"]
    docs = state["retrieved_docs"]

    # 拼接上下文
    context_parts = []
    for i, doc in enumerate(docs):
        source = doc["metadata"].get("source", "未知")
        page = doc["metadata"].get("page", "?")
        context_parts.append(f"[片段{i+1}] 来源: {source}, 页码: {page}\n{doc['content']}")

    context = "\n\n".join(context_parts) if context_parts else "未找到相关文档。"

    prompt = RAG_GENERATE_PROMPT.format(context=context, query=query)
    client = _get_llm()

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # RAG 场景低温度减少幻觉
        max_tokens=1000,
    )
    answer = resp.choices[0].message.content.strip()

    logger.info(f"RAG 生成完成: {len(answer)} 字符")
    return {
        "final_answer": answer,
        "rag_context": context,
    }


def query_rewrite_for_retrieval_node(state: RAGSubState) -> dict:
    """CRAG Query 重写节点：ambiguous 时改写检索词再重新检索"""
    query = state.get("rewritten_query") or state["query"]
    docs = state.get("retrieved_docs", [])

    # 用已检索到的部分信息辅助改写
    # 取每个文档中信息量最大的片段（跳过前30字可能是页眉/页脚），取中间部分
    partial_context = ""
    if docs:
        partial_parts = []
        for d in docs[:2]:
            content = d["content"]
            # 跳过可能的页眉/页脚，取中间段的关键信息
            start = min(30, len(content) // 4)
            end = min(start + 300, len(content))
            snippet = content[start:end].strip()
            if snippet:
                partial_parts.append(snippet)
        partial_context = "\n---\n".join(partial_parts)

    client = _get_llm()
    prompt = CRAG_QUERY_REWRITE_PROMPT.format(
        query=query,
        partial_context=partial_context or "无",
    )

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
        )
        new_query = resp.choices[0].message.content.strip()
        logger.info(f"CRAG Query 重写: '{query[:30]}' → '{new_query[:30]}'")
        return {"rewritten_query": new_query}
    except Exception as e:
        logger.warning(f"CRAG Query 重写失败: {e}")
        return {}


def _web_search(query: str, max_results: int = 3) -> list:
    """使用 DuckDuckGo 搜索学术信息"""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {"title": r.get("title", ""), "body": r.get("body", ""), "href": r.get("href", "")}
            for r in results
        ]
    except ImportError:
        logger.warning("duckduckgo-search 未安装，使用 httpx 简单搜索")
        # 退化方案：用 httpx 调 DuckDuckGo Lite
        try:
            import httpx
            resp = httpx.get(
                "https://lite.duckduckgo.com/lite/",
                params={"q": query},
                timeout=10,
                follow_redirects=True,
            )
            # 简单提取文本片段
            from html.parser import HTMLParser
            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.texts = []
                    self._in_result = False
                def handle_data(self, data):
                    text = data.strip()
                    if len(text) > 30:
                        self.texts.append(text)
            parser = TextExtractor()
            parser.feed(resp.text)
            return [{"title": "", "body": t, "href": ""} for t in parser.texts[:max_results]]
        except Exception as e:
            logger.warning(f"Web Search 备用方案也失败: {e}")
            return []
    except Exception as e:
        logger.warning(f"DuckDuckGo 搜索失败: {e}")
        return []


def web_search_fallback_node(state: RAGSubState) -> dict:
    """Web Search 兜底节点：本地知识库检索质量差时，搜索网络获取信息"""
    query = state.get("rewritten_query") or state["query"]

    # 执行 Web Search
    search_results = _web_search(query)

    if search_results:
        # 用搜索结果作为上下文，让 LLM 生成回答
        search_context_parts = []
        for i, r in enumerate(search_results):
            title = r.get("title", "无标题")
            body = r.get("body", "")
            href = r.get("href", "")
            search_context_parts.append(f"[搜索结果{i+1}] {title}\n{body}\n来源: {href}")
        search_context = "\n\n".join(search_context_parts)

        client = _get_llm()
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": (
                    "你是一个AI学术研究助手。本地知识库中未找到相关论文，"
                    "以下是从网络搜索到的信息。请基于搜索结果回答用户问题，"
                    "并注明信息来源于网络搜索而非本地论文。"
                )},
                {"role": "user", "content": f"搜索结果：\n{search_context}\n\n用户问题：{query}"},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        answer = resp.choices[0].message.content.strip()
        context_note = f"[Web Search 兜底 — 搜索到 {len(search_results)} 条结果]"
    else:
        # 搜索也失败了，最终降级到 LLM 自身知识
        client = _get_llm()
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": (
                    "你是一个AI学术研究助手。本地知识库和网络搜索都未找到相关信息，"
                    "请基于你的知识尽力回答，但要明确说明这不是来自论文或搜索结果的信息。"
                )},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        answer = resp.choices[0].message.content.strip()
        context_note = "[Web Search 失败，降级到 LLM 自身知识]"

    logger.info(f"Web Search 兜底完成: {context_note}")
    return {
        "final_answer": answer,
        "rag_context": context_note,
    }


# ── 条件路由 ─────────────────────────────────────────────

def route_by_grade(state: RAGSubState) -> str:
    """根据 CRAG 评分决定下一步"""
    grade = state.get("retrieval_grade", "irrelevant")
    attempts = state.get("retrieval_attempts", 0)

    if grade == "relevant":
        return "generate"
    elif grade == "ambiguous" and attempts < 3:
        # 重试检索（最多3次）
        logger.info(f"CRAG: ambiguous, 重试检索 (第{attempts}次)")
        return "re_retrieve"
    else:
        # irrelevant 或重试次数用完
        logger.info(f"CRAG: {grade}, 走 Web Search 兜底")
        return "web_search"


# ── 构建子图 ─────────────────────────────────────────────

def build_rag_subgraph() -> StateGraph:
    """
    构建 CRAG 子图

    流程：retrieve → rerank → grade
      ├── relevant → generate → END
      ├── ambiguous → query_rewrite → retrieve（循环，最多3次）
      └── irrelevant → web_search_fallback → END

    """
    graph = StateGraph(RAGSubState)

    # 添加节点
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("generate", generate_node)
    graph.add_node("query_rewrite_for_retrieval", query_rewrite_for_retrieval_node)
    graph.add_node("web_search_fallback", web_search_fallback_node)

    # 添加边
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "grade_documents")

    # 条件路由：根据评分走不同分支
    graph.add_conditional_edges(
        "grade_documents",
        route_by_grade,
        {
            "generate": "generate",
            "re_retrieve": "query_rewrite_for_retrieval",  # 先改写 query
            "web_search": "web_search_fallback",
        },
    )

    # 改写后重新检索
    graph.add_edge("query_rewrite_for_retrieval", "retrieve")

    graph.add_edge("generate", END)
    graph.add_edge("web_search_fallback", END)

    return graph
