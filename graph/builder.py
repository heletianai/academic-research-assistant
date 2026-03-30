"""
LangGraph 主图构建器
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from config.settings import CHECKPOINT_DB
from graph.nodes import (
    chitchat_node,
    human_confirm_node,
    intent_classify_node,
    query_rewrite_node,
    reflexion_node,
    route_by_intent,
    should_continue_tools,
    tool_execute_node,
    tool_planning_node,
    tool_synthesis_node,
)
from graph.rag_subgraph import build_rag_subgraph, init_rag_subgraph
from graph.state import AgentState


def build_main_graph(retriever=None, checkpointer=None):
    """
    构建完整的 LangGraph 主图

    流程：
    START → query_rewrite → intent_classify
      ├── knowledge → [RAG SubGraph] → END
      ├── task → tool_plan → human_confirm → tool_execute ↔ continue → synthesize → END
      └── chitchat → END

    Args:
        retriever: 混合检索器（HybridRetriever），用于 RAG 子图
        checkpointer: LangGraph Checkpointer，用于持久化
    """
    graph = StateGraph(AgentState)

    # ── 1. 添加节点 ──────────────────────────────────────

    # 通用节点
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("intent_classify", intent_classify_node)
    graph.add_node("chitchat", chitchat_node)

    # RAG 子图节点
    if retriever is not None:
        init_rag_subgraph(retriever)
    rag_subgraph = build_rag_subgraph()
    graph.add_node("rag", rag_subgraph.compile())

    # Agent 工具节点
    graph.add_node("tool_plan", tool_planning_node)
    graph.add_node("human_confirm", human_confirm_node)
    graph.add_node("tool_execute", tool_execute_node)
    graph.add_node("reflexion", reflexion_node)
    graph.add_node("tool_synthesize", tool_synthesis_node)

    # ── 2. 添加边 ────────────────────────────────────────

    # 入口 → 改写 → 分类
    graph.add_edge(START, "query_rewrite")
    graph.add_edge("query_rewrite", "intent_classify")

    # 意图分类条件路由
    graph.add_conditional_edges(
        "intent_classify",
        route_by_intent,
        {
            "knowledge": "rag",
            "task": "tool_plan",
            "chitchat": "chitchat",
        },
    )

    # RAG → END
    graph.add_edge("rag", END)

    # 闲聊 → END
    graph.add_edge("chitchat", END)

    # Agent 工具链：plan → confirm → execute ↔ continue/reflexion → synthesize → END
    graph.add_edge("tool_plan", "human_confirm")
    graph.add_edge("human_confirm", "tool_execute")
    graph.add_conditional_edges(
        "tool_execute",
        should_continue_tools,
        {
            "continue": "tool_execute",
            "reflexion": "reflexion",
            "synthesize": "tool_synthesize",
        },
    )
    # Reflexion 重规划后重新执行（跳过 human_confirm，因为是自动修正）
    graph.add_edge("reflexion", "tool_execute")
    graph.add_edge("tool_synthesize", END)

    # ── 3. 编译 ──────────────────────────────────────────

    if checkpointer is None:
        # 默认使用 SQLite 持久化
        import sqlite3
        CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    app = graph.compile(checkpointer=checkpointer)
    logger.info("LangGraph 主图编译完成")

    return app


def create_app(papers_dir: str = None, enable_mcp: bool = True):
    """
    一键创建完整应用

    自动完成：加载论文 → 构建索引 → 创建检索器 → 初始化MCP → 编译主图

    Args:
        papers_dir: 论文 PDF 目录路径
        enable_mcp: 是否启用 MCP 工具（默认启用）
    """
    import asyncio
    from config.settings import PAPERS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, MCP_SERVERS
    from rag.document_loader import DocumentLoader, TextSplitter
    from rag.retriever import BM25Retriever, HybridRetriever, VectorRetriever

    papers_path = Path(papers_dir) if papers_dir else PAPERS_DIR
    papers_path.mkdir(parents=True, exist_ok=True)

    # 1. 加载论文
    loader = DocumentLoader()
    raw_docs = loader.load_directory(papers_path)

    retriever = None
    if raw_docs:
        # 2. 切分
        splitter = TextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split(raw_docs)

        # 3. 构建检索器
        bm25 = BM25Retriever(chunks)
        vector = VectorRetriever(chunks)
        retriever = HybridRetriever(bm25, vector)
        logger.info(f"RAG 初始化: {len(raw_docs)} 页 → {len(chunks)} chunks")
    else:
        logger.warning(f"未找到论文，请将 PDF 放到: {papers_path}")

    # 4. 初始化 MCP Aggregator
    available_tools = []
    if enable_mcp:
        try:
            from mcp_tools.aggregator import MCPAggregator
            from graph.nodes import set_mcp_aggregator

            aggregator = MCPAggregator(MCP_SERVERS)
            available_tools = asyncio.run(aggregator.discover_all_tools())
            set_mcp_aggregator(aggregator)
            logger.info(f"MCP 初始化: {len(available_tools)} 个工具")
        except Exception as e:
            logger.warning(f"MCP 初始化失败（Agent 工具不可用）: {e}")

    # 5. 编译主图
    app = build_main_graph(retriever=retriever)

    # 保存工具列表供 chat() 使用
    app._available_tools = available_tools

    return app


def chat(app, query: str, thread_id: str = "default") -> dict:
    """
    对话接口：发送一条消息并获取回复

    Args:
        app: 编译后的 LangGraph 应用
        query: 用户问题
        thread_id: 会话 ID

    Returns:
        dict: {"status": "done"|"needs_confirm", "answer": str, "plan": list}
    """
    config = {"configurable": {"thread_id": thread_id}}

    # 传入 MCP 工具列表
    available_tools = getattr(app, "_available_tools", [])
    tools_for_state = [
        {"name": t["name"], "description": t["description"]}
        for t in available_tools
    ]

    result = app.invoke(
        {
            "query": query,
            "messages": [HumanMessage(content=query)],
            "retrieval_attempts": 0,
            "tool_step": 0,
            "reflexion_count": 0,
            "available_tools": tools_for_state,
        },
        config=config,
    )

    # 检查是否被 interrupt() 暂停（Human-in-the-Loop）
    state = app.get_state(config)
    if state.next:  # 图还有未执行的节点 → 被 interrupt 了
        # 从 interrupt 值中获取确认信息
        interrupt_data = {}
        if state.tasks:
            for task in state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    interrupt_data = task.interrupts[0].value
                    break

        return {
            "status": "needs_confirm",
            "answer": interrupt_data.get("message", "Agent 需要确认操作计划"),
            "plan": interrupt_data.get("plan", []),
        }

    answer = result.get("final_answer", "抱歉，我无法处理这个问题。")
    return {"status": "done", "answer": answer}


def resume_chat(app, decision: str, thread_id: str = "default") -> dict:
    """
    恢复被 interrupt 暂停的图执行（Human-in-the-Loop 第二步）

    Args:
        app: 编译后的 LangGraph 应用
        decision: 用户决定，如 "yes"/"no"
        thread_id: 会话 ID

    Returns:
        dict: {"status": "done", "answer": str}
    """
    from langgraph.types import Command

    config = {"configurable": {"thread_id": thread_id}}

    result = app.invoke(
        Command(resume=decision),
        config=config,
    )

    answer = result.get("final_answer", "操作已完成。")
    return {"status": "done", "answer": answer}
