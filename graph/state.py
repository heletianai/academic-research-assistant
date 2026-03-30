"""
LangGraph 状态定义
"""
from __future__ import annotations

from typing import Annotated, List

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """LangGraph 全局状态"""

    # ── 对话层 ──────────────────────────────────────────
    messages: Annotated[list, add_messages]  # 对话历史（Reducer 自动追加）

    # ── 查询层 ──────────────────────────────────────────
    query: str                     # 用户原始问题
    rewritten_query: str           # 指代消解后的问题

    # ── 路由层 ──────────────────────────────────────────
    intent: str                    # knowledge / task / chitchat

    # ── RAG 层 ──────────────────────────────────────────
    retrieved_docs: List[dict]     # 检索到的文档 [{content, metadata, score}]
    retrieval_grade: str           # CRAG 评分: relevant / ambiguous / irrelevant
    rag_context: str               # 拼接后的上下文文本
    retrieval_attempts: int        # 检索重试次数（CRAG 用）

    # ── Agent 层 ────────────────────────────────────────
    tool_plan: List[dict]          # 工具调用计划 [{tool, args, purpose}]
    tool_results: List[dict]       # 工具执行结果 [{tool, args, result, purpose}]
    tool_step: int                 # 当前执行到第几步
    reflexion_count: int           # Reflexion 重规划次数
    reflexion_reason: str          # 上一次反思的原因

    # ── 输出层 ──────────────────────────────────────────
    final_answer: str              # 最终回答

    # ── 元数据 ──────────────────────────────────────────
    needs_human_confirm: bool      # 是否需要人类确认
    available_tools: List[dict]    # MCP 动态发现的工具列表
