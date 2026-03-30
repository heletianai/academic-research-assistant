"""
LangGraph 主图节点实现

每个函数是一个节点：接收 State，返回要更新的字段。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt
from loguru import logger
from openai import OpenAI

from config.prompts import (
    CHITCHAT_PROMPT,
    INTENT_CLASSIFY_PROMPT,
    QUERY_REWRITE_PROMPT,
    TOOL_SYNTHESIS_PROMPT,
)
from config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL
from graph.state import AgentState


def _get_llm():
    return OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)


# ══════════════════════════════════════════════════════════
# 1. Query 改写节点
# ══════════════════════════════════════════════════════════

def query_rewrite_node(state: AgentState) -> dict:
    """指代消解：把含指代词的问题改写成独立完整的问题"""
    query = state["query"]
    messages = state.get("messages", [])

    # 首轮无历史，跳过改写
    if len(messages) <= 1:
        logger.info(f"首轮对话，跳过改写: '{query[:30]}'")
        return {"rewritten_query": query}

    # 构建历史文本（最近3轮）
    recent = messages[-6:]
    history_parts = []
    for msg in recent:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        content = msg.content if hasattr(msg, "content") else str(msg)
        history_parts.append(f"{role}: {content}")
    history_text = "\n".join(history_parts)

    client = _get_llm()
    prompt = QUERY_REWRITE_PROMPT.format(history=history_text, query=query)

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,
        )
        rewritten = resp.choices[0].message.content.strip()

        if rewritten != query:
            logger.info(f"Query 改写: '{query}' → '{rewritten}'")
        return {"rewritten_query": rewritten}

    except Exception as e:
        logger.warning(f"改写失败，使用原始 query: {e}")
        return {"rewritten_query": query}


# ══════════════════════════════════════════════════════════
# 2. 意图分类节点
# ══════════════════════════════════════════════════════════

def intent_classify_node(state: AgentState) -> dict:
    """意图分类：决定走 RAG / Agent / 闲聊 哪条路径"""
    query = state.get("rewritten_query") or state["query"]
    client = _get_llm()

    prompt = INTENT_CLASSIFY_PROMPT.format(query=query)
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        raw = resp.choices[0].message.content.strip().lower()

        for intent in ("knowledge", "task", "chitchat"):
            if intent in raw:
                logger.info(f"意图分类: '{query[:30]}' → {intent}")
                return {"intent": intent}

        logger.warning(f"意图未识别: '{raw}'，默认 chitchat")
        return {"intent": "chitchat"}

    except Exception as e:
        logger.error(f"意图分类失败: {e}")
        return {"intent": "chitchat"}


# ══════════════════════════════════════════════════════════
# 3. 闲聊节点
# ══════════════════════════════════════════════════════════

def chitchat_node(state: AgentState) -> dict:
    """闲聊回复：直接用 LLM 生成"""
    query = state.get("rewritten_query") or state["query"]
    client = _get_llm()

    prompt = CHITCHAT_PROMPT.format(query=query)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,  # 闲聊用高温度，更自然
        max_tokens=300,
    )
    answer = resp.choices[0].message.content.strip()

    logger.info(f"闲聊回复: {len(answer)} 字符")
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# ══════════════════════════════════════════════════════════
# 4. Agent 工具规划节点
# ══════════════════════════════════════════════════════════

def tool_planning_node(state: AgentState) -> dict:
    """Plan-and-Execute: 先制定完整计划，再逐步执行"""
    from agent.planner import AgentPlanner

    query = state.get("rewritten_query") or state["query"]
    available_tools = state.get("available_tools", [])

    planner = AgentPlanner()
    plan = planner.create_plan(query, available_tools)

    logger.info(f"工具规划: {len(plan)} 步")
    return {"tool_plan": plan, "tool_step": 0, "tool_results": []}


# ══════════════════════════════════════════════════════════
# 5. Human-in-the-Loop 确认节点
# ══════════════════════════════════════════════════════════

def human_confirm_node(state: AgentState) -> dict:
    """Human-in-the-Loop：暂停图执行，等待用户确认工具调用计划"""
    plan = state.get("tool_plan", [])

    if not plan:
        logger.info("工具计划为空，跳过确认")
        return {"needs_human_confirm": False}

    # 格式化计划供用户审阅
    plan_summary_parts = []
    for i, step in enumerate(plan):
        tool = step.get("tool", "?")
        args = step.get("args", {})
        purpose = step.get("purpose", "")
        plan_summary_parts.append(f"  步骤{i+1}: 调用 {tool}({args}) — {purpose}")
    plan_summary = "\n".join(plan_summary_parts)

    # interrupt() 暂停图执行，将计划展示给用户
    # 用户的回复会作为 resume 值传回
    user_decision = interrupt({
        "type": "human_confirm",
        "message": f"Agent 计划执行以下操作：\n{plan_summary}\n\n请确认是否执行？",
        "plan": plan,
    })

    # 用户拒绝时，清空计划
    if isinstance(user_decision, str) and user_decision.lower() in ("no", "n", "拒绝", "取消"):
        logger.info("用户拒绝了工具调用计划")
        return {
            "tool_plan": [],
            "needs_human_confirm": False,
            "final_answer": "已取消操作。请问还有其他需要帮助的吗？",
            "messages": [AIMessage(content="已取消操作。请问还有其他需要帮助的吗？")],
        }

    logger.info("用户确认了工具调用计划")
    return {"needs_human_confirm": False}


# ══════════════════════════════════════════════════════════
# 7. 工具执行节点
# ══════════════════════════════════════════════════════════

# ToolExecutor 全局单例（在 builder.py 中初始化）
_tool_executor = None


def set_mcp_aggregator(aggregator):
    """设置 MCP Aggregator（保持向后兼容的接口名）"""
    from agent.tool_executor import ToolExecutor
    global _tool_executor
    _tool_executor = ToolExecutor(mcp_aggregator=aggregator)


def _is_tool_error(result: str) -> bool:
    """判断工具执行结果是否为失败"""
    error_indicators = ["失败", "不可用", "超时", "错误", "error", "failed", "timeout", "not found"]
    result_lower = result.lower()
    return any(indicator in result_lower for indicator in error_indicators)


def tool_execute_node(state: AgentState) -> dict:
    """
    逐步执行工具调用计划

    每次只执行当前步骤，通过 should_continue_tools 条件循环实现多步执行。
    实际工具调用委托给 agent/tool_executor.py 的 ToolExecutor。
    """
    plan = state.get("tool_plan", [])
    step = state.get("tool_step", 0)
    results = list(state.get("tool_results", []))

    if step >= len(plan):
        logger.info("计划执行完毕")
        return {"tool_step": step}

    current = plan[step]
    tool_name = current.get("tool", "")
    tool_args = current.get("args", {})
    purpose = current.get("purpose", "")

    logger.info(f"执行步骤 {step+1}/{len(plan)}: {tool_name} - {purpose}")

    # 通过 ToolExecutor 调用 MCP 工具
    if _tool_executor and _tool_executor.is_available:
        result = _tool_executor.execute(tool_name, tool_args)
    else:
        result = f"工具 {tool_name} 不可用（MCP 未连接）"

    results.append({
        "step": step + 1,
        "tool": tool_name,
        "args": tool_args,
        "result": result,
        "purpose": purpose,
    })

    return {"tool_results": results, "tool_step": step + 1}


# ══════════════════════════════════════════════════════════
# 7.5 Reflexion 反思重规划节点
# ══════════════════════════════════════════════════════════

def reflexion_node(state: AgentState) -> dict:
    """
    Reflexion：工具执行失败时，分析错误原因并重新规划

    将失败的执行历史反馈给 LLM，让它生成新的计划。
    最多反思 2 次，防止无限循环。
    """
    import json

    query = state.get("rewritten_query") or state["query"]
    results = state.get("tool_results", [])
    available_tools = state.get("available_tools", [])
    reflexion_count = state.get("reflexion_count", 0) + 1

    # 构建失败历史
    failure_parts = []
    for r in results:
        status = "失败" if _is_tool_error(r["result"]) else "成功"
        failure_parts.append(
            f"步骤{r['step']}: {r['tool']}({r['args']}) → {status}\n结果: {r['result'][:200]}"
        )
    failure_history = "\n\n".join(failure_parts)

    tools_desc_parts = []
    for tool in available_tools:
        tools_desc_parts.append(f"- {tool.get('name', '?')}: {tool.get('description', '')}")
    tools_description = "\n".join(tools_desc_parts) if tools_desc_parts else "暂无可用工具"

    client = _get_llm()
    prompt = (
        f"你是一个 AI Agent 的反思模块。上一次执行计划遇到了问题，请分析失败原因并制定新计划。\n\n"
        f"用户问题：{query}\n\n"
        f"上一次执行历史：\n{failure_history}\n\n"
        f"可用工具：\n{tools_description}\n\n"
        f"要求：\n"
        f"1. 先用一句话分析失败原因\n"
        f"2. 制定新的执行计划（JSON数组，每步包含 tool/args/purpose）\n"
        f"3. 避免重复之前失败的调用方式，尝试不同的参数或工具组合\n"
        f"4. 最多3步\n\n"
        f"输出格式：\n"
        f"失败原因：<一句话>\n"
        f"新计划：\n```json\n[...]\n```"
    )

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
        )
        raw = resp.choices[0].message.content.strip()

        # 提取失败原因
        reason = ""
        if "失败原因：" in raw:
            reason = raw.split("失败原因：")[1].split("\n")[0].strip()

        # 提取新计划
        import re
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            new_plan = json.loads(match.group(1))
            if not isinstance(new_plan, list):
                new_plan = [new_plan]
            new_plan = new_plan[:3]
        else:
            new_plan = []

        logger.info(f"Reflexion (第{reflexion_count}次): 原因='{reason[:50]}', 新计划={len(new_plan)}步")
        return {
            "tool_plan": new_plan,
            "tool_step": 0,
            "tool_results": [],
            "reflexion_count": reflexion_count,
            "reflexion_reason": reason,
        }

    except Exception as e:
        logger.warning(f"Reflexion 失败: {e}")
        return {
            "reflexion_count": reflexion_count,
            "reflexion_reason": f"反思失败: {e}",
        }


# ══════════════════════════════════════════════════════════
# 6. 结果综合节点
# ══════════════════════════════════════════════════════════

def tool_synthesis_node(state: AgentState) -> dict:
    """综合所有工具调用结果，生成最终回答"""
    query = state.get("rewritten_query") or state["query"]
    results = state.get("tool_results", [])

    # 格式化工具调用历史
    history_parts = []
    for r in results:
        history_parts.append(
            f"步骤{r['step']}: 调用 {r['tool']}({r['args']})\n"
            f"目的: {r['purpose']}\n"
            f"结果: {r['result']}"
        )
    tool_history = "\n\n".join(history_parts) if history_parts else "无工具调用结果"

    client = _get_llm()
    prompt = TOOL_SYNTHESIS_PROMPT.format(query=query, tool_history=tool_history)

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000,
    )
    answer = resp.choices[0].message.content.strip()

    logger.info(f"工具结果综合完成: {len(answer)} 字符")
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# ══════════════════════════════════════════════════════════
# 8. 条件路由函数
# ══════════════════════════════════════════════════════════

def route_by_intent(state: AgentState) -> str:
    """根据意图分类结果路由到不同分支"""
    intent = state.get("intent", "chitchat")
    logger.info(f"路由: intent={intent}")
    return intent


def should_continue_tools(state: AgentState) -> str:
    """判断是否需要继续执行工具，或触发 Reflexion 反思重规划"""
    plan = state.get("tool_plan", [])
    step = state.get("tool_step", 0)
    results = state.get("tool_results", [])
    reflexion_count = state.get("reflexion_count", 0)

    if step >= len(plan):
        # 计划执行完毕，检查是否有失败的步骤需要反思
        has_failure = any(_is_tool_error(r["result"]) for r in results)
        if has_failure and reflexion_count < 2:
            logger.info(f"检测到工具执行失败，触发 Reflexion (已反思{reflexion_count}次)")
            return "reflexion"
        return "synthesize"
    if step >= 5:  # 安全上限
        logger.warning("工具执行达到安全上限(5步)")
        return "synthesize"
    return "continue"
