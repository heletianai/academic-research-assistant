"""
Plan-and-Execute Agent 规划器
"""
from __future__ import annotations

import json
import re
from typing import List

from loguru import logger
from openai import OpenAI

from config.prompts import TOOL_PLANNING_PROMPT
from config.settings import LLM_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL


class AgentPlanner:
    """
    工具调用规划器：根据用户问题和可用工具，制定执行计划

    设计决策：
    - 输出 JSON 数组，每项包含 tool/args/purpose
    - 限制最多3步，防止 LLM 幻觉出不存在的工具链
    - 解析失败时返回空计划，不阻塞流程
    """

    def __init__(self):
        self.client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    def create_plan(self, query: str, available_tools: List[dict]) -> List[dict]:
        """
        根据用户问题和可用工具制定计划

        Args:
            query: 用户问题
            available_tools: MCP 动态发现的工具列表 [{name, description}]

        Returns:
            计划列表 [{tool, args, purpose}]
        """
        # 构建工具描述
        tools_desc_parts = []
        for tool in available_tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "")
            tools_desc_parts.append(f"- {name}: {desc}")
        tools_description = "\n".join(tools_desc_parts) if tools_desc_parts else "暂无可用工具"

        prompt = TOOL_PLANNING_PROMPT.format(
            query=query,
            tools_description=tools_description,
        )

        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
            )
            raw = resp.choices[0].message.content.strip()

            # 从 markdown 代码块中提取 JSON
            if "```" in raw:
                match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
                if match:
                    raw = match.group(1)

            plan = json.loads(raw)
            if not isinstance(plan, list):
                plan = [plan]

            # 限制最多3步
            plan = plan[:3]
            logger.info(f"AgentPlanner: 生成 {len(plan)} 步计划")
            return plan

        except Exception as e:
            logger.warning(f"AgentPlanner: 规划失败: {e}")
            return []
