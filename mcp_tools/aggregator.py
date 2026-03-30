"""
MCP Aggregator：多 Server 工具聚合器
"""
from __future__ import annotations

from typing import Dict, List, Optional

from loguru import logger

from mcp_tools.mcp_client import MCPClient


class MCPAggregator:
    """多 MCP Server 聚合：动态发现工具列表 + 按工具名自动路由到对应 Server"""

    def __init__(self, server_configs: List[dict]):
        """
        Args:
            server_configs: Server 配置列表，每项包含 name/command/args
        """
        self.clients: Dict[str, MCPClient] = {}
        self.tool_to_server: Dict[str, str] = {}
        self.all_tools: List[dict] = []

        for config in server_configs:
            name = config["name"]
            self.clients[name] = MCPClient(
                command=config["command"],
                args=config["args"],
                name=name,
            )

    async def discover_all_tools(self, timeout_per_server: float = 10.0) -> List[dict]:
        """动态发现所有 Server 的工具（带超时）"""
        import asyncio

        self.all_tools = []
        self.tool_to_server = {}

        for name, client in self.clients.items():
            try:
                tools = await asyncio.wait_for(
                    client.list_tools(),
                    timeout=timeout_per_server,
                )
                for tool in tools:
                    tool_name = tool["name"]
                    self.tool_to_server[tool_name] = name
                    self.all_tools.append(tool)
                logger.info(f"Aggregator: [{name}] 注册 {len(tools)} 个工具")
            except asyncio.TimeoutError:
                logger.warning(f"Aggregator: [{name}] 超时({timeout_per_server}s)，跳过")
            except Exception as e:
                logger.warning(f"Aggregator: [{name}] 发现工具失败: {e}")

        logger.info(
            f"Aggregator 初始化完成: {len(self.clients)} 个 Server, "
            f"{len(self.all_tools)} 个工具"
        )
        return self.all_tools

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """
        调用工具（自动路由到正确的 Server）

        Args:
            tool_name: 工具名称
            arguments: 工具参数
        """
        server_name = self.tool_to_server.get(tool_name)
        if server_name is None:
            available = ", ".join(self.tool_to_server.keys())
            return f"未知工具: {tool_name}。可用工具: {available}"

        client = self.clients[server_name]
        logger.info(f"Aggregator: 调用 [{server_name}].{tool_name}({arguments})")

        result = await client.call_tool(tool_name, arguments)
        return result

    def get_tools_for_llm(self) -> List[dict]:
        """
        返回适合传给 LLM 的工具描述列表
        （不包含 server 字段，只有 name 和 description）
        """
        return [
            {"name": t["name"], "description": t["description"]}
            for t in self.all_tools
        ]
