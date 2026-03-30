"""
MCP Client 封装
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """单个 MCP Server 的客户端封装"""

    def __init__(self, command: str, args: List[str], name: str = "unknown"):
        self.server_params = StdioServerParameters(
            command=command,
            args=args,
        )
        self.name = name
        self._tools_cache: Optional[List[dict]] = None

    @asynccontextmanager
    async def _connect(self):
        """建立到 MCP Server 的连接"""
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        except FileNotFoundError:
            raise RuntimeError(
                f"MCP [{self.name}] Server 启动失败: "
                f"找不到命令 '{self.server_params.command}'。"
                f"请检查 Python 路径是否正确。"
            )
        except PermissionError:
            raise RuntimeError(
                f"MCP [{self.name}] Server 启动失败: "
                f"脚本无执行权限。请运行 chmod +x 相关文件。"
            )

    async def list_tools(self) -> List[dict]:
        """
        获取 Server 提供的所有工具列表

        返回 OpenAI Function Calling 格式的工具定义
        """
        if self._tools_cache is not None:
            return self._tools_cache

        try:
            async with self._connect() as session:
                response = await session.list_tools()

                tools = []
                for tool in response.tools:
                    tools.append({
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema if hasattr(tool, "inputSchema") else {},
                        "server": self.name,
                    })

                self._tools_cache = tools
                logger.info(f"MCP [{self.name}]: 发现 {len(tools)} 个工具")
                return tools

        except Exception as e:
            logger.error(f"MCP [{self.name}] 获取工具列表失败: {e}")
            return []

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """调用 MCP Server 上的工具"""
        try:
            async with self._connect() as session:
                result = await session.call_tool(tool_name, arguments)

                # 提取文本内容
                if result.content:
                    texts = []
                    for block in result.content:
                        if hasattr(block, "text"):
                            texts.append(block.text)
                    return "\n".join(texts) if texts else str(result.content)
                return "工具执行成功但无返回内容"

        except Exception as e:
            logger.error(f"MCP [{self.name}] 调用 {tool_name} 失败: {e}")
            return f"工具调用失败: {e}"
