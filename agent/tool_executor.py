"""
工具执行器：负责调用 MCP 工具并处理结果
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Optional

from loguru import logger


class ToolExecutor:
    """
    MCP 工具执行器

    封装了同步→异步的桥接逻辑，提供统一的调用接口。
    """

    def __init__(self, mcp_aggregator=None):
        self._aggregator = mcp_aggregator

    def set_aggregator(self, aggregator):
        """动态设置 MCP Aggregator（在应用启动时调用）"""
        self._aggregator = aggregator

    @property
    def is_available(self) -> bool:
        return self._aggregator is not None

    def execute(self, tool_name: str, tool_args: dict, timeout: int = 30) -> str:
        """
        执行单个工具调用

        使用 ThreadPoolExecutor + asyncio.run() 桥接同步→异步，
        避免在已有 event loop（如 Streamlit）中出现嵌套 loop 错误。

        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            timeout: 超时秒数

        Returns:
            工具执行结果字符串
        """
        if not self.is_available:
            return f"工具 {tool_name} 不可用（MCP 未连接）"

        try:
            def _run_in_new_loop():
                return asyncio.run(
                    self._aggregator.call_tool(tool_name, tool_args)
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run_in_new_loop)
                result = future.result(timeout=timeout)

            logger.info(f"ToolExecutor: {tool_name} 执行成功")
            return str(result)

        except concurrent.futures.TimeoutError:
            logger.error(f"ToolExecutor: {tool_name} 超时({timeout}s)")
            return f"工具调用超时（{timeout}秒）: {tool_name}"
        except Exception as e:
            logger.error(f"ToolExecutor: {tool_name} 失败: {e}")
            return f"工具调用失败: {e}"
