"""
快速测试入口 - 命令行对话

用法：
    python main.py                          # 交互式对话
    python main.py --query "RAG是什么？"     # 单次提问
"""
import os
import sys
from pathlib import Path

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from loguru import logger

# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI学术研究助手")
    parser.add_argument("--query", type=str, help="单次提问")
    parser.add_argument("--papers-dir", type=str, help="论文PDF目录")
    args = parser.parse_args()

    from graph.builder import create_app, chat, resume_chat

    print("=" * 60)
    print("  AI 学术研究助手")
    print("  LangGraph + CRAG + MCP + Human-in-the-Loop")
    print("=" * 60)
    print("正在初始化...")

    app = create_app(papers_dir=args.papers_dir)
    print("初始化完成！\n")

    if args.query:
        # 单次提问模式
        result = chat(app, args.query)
        print(f"\n回答：{result['answer']}")
        return

    # 交互式对话模式
    thread_id = "cli-session"
    print("输入问题开始对话（输入 'quit' 退出）：\n")

    while True:
        try:
            query = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        try:
            result = chat(app, query, thread_id=thread_id)

            if result["status"] == "needs_confirm":
                # Human-in-the-Loop：展示计划，等待确认
                print(f"\n{result['answer']}")
                try:
                    decision = input("\n请确认 (yes/no): ").strip()
                except (EOFError, KeyboardInterrupt):
                    decision = "no"

                result = resume_chat(app, decision, thread_id=thread_id)

            print(f"\n助手: {result['answer']}\n")
        except Exception as e:
            logger.error(f"处理失败: {e}")
            print(f"\n处理出错: {e}\n")


if __name__ == "__main__":
    main()
