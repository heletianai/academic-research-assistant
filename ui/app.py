"""
Streamlit UI - AI 学术研究助手

启动命令：
    streamlit run ui/app.py

功能：
    - 左侧：论文 PDF 上传 + 系统状态
    - 主区：聊天对话界面
    - 显示引用来源和工具调用过程
"""
import os
import sys
from pathlib import Path

# 项目根目录加入 path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import streamlit as st
from loguru import logger

# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")


def init_app():
    """初始化 LangGraph 应用（只在首次加载时运行）"""
    if "app" not in st.session_state:
        with st.spinner("正在初始化系统..."):
            from graph.builder import create_app
            st.session_state.app = create_app(enable_mcp=True)
            st.session_state.messages = []
            st.session_state.thread_id = "streamlit-session"


def upload_papers():
    """论文上传侧边栏"""
    st.sidebar.header("论文管理")

    uploaded_files = st.sidebar.file_uploader(
        "上传论文 PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        papers_dir = ROOT / "data" / "papers"
        papers_dir.mkdir(parents=True, exist_ok=True)

        new_files = []
        for f in uploaded_files:
            save_path = papers_dir / f.name
            if not save_path.exists():
                save_path.write_bytes(f.getvalue())
                new_files.append(f.name)

        if new_files:
            st.sidebar.success(f"已上传 {len(new_files)} 个文件")
            # 重新初始化应用以重建索引
            if st.sidebar.button("重建索引"):
                del st.session_state["app"]
                st.rerun()

    # 显示已有论文
    papers_dir = ROOT / "data" / "papers"
    if papers_dir.exists():
        pdfs = list(papers_dir.glob("*.pdf"))
        if pdfs:
            st.sidebar.markdown("**已加载论文：**")
            for pdf in pdfs:
                st.sidebar.markdown(f"- {pdf.name}")
        else:
            st.sidebar.info("暂无论文，请上传 PDF")


def show_system_info():
    """显示系统信息"""
    st.sidebar.markdown("---")
    st.sidebar.header("系统信息")

    app = st.session_state.get("app")
    if app:
        tools = getattr(app, "_available_tools", [])
        st.sidebar.markdown(f"**MCP 工具数**: {len(tools)}")
        if tools:
            with st.sidebar.expander("查看工具列表"):
                for t in tools:
                    server = t.get("server", "?")
                    st.markdown(f"- `[{server}]` **{t['name']}**")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**技术栈**: LangGraph + CRAG + MCP\n\n"
        "**LLM**: DeepSeek Chat\n\n"
        "**检索**: BM25 + Vector + RRF"
    )


def main():
    st.set_page_config(
        page_title="AI 学术研究助手",
        page_icon="🎓",
        layout="wide",
    )

    st.title("🎓 AI 学术研究助手")
    st.caption("LangGraph + CRAG + MCP | 论文搜索 · 知识问答 · 阅读笔记")

    # 初始化
    init_app()
    upload_papers()
    show_system_info()

    # 聊天历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Human-in-the-Loop 确认区域
    if st.session_state.get("pending_confirm"):
        st.warning("Agent 需要你确认以下操作计划：")
        plan = st.session_state.get("pending_plan", [])
        for i, step in enumerate(plan):
            st.markdown(f"**步骤{i+1}**: 调用 `{step.get('tool', '?')}` — {step.get('purpose', '')}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("确认执行", type="primary"):
                with st.spinner("执行中..."):
                    from graph.builder import resume_chat
                    result = resume_chat(
                        st.session_state.app, "yes",
                        thread_id=st.session_state.thread_id,
                    )
                    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                    st.session_state.pending_confirm = False
                    st.session_state.pending_plan = []
                    st.rerun()
        with col2:
            if st.button("取消"):
                from graph.builder import resume_chat
                result = resume_chat(
                    st.session_state.app, "no",
                    thread_id=st.session_state.thread_id,
                )
                st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                st.session_state.pending_confirm = False
                st.session_state.pending_plan = []
                st.rerun()

    # 用户输入
    if prompt := st.chat_input("输入问题（如：搜索RAG最新论文、Transformer是什么）"):
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成回复
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    from graph.builder import chat
                    result = chat(
                        st.session_state.app,
                        prompt,
                        thread_id=st.session_state.thread_id,
                    )

                    if result["status"] == "needs_confirm":
                        # Human-in-the-Loop：暂停等待用户确认
                        st.session_state.pending_confirm = True
                        st.session_state.pending_plan = result.get("plan", [])
                        st.markdown(result["answer"])
                        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
                        st.rerun()
                    else:
                        st.markdown(result["answer"])
                        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

                except Exception as e:
                    error_msg = f"处理出错: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # 清除对话按钮
    if st.session_state.messages:
        if st.sidebar.button("清除对话"):
            st.session_state.messages = []
            st.session_state.thread_id = f"streamlit-{os.urandom(4).hex()}"
            st.rerun()


if __name__ == "__main__":
    main()
