"""
全局配置 - 所有模型、路径、参数统一管理
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── 项目根目录 ──────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent

# ── 加载 .env 文件 ────────────────────────────────────────
load_dotenv(BASE_DIR / ".env")

# ── LLM Provider 切换（OpenRouter / Zhipu）─────────────
# Provider 切换：zhipu (默认免费 GLM-4-Flash) / openrouter (DeepSeek)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "zhipu").lower()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")

if LLM_PROVIDER == "zhipu":
    OPENROUTER_API_KEY = ZHIPU_API_KEY  # 复用 OPENROUTER_API_KEY 变量名（向后兼容）
    OPENROUTER_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
    if not ZHIPU_API_KEY:
        print("[警告] ZHIPU_API_KEY 未设置，LLM 调用将失败。请检查 .env 文件。",
              file=sys.stderr)
else:
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    if not OPENROUTER_API_KEY:
        print("[警告] OPENROUTER_API_KEY 未设置，LLM 调用将失败。请检查 .env 文件。",
              file=sys.stderr)

# ── LangSmith Observability ─────────────────────────────
# LangChain 会自动读 LANGSMITH_TRACING / LANGSMITH_API_KEY / LANGSMITH_PROJECT 环境变量
# 这里显式同步到 LANGCHAIN_* 别名（兼容老版本 SDK），并打印状态供 CLI 启动时查看
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "academic-research-assistant")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

if LANGSMITH_TRACING and LANGSMITH_API_KEY:
    # 同步到 LANGCHAIN_* 别名 (LangChain SDK 两套环境变量都识别)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", LANGSMITH_API_KEY)
    os.environ.setdefault("LANGCHAIN_PROJECT", LANGSMITH_PROJECT)
    os.environ.setdefault("LANGCHAIN_ENDPOINT", LANGSMITH_ENDPOINT)
    print(f"[LangSmith] tracing enabled → project={LANGSMITH_PROJECT}", file=sys.stderr)
elif LANGSMITH_TRACING:
    print("[LangSmith] LANGSMITH_TRACING=true 但 LANGSMITH_API_KEY 缺失，trace 不会上传", file=sys.stderr)

# ── 模型选择 ────────────────────────────────────────────
if LLM_PROVIDER == "zhipu":
    LLM_MODEL = "glm-4-flash"
else:
    LLM_MODEL = "deepseek/deepseek-chat"
EMBEDDING_MODEL = "openai/text-embedding-3-small"  # Always use OpenRouter for embeddings (existing index)

# ── RAG 参数 ────────────────────────────────────────────
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 3

# ── 数据路径 ────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
CHECKPOINT_DB = DATA_DIR / "checkpoints.db"
NOTES_DB = DATA_DIR / "notes.db"

# ── MCP Server 配置 ─────────────────────────────────────
MCP_SERVERS = [
    {
        "name": "arxiv",
        "command": "python",
        "args": [str(BASE_DIR / "mcp_tools" / "servers" / "arxiv_server.py")],
    },
    {
        "name": "scholar",
        "command": "python",
        "args": [str(BASE_DIR / "mcp_tools" / "servers" / "scholar_server.py")],
    },
    {
        "name": "notes",
        "command": "python",
        "args": [str(BASE_DIR / "mcp_tools" / "servers" / "notes_server.py")],
    },
]
