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

# ── OpenRouter API ──────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

if not OPENROUTER_API_KEY:
    print("[警告] OPENROUTER_API_KEY 未设置，LLM 调用将失败。请检查 .env 文件。",
          file=sys.stderr)

# ── 模型选择 ────────────────────────────────────────────
LLM_MODEL = "deepseek/deepseek-chat"
EMBEDDING_MODEL = "openai/text-embedding-3-small"

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
