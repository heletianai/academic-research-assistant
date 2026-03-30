"""
阅读笔记管理 MCP Server
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("research-notes")

# 数据库路径
DB_PATH = Path(__file__).parent.parent.parent / "data" / "notes.db"


def _get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_title TEXT NOT NULL,
            paper_id TEXT,
            content TEXT NOT NULL,
            tags TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


@mcp.tool()
def save_reading_note(paper_title: str, content: str, paper_id: str = "", tags: str = "") -> dict:
    """
    保存论文阅读笔记

    Args:
        paper_title: 论文标题
        content: 笔记内容（关键发现、方法总结等）
        paper_id: 论文 ID（ArXiv ID 等，可选）
        tags: 标签（逗号分隔，如 "RAG,retrieval,2024"）
    """
    conn = _get_db()
    now = datetime.now().isoformat()

    conn.execute(
        "INSERT INTO notes (paper_title, paper_id, content, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (paper_title, paper_id, content, tags, now, now),
    )
    conn.commit()
    note_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()

    return {"success": True, "note_id": note_id, "message": f"笔记已保存: {paper_title}"}


@mcp.tool()
def search_notes(keyword: str) -> dict:
    """
    搜索阅读笔记

    Args:
        keyword: 搜索关键词（在标题、内容、标签中查找）
    """
    conn = _get_db()
    cursor = conn.execute(
        """SELECT id, paper_title, paper_id, content, tags, created_at
           FROM notes
           WHERE paper_title LIKE ? OR content LIKE ? OR tags LIKE ?
           ORDER BY created_at DESC
           LIMIT 10""",
        (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"),
    )
    rows = cursor.fetchall()
    conn.close()

    notes = [
        {
            "id": r[0],
            "paper_title": r[1],
            "paper_id": r[2],
            "content": r[3][:200] + "..." if len(r[3]) > 200 else r[3],
            "tags": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]

    return {"notes": notes, "total": len(notes), "keyword": keyword}


@mcp.tool()
def list_all_notes() -> dict:
    """列出所有阅读笔记（按时间倒序）"""
    conn = _get_db()
    cursor = conn.execute(
        """SELECT id, paper_title, paper_id, tags, created_at
           FROM notes ORDER BY created_at DESC LIMIT 20""",
    )
    rows = cursor.fetchall()
    conn.close()

    notes = [
        {"id": r[0], "paper_title": r[1], "paper_id": r[2], "tags": r[3], "created_at": r[4]}
        for r in rows
    ]
    return {"notes": notes, "total": len(notes)}


if __name__ == "__main__":
    mcp.run()
