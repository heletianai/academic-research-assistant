"""
Semantic Scholar 学术工具 MCP Server

提供引用分析、相关论文发现、作者信息查询等功能
使用 Semantic Scholar 免费 API（无需 Key）
"""
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("semantic-scholar")

BASE_URL = "https://api.semanticscholar.org/graph/v1"


@mcp.tool()
def find_related_papers(query: str, limit: int = 5) -> dict:
    """
    搜索与主题相关的论文

    Args:
        query: 搜索主题（如 "large language model agents"）
        limit: 返回数量
    """
    try:
        resp = httpx.get(
            f"{BASE_URL}/paper/search",
            params={
                "query": query,
                "limit": limit,
                "fields": "title,authors,year,citationCount,abstract,externalIds",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        papers = []
        for p in data.get("data", []):
            authors = [a["name"] for a in (p.get("authors") or [])[:3]]
            papers.append({
                "title": p.get("title", ""),
                "authors": authors,
                "year": p.get("year"),
                "citations": p.get("citationCount", 0),
                "abstract": (p.get("abstract") or "")[:200],
                "paper_id": p.get("paperId", ""),
            })

        return {"papers": papers, "total": len(papers)}

    except Exception as e:
        return {"error": str(e), "papers": []}


@mcp.tool()
def get_citations(paper_title: str, limit: int = 5) -> dict:
    """
    获取引用某篇论文的后续论文

    Args:
        paper_title: 论文标题（用于搜索）
        limit: 返回引用数量
    """
    try:
        # 先搜索找到 paper_id
        search_resp = httpx.get(
            f"{BASE_URL}/paper/search",
            params={"query": paper_title, "limit": 1, "fields": "paperId,title"},
            timeout=15,
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()

        if not search_data.get("data"):
            return {"error": f"未找到论文: {paper_title}", "citations": []}

        paper_id = search_data["data"][0]["paperId"]

        # 获取引用
        cite_resp = httpx.get(
            f"{BASE_URL}/paper/{paper_id}/citations",
            params={
                "limit": limit,
                "fields": "title,authors,year,citationCount",
            },
            timeout=15,
        )
        cite_resp.raise_for_status()
        cite_data = cite_resp.json()

        citations = []
        for item in cite_data.get("data", []):
            citing = item.get("citingPaper", {})
            authors = [a["name"] for a in (citing.get("authors") or [])[:3]]
            citations.append({
                "title": citing.get("title", ""),
                "authors": authors,
                "year": citing.get("year"),
                "citations": citing.get("citationCount", 0),
            })

        return {
            "source_paper": search_data["data"][0]["title"],
            "citations": citations,
            "total": len(citations),
        }

    except Exception as e:
        return {"error": str(e), "citations": []}


@mcp.tool()
def get_author_profile(author_name: str) -> dict:
    """
    获取作者信息和发表记录

    Args:
        author_name: 作者姓名（如 "Yoshua Bengio"）
    """
    try:
        resp = httpx.get(
            f"{BASE_URL}/author/search",
            params={"query": author_name, "limit": 1},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("data"):
            return {"error": f"未找到作者: {author_name}"}

        author_id = data["data"][0]["authorId"]

        # 获取详细信息
        detail_resp = httpx.get(
            f"{BASE_URL}/author/{author_id}",
            params={"fields": "name,hIndex,citationCount,paperCount,papers.title,papers.year,papers.citationCount"},
            timeout=15,
        )
        detail_resp.raise_for_status()
        detail = detail_resp.json()

        top_papers = sorted(
            detail.get("papers", []),
            key=lambda p: p.get("citationCount", 0),
            reverse=True,
        )[:5]

        return {
            "name": detail.get("name", ""),
            "h_index": detail.get("hIndex"),
            "total_citations": detail.get("citationCount"),
            "paper_count": detail.get("paperCount"),
            "top_papers": [
                {"title": p["title"], "year": p.get("year"), "citations": p.get("citationCount", 0)}
                for p in top_papers
            ],
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run()
