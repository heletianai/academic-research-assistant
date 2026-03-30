"""
ArXiv 论文搜索 MCP Server
"""
import arxiv
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("arxiv-scholar")


@mcp.tool()
def search_papers(query: str, max_results: int = 5) -> dict:
    """
    搜索 ArXiv 论文

    Args:
        query: 搜索关键词（如 "retrieval augmented generation"）
        max_results: 最多返回几篇（默认5）
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "authors": [a.name for a in result.authors[:3]],
            "abstract": result.summary[:300] + "..." if len(result.summary) > 300 else result.summary,
            "arxiv_id": result.entry_id.split("/")[-1],
            "published": str(result.published.date()),
            "pdf_url": result.pdf_url,
        })

    return {"papers": papers, "total": len(papers), "query": query}


@mcp.tool()
def get_paper_details(arxiv_id: str) -> dict:
    """
    获取论文详细信息

    Args:
        arxiv_id: ArXiv 论文 ID（如 "2005.11401"）
    """
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])

    results = list(client.results(search))
    if not results:
        return {"error": f"未找到论文: {arxiv_id}"}

    paper = results[0]
    return {
        "title": paper.title,
        "authors": [a.name for a in paper.authors],
        "abstract": paper.summary,
        "published": str(paper.published.date()),
        "updated": str(paper.updated.date()),
        "categories": paper.categories,
        "pdf_url": paper.pdf_url,
        "doi": paper.doi,
        "comment": paper.comment,
    }


@mcp.tool()
def generate_bibtex(arxiv_id: str) -> str:
    """
    生成 BibTeX 引用格式

    Args:
        arxiv_id: ArXiv 论文 ID
    """
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])

    results = list(client.results(search))
    if not results:
        return f"未找到论文: {arxiv_id}"

    paper = results[0]
    authors = " and ".join(a.name for a in paper.authors)
    year = paper.published.year
    key = f"{paper.authors[0].name.split()[-1].lower()}{year}"

    bibtex = f"""@article{{{key},
  title={{{paper.title}}},
  author={{{authors}}},
  journal={{arXiv preprint arXiv:{arxiv_id}}},
  year={{{year}}}
}}"""
    return bibtex


if __name__ == "__main__":
    mcp.run()
