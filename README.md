# Academic Research Assistant

RAG + Multi-Agent + MCP 学术研究智能对话系统，支持知识问答、任务执行、闲聊三种意图自动分流。

## Architecture

```
User Query
  → Query Rewrite → Intent Classification
      ├── Knowledge → [RAG Pipeline]
      │     BM25 + FAISS Hybrid Search → RRF Fusion → CrossEncoder Rerank
      │     → CRAG Grading (relevant / ambiguous / irrelevant)
      │         ├── relevant  → Generate Answer
      │         ├── ambiguous → Query Rewrite Retry (max 3)
      │         └── irrelevant → Web Search Fallback
      ├── Task → Plan-and-Execute Agent
      │     Plan → Human Confirm → Tool Execute ↔ Reflexion → Synthesize
      │     MCP Servers: ArXiv / Semantic Scholar / Notes (9 tools)
      └── Chitchat → Direct Response
```

## Key Features

- **Hybrid Search + Rerank**: BM25 + FAISS vector dual retrieval, RRF (k=60) fusion, CrossEncoder (ms-marco-MiniLM) reranking top-3
- **CRAG Self-Correction**: Three-level degradation strategy to eliminate hallucination on retrieval failure
- **Agent + Reflexion**: Plan-and-Execute with Reflexion self-correction (tool failure → root cause analysis → replan, max 2 rounds)
- **MCP Integration**: ArXiv / Semantic Scholar / Notes — 3 MCP Servers, 9 tools
- **LangGraph Orchestration**: State graph + SQLite Checkpointer for checkpoint recovery

## Evaluation Results

30 test queries, 4-group ablation study:

| Method | Hit@3 | Hit@5 | MRR |
|--------|-------|-------|-----|
| BM25 Only | 93.3% | 100.0% | 0.737 |
| Vector Only | 83.3% | 83.3% | 0.775 |
| Hybrid (RRF) | 90.0% | 96.7% | 0.819 |
| **Hybrid + Reranker** | **93.3%** | **93.3%** | **0.917** |

Generation quality evaluated via LLM-as-Judge (Faithfulness + Relevancy).

## Tech Stack

- **Framework**: LangGraph, LangChain
- **LLM**: DeepSeek (via OpenRouter)
- **Retrieval**: FAISS, BM25 (rank_bm25), CrossEncoder (ms-marco-MiniLM-L-6-v2)
- **MCP**: FastMCP
- **UI**: Streamlit
- **Storage**: SQLite (checkpointer)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key in config/settings.py

# Run the app
streamlit run ui/app.py
```

## Project Structure

```
├── main.py              # Entry point, build main graph
├── graph/               # LangGraph state & nodes
│   ├── state.py         # Graph state definition (incl. Reflexion fields)
│   ├── nodes.py         # Node functions (rewrite, classify, generate, reflexion...)
│   ├── rag_subgraph.py  # RAG sub-graph builder
│   └── builder.py       # Main graph builder with routing
├── rag/                 # RAG components
│   ├── retriever.py     # BM25 + FAISS hybrid retrieval + RRF fusion
│   ├── reranker.py      # CrossEncoder reranker
│   ├── grader.py        # CRAG document grader
│   └── document_loader.py
├── agent/               # Agent components
│   ├── planner.py       # Plan-and-Execute planner
│   └── tool_executor.py # MCP tool executor
├── mcp_tools/           # MCP server integration
│   ├── mcp_client.py    # MCP client
│   └── servers/         # ArXiv, Semantic Scholar, Notes servers
├── config/              # Configuration & prompts
├── tests/               # Evaluation scripts
│   ├── evaluate_retrieval.py  # Hit@K, MRR evaluation
│   ├── evaluate_generation.py # LLM-as-Judge evaluation
│   └── test_cases.json        # 30 test queries
└── ui/app.py            # Streamlit interface
```
