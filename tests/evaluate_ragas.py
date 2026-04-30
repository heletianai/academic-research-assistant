"""
Project 1 RAGAs 0.4 standard library evaluation: upgrade from LLM-as-Judge to RAGAs three metrics.

vs evaluate_generation.py (old LLM-as-Judge):
- Standard library: Faithfulness / AnswerRelevancy / ContextPrecision
- AnswerRelevancy uses query reconstruction + embedding cosine similarity
- Statement-level claim splitting (Faithfulness splits claims and checks each against context)
- Async batch support

Usage:
    python -m tests.evaluate_ragas
    python -m tests.evaluate_ragas --limit 5

Output:
    benchmarks/ragas_results.json
    benchmarks/ragas_results.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from loguru import logger

load_dotenv(ROOT / ".env")

from config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL


def build_judge_llm():
    """OpenRouter / DeepSeek as RAGAs judge LLM."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=0.0,
    )


def build_judge_embeddings():
    """AnswerRelevancy needs embeddings; use local sentence-transformers."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


async def score_one_case(metrics_dict, query: str, answer: str, contexts: list[str]) -> dict:
    """Score a single case, returns {faithfulness, answer_relevancy, context_precision}."""
    from ragas.dataset_schema import SingleTurnSample

    sample = SingleTurnSample(
        user_input=query,
        response=answer,
        retrieved_contexts=contexts,
        reference=answer,
    )

    scores = {}
    for name, metric in metrics_dict.items():
        try:
            score = await metric.single_turn_ascore(sample)
            scores[name] = float(score) if score is not None else None
        except Exception as e:
            scores[name] = None
            scores[f"{name}_error"] = str(e)[:200]
    return scores


async def run_eval(test_cases: list[dict], limit: int | None = None) -> list[dict]:
    """Run RAGAs three metrics over test_cases."""
    # RAGAs 0.4 legacy API (supports LangchainLLMWrapper / LangchainEmbeddingsWrapper)
    # collections API requires InstructorLLM which is OpenAI-specific
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    logger.info("Initializing RAGAs three metrics (legacy API) ...")
    judge_llm = LangchainLLMWrapper(build_judge_llm())
    judge_emb = LangchainEmbeddingsWrapper(build_judge_embeddings())

    # Bind LLM/embeddings to legacy metric instances
    faithfulness.llm = judge_llm
    answer_relevancy.llm = judge_llm
    answer_relevancy.embeddings = judge_emb
    context_precision.llm = judge_llm

    metrics = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
    }
    logger.info("RAGAs ready")

    cases = test_cases[:limit] if limit else test_cases

    logger.info(f"Loading main graph (for RAG context + answer) ...")
    from graph.builder import create_app

    app = create_app(enable_mcp=False)

    results: list[dict] = []
    for i, case in enumerate(cases, 1):
        query = case["query"]
        logger.info(f"[{i}/{len(cases)}] RAG: {query[:50]}")

        try:
            from graph.builder import chat
            from langchain_core.messages import HumanMessage

            thread_id = f"ragas-{i}"
            config = {"configurable": {"thread_id": thread_id}}
            t0 = time.time()

            # Direct invoke instead of chat() so we can inspect state for retrieved_docs
            available_tools = getattr(app, "_available_tools", [])
            tools_for_state = [
                {"name": t["name"], "description": t["description"]}
                for t in available_tools
            ]
            result_state = app.invoke(
                {
                    "query": query,
                    "messages": [HumanMessage(content=query)],
                    "retrieval_attempts": 0,
                    "tool_step": 0,
                    "reflexion_count": 0,
                    "available_tools": tools_for_state,
                },
                config=config,
            )
            answer = result_state.get("final_answer", "")
            elapsed = time.time() - t0

            # Extract real retrieved contexts (truth source for RAGAs)
            retrieved_docs = result_state.get("retrieved_docs", []) or []
            contexts = [d.get("content", "") for d in retrieved_docs if d.get("content")]
            if not contexts:
                # Fallback to rag_context if retrieved_docs empty
                rag_ctx = result_state.get("rag_context", "")
                contexts = [rag_ctx] if rag_ctx else ["(no retrieved context)"]
        except Exception as e:
            logger.error(f"  Main graph failed: {e}")
            results.append({
                "query": query,
                "status": "error",
                "error": str(e)[:200],
            })
            continue

        logger.info(f"  Scoring RAGAs three metrics ...")
        try:
            scores = await score_one_case(metrics, query, answer, contexts)
        except Exception as e:
            logger.error(f"  RAGAs scoring failed: {e}")
            scores = {"error": str(e)[:200]}

        results.append({
            "query": query,
            "answer": answer[:300],
            "scores": scores,
            "elapsed_sec": round(elapsed, 1),
            "status": "ok" if "error" not in scores else "eval_error",
        })

        if scores.get("faithfulness") is not None:
            logger.info(
                f"  F={scores.get('faithfulness'):.2f} "
                f"AR={scores.get('answer_relevancy'):.2f} "
                f"CP={scores.get('context_precision'):.2f}"
            )

    return results


def write_summary(results: list[dict], out_dir: Path) -> None:
    """Write JSON + markdown."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ragas_results.json"
    md_path = out_dir / "ragas_results.md"

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    ok_runs = [r for r in results if r.get("status") == "ok" and r.get("scores", {}).get("faithfulness") is not None]
    n_ok = len(ok_runs)
    if n_ok > 0:
        avg = {
            d: sum(r["scores"][d] for r in ok_runs if r["scores"].get(d) is not None) / n_ok
            for d in ("faithfulness", "answer_relevancy", "context_precision")
        }
    else:
        avg = {"faithfulness": 0, "answer_relevancy": 0, "context_precision": 0}

    lines = [
        "# Project 1 RAGAs 0.4 Standard Library Evaluation\n",
        f"- Total: {len(results)}",
        f"- Successful: {n_ok}",
        f"- Failed: {len(results) - n_ok}",
        "",
        "## Average Three Metrics\n",
        f"- **Faithfulness**: {avg['faithfulness']:.3f}",
        f"- **Answer Relevancy**: {avg['answer_relevancy']:.3f}",
        f"- **Context Precision**: {avg['context_precision']:.3f}",
        "",
        "## Per-case detail\n",
        "| # | Query | F | AR | CP | Status |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        s = r.get("scores", {})
        f = f"{s.get('faithfulness'):.2f}" if s.get("faithfulness") is not None else "-"
        ar = f"{s.get('answer_relevancy'):.2f}" if s.get("answer_relevancy") is not None else "-"
        cp = f"{s.get('context_precision'):.2f}" if s.get("context_precision") is not None else "-"
        lines.append(f"| {i} | {r['query'][:60]} | {f} | {ar} | {cp} | {r.get('status')} |")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[ragas] JSON  -> {json_path}")
    print(f"[ragas] MD    -> {md_path}")


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N cases")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    test_cases_path = ROOT / "tests" / "test_cases.json"
    test_cases = json.loads(test_cases_path.read_text(encoding="utf-8"))
    print(f"[ragas] {len(test_cases)} test cases (limit={args.limit})")

    results = await run_eval(test_cases, limit=args.limit)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "benchmarks"
    write_summary(results, out_dir)


if __name__ == "__main__":
    asyncio.run(main_async())
