"""
Project 1 GEval CoT 4-dimensional scoring (升级 LLM-as-Judge 第二档)。

vs evaluate_ragas.py (RAGAs three metrics):
- 4 dimensions instead of 3 (Faithfulness / Relevance / Coherence / Conciseness)
- CoT prompt: judge reasons step-by-step BEFORE giving score
- Multi-sample averaging (default 3 samples) to reduce variance
- No external library, just LLM call

Why GEval matters:
- Standard LLM-as-Judge has high variance (same answer scored 0.5 / 0.8 / 0.6 across runs)
- GEval (CoT + multi-sample averaging) reduces variance ~30%
- Industry standard for academic RAG evaluation 2024+

Usage:
    python -m tests.evaluate_geval
    python -m tests.evaluate_geval --limit 5

Output:
    benchmarks/geval_results.json
    benchmarks/geval_results.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from loguru import logger

load_dotenv(ROOT / ".env")

from config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL
from openai import OpenAI


# CoT prompt template (4 dimensions)
GEVAL_PROMPT_TEMPLATE = """\
You are an expert academic evaluator. Score the following RAG system answer on dimension: {dimension}.

Definition of {dimension}:
{definition}

Scoring rubric:
{rubric}

Instructions:
1. First, reason step-by-step about how the answer matches the rubric (Chain of Thought).
2. Identify specific evidence from the answer.
3. Then output a score 1-5 (integer).

Strictly output JSON:
{{"reasoning": "<step-by-step CoT, 50-150 words>", "evidence": "<quote from answer>", "score": <int 1-5>}}

=== Query ===
{query}

=== Retrieved Context ===
{context}

=== Generated Answer ===
{answer}

Score this answer for {dimension}.
"""


DIMENSIONS = {
    "faithfulness": {
        "definition": "Does the answer's claims fully derive from the retrieved context, with no fabrication?",
        "rubric": (
            "5: Every claim is directly supported by the context.\n"
            "4: 1-2 minor claims unsupported but plausible.\n"
            "3: Several claims unsupported, mixed with retrieved content.\n"
            "2: Most claims unsupported, answer mostly fabricated.\n"
            "1: Answer ignores context, fully fabricated."
        ),
    },
    "relevance": {
        "definition": "Does the answer directly address the user's query?",
        "rubric": (
            "5: Directly answers every aspect of the query.\n"
            "4: Answers main question, missing 1 minor aspect.\n"
            "3: Partially answers, some off-topic content.\n"
            "2: Mostly off-topic with brief on-topic content.\n"
            "1: Doesn't answer the query at all."
        ),
    },
    "coherence": {
        "definition": "Is the answer logically structured and easy to follow?",
        "rubric": (
            "5: Clear logical flow, well-organized paragraphs, smooth transitions.\n"
            "4: Mostly coherent, 1-2 awkward transitions.\n"
            "3: Some logical gaps, choppy structure.\n"
            "2: Disjointed paragraphs, unclear flow.\n"
            "1: Incoherent, no logical structure."
        ),
    },
    "conciseness": {
        "definition": "Is the answer appropriately concise without redundancy?",
        "rubric": (
            "5: Just right, no padding, every sentence adds value.\n"
            "4: Slight verbosity, 1-2 redundant phrases.\n"
            "3: Notable redundancy, could be 30% shorter.\n"
            "2: Verbose, padded with filler.\n"
            "1: Mostly redundant, key info buried in noise."
        ),
    },
}


def parse_score(text: str) -> dict:
    """Parse JSON from LLM output, fallback to regex."""
    # Try direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Extract JSON block
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Last resort: extract score
    score_match = re.search(r'"score":\s*(\d+)', text)
    score = int(score_match.group(1)) if score_match else 3
    return {"reasoning": text[:200], "evidence": "", "score": score, "_parse_error": True}


def score_one_dimension(
    client: OpenAI,
    model: str,
    dimension: str,
    query: str,
    context: str,
    answer: str,
    n_samples: int = 3,
) -> dict:
    """Score one dimension with n_samples and average."""
    prompt = GEVAL_PROMPT_TEMPLATE.format(
        dimension=dimension,
        definition=DIMENSIONS[dimension]["definition"],
        rubric=DIMENSIONS[dimension]["rubric"],
        query=query,
        context=context[:2000],
        answer=answer[:2000],
    )

    samples = []
    for i in range(n_samples):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3 + i * 0.2,  # Vary temperature for diverse samples
                max_tokens=500,
            )
            parsed = parse_score(resp.choices[0].message.content or "")
            samples.append(parsed)
        except Exception as e:
            samples.append({"score": 3, "reasoning": f"err: {str(e)[:100]}", "_error": True})

    scores = [s.get("score", 3) for s in samples]
    avg = sum(scores) / len(scores) if scores else 3
    return {
        "score": round(avg / 5.0, 3),  # Normalize to 0-1
        "raw_score": round(avg, 2),
        "samples": scores,
        "reasonings": [s.get("reasoning", "")[:150] for s in samples],
    }


def run_geval(test_cases: list, limit: int = None, n_samples: int = 3) -> list:
    """Run GEval on test cases."""
    cases = test_cases[:limit] if limit else test_cases
    logger.info(f"GEval: {len(cases)} cases x 4 dims x {n_samples} samples = {len(cases)*4*n_samples} calls")

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    logger.info("Loading main graph ...")
    from graph.builder import create_app
    from langchain_core.messages import HumanMessage

    app = create_app(enable_mcp=False)

    results = []
    for i, case in enumerate(cases, 1):
        query = case["query"]
        logger.info(f"[{i}/{len(cases)}] {query[:50]}")

        # Get answer + context
        try:
            t0 = time.time()
            available_tools = getattr(app, "_available_tools", [])
            tools_for_state = [{"name": t["name"], "description": t["description"]} for t in available_tools]
            state = app.invoke(
                {
                    "query": query,
                    "messages": [HumanMessage(content=query)],
                    "retrieval_attempts": 0,
                    "tool_step": 0,
                    "reflexion_count": 0,
                    "available_tools": tools_for_state,
                },
                config={"configurable": {"thread_id": f"geval-{i}"}},
            )
            answer = state.get("final_answer", "")
            retrieved = state.get("retrieved_docs", []) or []
            context = "\n".join(d.get("content", "")[:500] for d in retrieved[:3]) or state.get("rag_context", "")
            elapsed = time.time() - t0
        except Exception as e:
            logger.error(f"  graph fail: {e}")
            results.append({"query": query, "status": "error", "error": str(e)[:200]})
            continue

        # Score 4 dimensions
        scores_4d = {}
        for dim in DIMENSIONS:
            logger.info(f"  GEval {dim} x{n_samples} ...")
            scores_4d[dim] = score_one_dimension(
                client, LLM_MODEL, dim, query, context, answer, n_samples=n_samples
            )

        normalized = {d: scores_4d[d]["score"] for d in DIMENSIONS}
        weighted_avg = round(
            normalized["faithfulness"] * 0.30
            + normalized["relevance"] * 0.30
            + normalized["coherence"] * 0.20
            + normalized["conciseness"] * 0.20,
            3,
        )

        logger.info(
            f"  F={normalized['faithfulness']:.2f} R={normalized['relevance']:.2f} "
            f"Co={normalized['coherence']:.2f} Cn={normalized['conciseness']:.2f} | wAvg={weighted_avg}"
        )

        results.append({
            "query": query,
            "answer": answer[:300],
            "scores_4d": scores_4d,
            "normalized": normalized,
            "weighted_average": weighted_avg,
            "elapsed_sec": round(elapsed, 1),
            "status": "ok",
        })

    return results


def write_summary(results: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "geval_results.json"
    md_path = out_dir / "geval_results.md"

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    ok_runs = [r for r in results if r.get("status") == "ok"]
    if ok_runs:
        avg = {
            d: sum(r["normalized"][d] for r in ok_runs) / len(ok_runs)
            for d in DIMENSIONS
        }
        avg_weighted = sum(r["weighted_average"] for r in ok_runs) / len(ok_runs)
    else:
        avg = {d: 0 for d in DIMENSIONS}
        avg_weighted = 0

    lines = [
        "# Project 1 GEval CoT Evaluation\n",
        f"- Total: {len(results)}",
        f"- Successful: {len(ok_runs)}",
        f"- Per-case CoT samples: 3",
        "",
        "## Average 4-Dimension Scores (normalized 0-1)\n",
        f"- **Faithfulness**: {avg['faithfulness']:.3f}",
        f"- **Relevance**: {avg['relevance']:.3f}",
        f"- **Coherence**: {avg['coherence']:.3f}",
        f"- **Conciseness**: {avg['conciseness']:.3f}",
        f"- **Weighted Average**: {avg_weighted:.3f}",
        "",
        "## Per-case detail\n",
        "| # | Query | F | R | Co | Cn | wAvg |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        if r.get("status") == "ok":
            n = r["normalized"]
            lines.append(f"| {i} | {r['query'][:60]} | {n['faithfulness']:.2f} | {n['relevance']:.2f} | {n['coherence']:.2f} | {n['conciseness']:.2f} | {r['weighted_average']:.3f} |")
        else:
            lines.append(f"| {i} | {r.get('query', '')[:60]} | - | - | - | - | error |")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[geval] JSON  -> {json_path}")
    print(f"[geval] MD    -> {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    test_cases_path = ROOT / "tests" / "test_cases.json"
    test_cases = json.loads(test_cases_path.read_text(encoding="utf-8"))
    print(f"[geval] {len(test_cases)} cases (limit={args.limit}, samples={args.samples})")

    results = run_geval(test_cases, limit=args.limit, n_samples=args.samples)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "benchmarks"
    write_summary(results, out_dir)


if __name__ == "__main__":
    main()
