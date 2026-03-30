"""
生成质量评估模块 (LLM-as-Judge)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from loguru import logger
from openai import OpenAI

from config.prompts import FAITHFULNESS_JUDGE_PROMPT, RELEVANCY_JUDGE_PROMPT
from config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL


class GenerationEvaluator:
    """LLM-as-Judge 生成质量评估器"""

    def __init__(self):
        self.client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
        self.model = LLM_MODEL

    def evaluate_faithfulness(self, answer: str, context: str) -> dict:
        """
        评估忠实度：答案中的信息是否都有文档支撑

        返回: {"score": 0-1, "unsupported_claims": [...]}
        """
        prompt = FAITHFULNESS_JUDGE_PROMPT.format(context=context, answer=answer)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300,
            )
            raw = resp.choices[0].message.content.strip()

            # 尝试解析 JSON
            if "{" in raw:
                import re
                match = re.search(r"\{.*\}", raw, re.DOTALL)
                if match:
                    return json.loads(match.group())

            return {"score": 0.5, "unsupported_claims": [], "raw": raw}

        except Exception as e:
            logger.error(f"Faithfulness 评估失败: {e}")
            return {"score": 0.0, "error": str(e)}

    def evaluate_relevancy(self, query: str, answer: str) -> float:
        """
        评估答案相关性：答案是否回答了用户的问题

        返回: 0-1 分数
        """
        prompt = RELEVANCY_JUDGE_PROMPT.format(query=query, answer=answer)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            raw = resp.choices[0].message.content.strip()
            return float(raw)

        except Exception as e:
            logger.error(f"Relevancy 评估失败: {e}")
            return 0.0

    def evaluate_all(self, query: str, answer: str, context: str) -> dict:
        """综合评估"""
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevancy = self.evaluate_relevancy(query, answer)

        return {
            "query": query[:50],
            "faithfulness": faithfulness.get("score", 0),
            "relevancy": relevancy,
            "unsupported_claims": faithfulness.get("unsupported_claims", []),
        }


def run_generation_evaluation(app, test_cases: list = None):
    """
    运行端到端生成评估

    Args:
        app: 编译后的 LangGraph 应用
        test_cases: 测试用例列表
    """
    from graph.builder import chat

    if test_cases is None:
        test_cases = [
            {"query": "What is the self-attention mechanism in Transformer?"},
            {"query": "How does RAG improve language model performance?"},
            {"query": "What are the key components of a Transformer model?"},
        ]

    evaluator = GenerationEvaluator()

    print("\n" + "=" * 70)
    print("生成质量评估 (LLM-as-Judge)")
    print("=" * 70)

    all_results = []
    for case in test_cases:
        query = case["query"]
        logger.info(f"评估: {query[:40]}")

        # 获取回答
        answer = chat(app, query, thread_id=f"eval-{hash(query)}")

        # 评估（使用回答中的引用上下文）
        result = evaluator.evaluate_all(query, answer, context=answer)
        all_results.append(result)

        print(f"\nQuery: {query}")
        print(f"  Faithfulness: {result['faithfulness']:.2f}")
        print(f"  Relevancy:    {result['relevancy']:.2f}")
        if result["unsupported_claims"]:
            print(f"  无据声明:     {result['unsupported_claims']}")

    # 汇总
    avg_faith = sum(r["faithfulness"] for r in all_results) / len(all_results) if all_results else 0
    avg_rel = sum(r["relevancy"] for r in all_results) / len(all_results) if all_results else 0

    print(f"\n{'='*70}")
    print(f"平均忠实度: {avg_faith:.2f}")
    print(f"平均相关性: {avg_rel:.2f}")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    from graph.builder import create_app

    app = create_app(enable_mcp=False)
    run_generation_evaluation(app)
