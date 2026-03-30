"""
CRAG 文档质量评分模块
"""
from __future__ import annotations

from typing import List, Literal, Tuple

from loguru import logger
from openai import OpenAI

from config.prompts import CRAG_GRADING_PROMPT
from config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL
from rag.document_loader import Document

GradeResult = Literal["relevant", "ambiguous", "irrelevant"]


class DocumentGrader:
    """
    CRAG 文档质量评分器
    对检索到的每篇文档评分，决定整体检索质量
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        self.model = LLM_MODEL

    def grade_documents(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
    ) -> Tuple[GradeResult, List[Tuple[Document, float]]]:
        """
        评估检索到的文档集合质量

        返回:
            - overall_grade: 整体评分
            - filtered_docs: 过滤后保留的相关文档
        """
        if not documents:
            return "irrelevant", []

        grades = []
        filtered = []

        for doc, score in documents:
            grade = self._grade_single(query, doc)
            grades.append(grade)
            if grade in ("relevant", "ambiguous"):
                filtered.append((doc, score))

        # 整体评分逻辑：
        # - 超过一半文档 relevant → 整体 relevant
        # - 有 relevant 但不到一半 → ambiguous
        # - 全部 irrelevant → irrelevant
        relevant_count = grades.count("relevant")
        total = len(grades)

        if relevant_count >= total / 2:
            overall = "relevant"
        elif relevant_count > 0 or grades.count("ambiguous") > 0:
            overall = "ambiguous"
        else:
            overall = "irrelevant"

        logger.info(
            f"CRAG 评分: {relevant_count}/{total} relevant, "
            f"overall={overall}, 保留 {len(filtered)} 条"
        )
        return overall, filtered

    def _grade_single(self, query: str, doc: Document) -> GradeResult:
        """对单篇文档评分"""
        prompt = CRAG_GRADING_PROMPT.format(
            query=query,
            document=doc.content[:500]  # 限制长度节省 token
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=20,
            )
            raw = resp.choices[0].message.content.strip().lower()

            for grade in ("relevant", "ambiguous", "irrelevant"):
                if grade in raw:
                    return grade
            return "ambiguous"  # 默认 ambiguous

        except Exception as e:
            logger.warning(f"CRAG 评分失败: {e}")
            return "ambiguous"
