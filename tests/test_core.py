"""
核心模块单元测试

覆盖：
1. Document + TextSplitter - 文本切分逻辑
2. HybridRetriever._doc_key - RRF 去重关键函数
3. 意图路由 - 路由函数逻辑
4. State 结构完整性

运行方式：
    python -m pytest tests/test_core.py -v
    或
    python tests/test_core.py
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


class TestDocument(unittest.TestCase):
    """测试 Document 数据结构"""

    def test_create(self):
        from rag.document_loader import Document
        doc = Document(content="test content", metadata={"source": "a.pdf", "page": 1})
        self.assertEqual(doc.content, "test content")
        self.assertEqual(doc.metadata["source"], "a.pdf")

    def test_default_metadata(self):
        from rag.document_loader import Document
        doc = Document(content="hello")
        self.assertEqual(doc.metadata, {})


class TestTextSplitter(unittest.TestCase):
    """测试文本切分器"""

    def test_basic_split(self):
        from rag.document_loader import TextSplitter, Document
        splitter = TextSplitter(chunk_size=100, chunk_overlap=10)
        docs = [Document(content="a" * 300, metadata={"source": "test", "page": 1})]
        chunks = splitter.split(docs)
        self.assertGreater(len(chunks), 1)

    def test_short_text_no_split(self):
        from rag.document_loader import TextSplitter, Document
        splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
        docs = [Document(content="short text", metadata={"source": "test", "page": 1})]
        # "short text" 小于20字，会被过滤
        # 用稍长的文本
        docs = [Document(content="这是一段超过二十个字符的中文文本用于测试切分逻辑", metadata={"source": "test", "page": 1})]
        chunks = splitter.split(docs)
        self.assertEqual(len(chunks), 1)

    def test_preserves_metadata(self):
        from rag.document_loader import TextSplitter, Document
        splitter = TextSplitter(chunk_size=50, chunk_overlap=5)
        docs = [Document(content="x" * 200, metadata={"source": "paper.pdf", "page": 3})]
        chunks = splitter.split(docs)
        for chunk in chunks:
            self.assertEqual(chunk.metadata["source"], "paper.pdf")
            self.assertEqual(chunk.metadata["page"], 3)
            self.assertIn("chunk_id", chunk.metadata)


class TestHybridRetrieverDocKey(unittest.TestCase):
    """测试 RRF 融合的文档去重 key"""

    def test_same_doc_same_key(self):
        """相同 metadata 的文档应该生成相同的 key"""
        from rag.document_loader import Document
        from rag.retriever import HybridRetriever

        doc1 = Document(content="hello", metadata={"source": "a.pdf", "page": 1, "chunk_id": 0})
        doc2 = Document(content="hello", metadata={"source": "a.pdf", "page": 1, "chunk_id": 0})

        # doc1 和 doc2 是不同的 Python 对象
        self.assertIsNot(doc1, doc2)
        # 但 key 应该相同
        self.assertEqual(
            HybridRetriever._doc_key(doc1),
            HybridRetriever._doc_key(doc2),
        )

    def test_different_doc_different_key(self):
        """不同 metadata 的文档应该生成不同的 key"""
        from rag.document_loader import Document
        from rag.retriever import HybridRetriever

        doc1 = Document(content="hello", metadata={"source": "a.pdf", "page": 1, "chunk_id": 0})
        doc2 = Document(content="world", metadata={"source": "a.pdf", "page": 2, "chunk_id": 0})

        self.assertNotEqual(
            HybridRetriever._doc_key(doc1),
            HybridRetriever._doc_key(doc2),
        )


class TestIntentRouting(unittest.TestCase):
    """测试意图路由函数"""

    def test_route_knowledge(self):
        from graph.nodes import route_by_intent
        result = route_by_intent({"intent": "knowledge"})
        self.assertEqual(result, "knowledge")

    def test_route_task(self):
        from graph.nodes import route_by_intent
        result = route_by_intent({"intent": "task"})
        self.assertEqual(result, "task")

    def test_route_default_chitchat(self):
        from graph.nodes import route_by_intent
        result = route_by_intent({})
        self.assertEqual(result, "chitchat")

    def test_should_continue_done(self):
        from graph.nodes import should_continue_tools
        result = should_continue_tools({"tool_plan": [{"tool": "a"}], "tool_step": 1})
        self.assertEqual(result, "synthesize")

    def test_should_continue_more(self):
        from graph.nodes import should_continue_tools
        result = should_continue_tools({"tool_plan": [{"tool": "a"}, {"tool": "b"}], "tool_step": 1})
        self.assertEqual(result, "continue")

    def test_should_continue_safety_limit(self):
        from graph.nodes import should_continue_tools
        result = should_continue_tools({"tool_plan": [{"tool": "a"}] * 10, "tool_step": 5})
        self.assertEqual(result, "synthesize")


class TestAgentState(unittest.TestCase):
    """测试 State 结构完整性"""

    def test_state_has_required_fields(self):
        from graph.state import AgentState
        import typing
        hints = typing.get_type_hints(AgentState, include_extras=True)
        required = ["messages", "query", "intent", "retrieved_docs",
                     "retrieval_grade", "tool_plan", "final_answer",
                     "needs_human_confirm"]
        for field in required:
            self.assertIn(field, hints, f"AgentState 缺少字段: {field}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
