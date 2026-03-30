"""
文档加载与切分模块
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from loguru import logger


@dataclass
class Document:
    """单个文本块的数据结构"""
    content: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        preview = self.content[:60].replace("\n", " ")
        return f"Document('{preview}...', meta={self.metadata})"


class DocumentLoader:
    """文档加载器：支持 PDF / TXT / Markdown"""

    def load(self, file_path: str | Path) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(path)
        elif suffix in (".txt", ".md"):
            return self._load_text(path)
        else:
            raise ValueError(f"不支持的格式: {suffix}，支持 .pdf / .txt / .md")

    def load_directory(self, dir_path: str | Path) -> List[Document]:
        """加载目录下所有支持的文件"""
        dir_path = Path(dir_path)
        all_docs = []
        for ext in ("*.pdf", "*.txt", "*.md"):
            for file in sorted(dir_path.glob(ext)):
                try:
                    docs = self.load(file)
                    all_docs.extend(docs)
                except Exception as e:
                    logger.warning(f"加载 {file.name} 失败: {e}")
        logger.info(f"目录加载完成: {dir_path}, 共 {len(all_docs)} 个文档片段")
        return all_docs

    def _load_pdf(self, path: Path) -> List[Document]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("请安装 PyMuPDF: pip install pymupdf")

        docs = []
        pdf = fitz.open(str(path))
        for page_num, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if text:
                docs.append(Document(
                    content=text,
                    metadata={"source": path.name, "page": page_num + 1}
                ))
        pdf.close()
        logger.info(f"PDF 加载: {path.name}, {len(docs)} 页")
        return docs

    def _load_text(self, path: Path) -> List[Document]:
        text = path.read_text(encoding="utf-8")
        logger.info(f"文本加载: {path.name}, {len(text)} 字符")
        return [Document(content=text, metadata={"source": path.name, "page": 1})]


class TextSplitter:
    """文本切分器：段落优先 + 滑窗 overlap"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents: List[Document]) -> List[Document]:
        all_chunks = []
        for doc in documents:
            chunks = self._split_text(doc.content)
            for i, chunk in enumerate(chunks):
                all_chunks.append(Document(
                    content=chunk,
                    metadata={**doc.metadata, "chunk_id": i}
                ))
        logger.info(f"切分完成: {len(documents)} 文档 → {len(all_chunks)} chunks")
        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        paragraphs = re.split(r"\n{2,}", text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                for sub in self._force_split(para):
                    chunks.append(sub)
            elif len(current_chunk) + len(para) + 1 <= self.chunk_size:
                current_chunk = current_chunk + "\n" + para if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                overlap_text = current_chunk[-self.chunk_overlap:] if current_chunk else ""
                current_chunk = overlap_text + "\n" + para if overlap_text else para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return [c for c in chunks if len(c) >= 20]

    def _force_split(self, text: str) -> List[str]:
        """超长段落强制滑窗切分"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end].strip())
            start += self.chunk_size - self.chunk_overlap
        return [c for c in chunks if len(c) >= 20]
