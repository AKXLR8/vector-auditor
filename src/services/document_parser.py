"""Document parsing for PDF, DOCX, and plain text."""
import io
import logging
from typing import Optional

logger = logging.getLogger("rga_auditor.parser")

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def parse_pdf(content: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(content))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception as e:
            logger.warning("page extraction failed: %s", e)
    return "\n\n".join(parts).strip()


def parse_docx(content: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(content))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def parse_text(content: bytes) -> str:
    return content.decode("utf-8", errors="replace")


def parse_document(filename: str, content: bytes) -> str:
    """Dispatch by extension. Raises ValueError for unsupported types."""
    if len(content) > MAX_FILE_SIZE:
        raise ValueError(f"file too large: {len(content)} bytes (max {MAX_FILE_SIZE})")

    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return parse_pdf(content)
    if name.endswith(".docx"):
        return parse_docx(content)
    if name.endswith((".txt", ".md", ".markdown")):
        return parse_text(content)
    raise ValueError(f"unsupported file type: {filename}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Section-aware chunking: prefer splitting on headers, fall back to char windows."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)
