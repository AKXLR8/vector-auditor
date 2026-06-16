"""Document parsing for PDF, DOCX, PPTX, XLSX, and plain text — powered by MarkItDown."""
import io
import logging
from typing import Optional

logger = logging.getLogger("rga_auditor.parser")

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

_md = None


def _get_markitdown():
    global _md
    if _md is None:
        from markitdown import MarkItDown
        _md = MarkItDown()
    return _md


def _parse_with_markitdown(content: bytes, ext: str = "") -> str:
    md = _get_markitdown()
    result = md.convert_stream(io.BytesIO(content), file_extension=ext or None)
    return result.text_content


def parse_pdf(content: bytes) -> str:
    return _parse_with_markitdown(content, ".pdf")


def parse_pdf_with_pages(content: bytes) -> tuple[str, list[dict]]:
    """Parse PDF and return (full_text, page_ranges) where page_ranges
    contains {start, end, page} character-offset-to-page mappings.
    Uses pypdf (faster than pdfplumber for text extraction)."""
    from pypdf import PdfReader
    pages: list[dict] = []
    texts: list[str] = []
    offset = 0
    reader = PdfReader(io.BytesIO(content))
    for i, page in enumerate(reader.pages):
        t = (page.extract_text() or "").strip()
        if t:
            texts.append(t)
            start = offset
            offset += len(t)
            pages.append({"start": start, "end": offset, "page": i + 1})
            offset += 2  # separator
    full_text = "\n\n".join(texts)
    return full_text, pages


def parse_docx(content: bytes) -> str:
    return _parse_with_markitdown(content, ".docx")


def parse_pptx(content: bytes) -> str:
    return _parse_with_markitdown(content, ".pptx")


def parse_xlsx(content: bytes) -> str:
    return _parse_with_markitdown(content, ".xlsx")


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
    if name.endswith(".pptx"):
        return parse_pptx(content)
    if name.endswith(".xlsx"):
        return parse_xlsx(content)
    if name.endswith((".txt", ".md", ".markdown")):
        return parse_text(content)
    raise ValueError(f"unsupported file type: {filename}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Section-aware chunking: prefer splitting on headers, fall back to char windows."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)
