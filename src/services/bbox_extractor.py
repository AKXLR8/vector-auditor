"""On-demand bbox extraction from PDFs for citation highlighting."""
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("rga_auditor.bbox")


def _find_pdf_path(document_id: str, upload_dir: str = "uploads") -> Optional[Path]:
    matches = sorted(Path(upload_dir).glob(f"{document_id}_*"))
    for m in matches:
        if m.suffix.lower() == ".pdf":
            return m
    return None


def extract_bboxes(document_id: str, page_num: int, quote: str, upload_dir: str = "uploads") -> list[list[float]]:
    pdf_path = _find_pdf_path(document_id, upload_dir)
    bboxes, _, _ = _extract(pdf_path, page_num, quote)
    return bboxes


def extract_bboxes_with_dimensions(document_id: str, page_num: int, quote: str, upload_dir: str = "uploads") -> tuple[list[list[float]], Optional[float], Optional[float]]:
    pdf_path = _find_pdf_path(document_id, upload_dir)
    return _extract(pdf_path, page_num, quote)


def _extract(pdf_path: Optional[Path], page_num: int, quote: str) -> tuple[list[list[float]], Optional[float], Optional[float]]:
    """Returns (bboxes, page_width, page_height)."""
    if pdf_path is None:
        return [], None, None

    try:
        import pdfplumber
    except ImportError:
        logger.warning("bbox: pdfplumber not available")
        return [], None, None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num < 1 or page_num > len(pdf.pages):
                logger.info("bbox: page %d out of range (pages=%d)", page_num, len(pdf.pages))
                return [], None, None
            page = pdf.pages[page_num - 1]
            page_width = float(page.width)
            page_height = float(page.height)
            words = page.extract_words(keep_blank_chars=True, x_tolerance=3)
            if not words:
                logger.info("bbox: no words on page %d", page_num)
                return [], page_width, page_height
    except Exception as e:
        logger.warning("bbox: failed to extract words on page %d: %s", page_num, e)
        return [], None, None

    page_words_text = " ".join(w["text"] for w in words)
    normalized_page = re.sub(r"\s+", "", page_words_text)
    normalized_quote = re.sub(r"\s+", "", quote)

    start = normalized_page.find(normalized_quote)
    if start == -1:
        logger.info("bbox: quote not found on page %d (len=%d)", page_num, len(normalized_quote))
        return [], page_width, page_height
    end = start + len(normalized_quote)

    char_count = 0
    in_quote = False
    bboxes: list[list[float]] = []
    collected: list[float] = []
    for w in words:
        wt = w["text"]
        w_len = len(wt)
        w_start = char_count
        w_end = char_count + w_len
        char_count = w_end + 1
        overlap = w_start < end and w_end > start
        if overlap and not in_quote:
            in_quote = True
            collected = [w["x0"], w["top"], w["x1"], w["bottom"]]
        elif overlap and in_quote:
            collected[0] = min(collected[0], w["x0"])
            collected[1] = min(collected[1], w["top"])
            collected[2] = max(collected[2], w["x1"])
            collected[3] = max(collected[3], w["bottom"])
        elif not overlap and in_quote:
            bboxes.append(collected)
            in_quote = False
            collected = []
    if in_quote and collected:
        bboxes.append(collected)

    logger.info("bbox: page %d → %d bboxes for quote (%d chars)", page_num, len(bboxes), len(quote))
    return bboxes, page_width, page_height
