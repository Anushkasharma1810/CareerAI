"""
pdf_extractor.py
----------------
Robust PDF text extraction with fallback strategy.

Primary:   PyMuPDF (fitz)  — fast, handles most PDFs
Fallback:  pdfplumber      — better for tabular/layout-heavy PDFs

Handles: invalid PDF, empty PDF, oversized files, encoding issues.
"""

import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── limits ────────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MIN_TEXT_LENGTH = 50  # characters — below this is considered empty


class PDFExtractionError(Exception):
    """Raised for user-facing PDF errors."""


def _extract_with_pymupdf(path: str) -> str:
    """Extract text using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(path)
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        return "\n".join(pages)
    except ImportError:
        return None
    except Exception as e:
        raise PDFExtractionError(f"PyMuPDF extraction failed: {str(e)}")


def _extract_with_pdfplumber(path: str) -> str:
    """Extract text using pdfplumber (fallback)."""
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n".join(pages)
    except ImportError:
        return None
    except Exception as e:
        raise PDFExtractionError(f"pdfplumber extraction failed: {str(e)}")


def _clean_extracted_text(raw: str) -> str:
    """
    Light cleaning pass on extracted PDF text.
    Preserves structure (newlines) but removes encoding artifacts.
    """
    if not raw:
        return ""

    # Remove null bytes and control characters (except newline/tab)
    text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", raw)

    # Collapse 3+ newlines to 2 (preserve paragraph structure)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove lines that are purely decorative (e.g. "----" "====")
    lines = [
        line for line in text.split("\n")
        if not re.fullmatch(r"[\-=_*#|~ ]{3,}", line.strip())
    ]
    text = "\n".join(lines)

    return text.strip()


def extract_text_from_pdf(file_path: str | Path) -> dict:
    """
    Extract text from a PDF file.

    Parameters
    ----------
    file_path : str | Path  Path to the uploaded PDF.

    Returns
    -------
    dict:
        text          str   extracted text
        page_count    int   number of pages
        char_count    int   characters extracted
        method        str   which library succeeded
        warnings      list  any non-fatal issues
    """
    path = Path(file_path)
    warnings = []

    # ── file validation ────────────────────────────────────────────────────
    if not path.exists():
        raise PDFExtractionError(f"File not found: {path}")

    file_size = path.stat().st_size
    if file_size == 0:
        raise PDFExtractionError("The uploaded file is empty.")
    if file_size > MAX_FILE_SIZE_BYTES:
        raise PDFExtractionError(
            f"File too large ({file_size / 1024 / 1024:.1f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB."
        )

    suffix = path.suffix.lower()
    if suffix not in (".pdf",):
        raise PDFExtractionError(
            f"Unsupported file type '{suffix}'. Only PDF files are accepted."
        )

    # ── extraction with fallback ───────────────────────────────────────────
    text = None
    method = None
    page_count = 0

    # Try PyMuPDF first
    try:
        import fitz
        doc = fitz.open(str(path))
        page_count = len(doc)
        raw = []
        for page in doc:
            raw.append(page.get_text("text"))
        doc.close()
        text = "\n".join(raw)
        method = "PyMuPDF"
    except ImportError:
        warnings.append("PyMuPDF not available, trying pdfplumber.")
    except Exception as e:
        warnings.append(f"PyMuPDF failed ({e}), trying pdfplumber.")

    # Fallback to pdfplumber
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        try:
            import pdfplumber
            with pdfplumber.open(str(path)) as pdf:
                page_count = len(pdf.pages)
                raw = []
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        raw.append(t)
            text = "\n".join(raw)
            method = "pdfplumber"
        except ImportError:
            warnings.append("pdfplumber not available.")
        except Exception as e:
            warnings.append(f"pdfplumber failed: {e}")

    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        raise PDFExtractionError(
            "Could not extract meaningful text from this PDF. "
            "The file may be scanned/image-based or password-protected. "
            "Please provide a text-based PDF."
        )

    cleaned = _clean_extracted_text(text)

    return {
        "text": cleaned,
        "page_count": page_count,
        "char_count": len(cleaned),
        "method": method,
        "warnings": warnings,
    }


def extract_text_from_bytes(file_bytes: bytes, filename: str = "upload.pdf") -> dict:
    """
    Extract text from raw bytes (e.g. from Flask request.files).
    Writes to a temp file and delegates to extract_text_from_pdf.
    """
    if not filename.lower().endswith(".pdf"):
        raise PDFExtractionError(
            f"Unsupported file type. Only PDF files are accepted (got '{filename}')."
        )

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise PDFExtractionError(
            f"File too large ({len(file_bytes) / 1024 / 1024:.1f} MB). Maximum: {MAX_FILE_SIZE_MB} MB."
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        result = extract_text_from_pdf(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result


# ── test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <path_to_pdf>")
        sys.exit(1)
    result = extract_text_from_pdf(sys.argv[1])
    print(f"Pages    : {result['page_count']}")
    print(f"Method   : {result['method']}")
    print(f"Chars    : {result['char_count']}")
    print(f"Warnings : {result['warnings']}")
    print(f"\nText preview (first 500 chars):\n{result['text'][:500]}")
