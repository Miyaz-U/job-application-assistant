"""
Shared utility functions used across agents.
"""
import time
from typing import Callable, TypeVar
from pypdf import PdfReader
import io

T = TypeVar("T")


def call_with_retry(
    fn: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 2.0,
) -> T:
    """
    Calls fn() and retries with exponential backoff if it raises
    an exception (e.g. model overloaded, transient network error).

    Raises the last exception if all retries are exhausted.
    """
    delay = initial_delay
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt == max_retries:
                break
            print(f"[retry] Attempt {attempt} failed ({e}). Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2  # exponential backoff: 2s, 4s, 8s...

    # pyrefly: ignore [bad-raise]
    raise last_error

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts plain text from an uploaded PDF resume."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise ValueError("Could not extract text from the PDF. Try pasting the resume text instead.")
    return text