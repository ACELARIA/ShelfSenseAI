"""AI service wrapper for OCR-based product analysis.

This service accepts raw OCR text, builds a plain OCR prompt, and sends that
text directly to the model for structured JSON analysis.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Iterable

import ollama

from backend.prompts import SYSTEM_PROMPT
from backend.schemas import OCRResult

MAX_PROMPT_LINES = 25
MAX_PROMPT_CHARS = 1800
MODEL_NAME = "qwen3.5:4b"


def _normalize_ocr_lines(lines: Iterable[str]) -> list[str]:
    """Normalize OCR lines into compact, deduplicated English-only text lines."""
    cleaned: list[str] = []
    seen: set[str] = set()

    for line in lines:
        text = re.sub(r"[^\w\s/%\-.,]", " ", str(line))
        text = re.sub(r"\s+", " ", text).strip(" .,-/")
        if not text:
            continue

        if any(ch.isalpha() and not ch.isascii() for ch in text):
            continue

        lowered = text.lower()
        if lowered in seen:
            continue

        seen.add(lowered)
        cleaned.append(text)

        if len(cleaned) >= MAX_PROMPT_LINES:
            break

    return cleaned


def _build_prompt(ocr_result: OCRResult | list[str]) -> str:
    """Build a compact raw OCR prompt for the model.

    Args:
        ocr_result: OCR text output or a plain list of OCR lines.

    Returns:
        A text prompt containing the raw OCR lines.
    """
    if isinstance(ocr_result, OCRResult):
        lines = _normalize_ocr_lines(ocr_result.text)
    else:
        lines = _normalize_ocr_lines(ocr_result)

    prompt_text = "\n".join(lines)
    if len(prompt_text) > MAX_PROMPT_CHARS:
        prompt_text = prompt_text[:MAX_PROMPT_CHARS]

    return f"OCR TEXT:\n\n{prompt_text}"


@lru_cache(maxsize=256)
def _run_analysis(prompt: str) -> dict[str, Any]:
    """Run the Ollama chat request on raw OCR text.

    Args:
        prompt: Prompt string to send to the model.

    Returns:
        Parsed JSON content returned by the model.
    """
    response = ollama.chat(
        model=MODEL_NAME,
        format="json",
        think=False,
        keep_alive=120,
        options={
            "temperature": 0.0,
            "num_predict": 96,
            "top_p": 0.9,
        },
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    content = response["message"]["content"]
    return json.loads(content)


def analyze_product(ocr_result: OCRResult | list[str]) -> dict[str, Any]:
    """Analyze OCR output and return a JSON object.

    Args:
        ocr_result: Raw OCR output or OCRResult instance.

    Returns:
        Parsed analysis result from the model or an error payload.
    """
    prompt = _build_prompt(ocr_result)
    has_content = bool(prompt.strip().replace("OCR TEXT:\n\n", "").strip())

    if not has_content:
        return {
            "error": "No OCR text was provided for analysis.",
        }

    try:
        return _run_analysis(prompt)
    except json.JSONDecodeError:
        return {
            "error": "Model did not return valid JSON",
            "raw_response": prompt,
        }
    except Exception as e:
        return {
            "error": str(e),
        }
