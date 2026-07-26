"""
AI service wrapper for OCR-based product analysis.

This service accepts processed OCR output, builds a structured prompt,
and sends it to the model for structured JSON analysis.
"""
import json
import re
from backend.services.ocr_postprocessor import ProcessedOCRResult
from functools import lru_cache

import ollama

from backend.prompts import SYSTEM_PROMPT
import logging

logger = logging.getLogger(__name__)

MAX_PROMPT_LINES = 25
MAX_PROMPT_CHARS = 1800
MODEL_NAME = "qwen3.5:4b"

def _build_prompt(processed: ProcessedOCRResult) -> str:
    """
    Build a clean prompt for the LLM using reconstructed OCR text
    and extracted structured fields.
    """

    fields = processed.structured_fields

    prompt = f"""
OCR TEXT

{processed.reconstructed_text}

Structured Fields

Product Name:
{fields.product_name}

Dosage:
{fields.dosage}

Expiry Date:
{fields.expiry_date}

Manufacturing Date:
{fields.manufacturing_date}

Batch Number:
{fields.batch_number}

MRP:
{fields.mrp}

Ingredients:
{", ".join(fields.ingredients)}

OCR Confidence:
{processed.confidence:.2f}
"""

    return prompt.strip()

@lru_cache(maxsize=64)
def _run_analysis(prompt: str) -> dict:
    """
    Send the prompt to Ollama and return parsed JSON.
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

def analyze_product(processed: ProcessedOCRResult) -> dict:
    """
    Analyze processed OCR output using the local Qwen model.

    Parameters
    ----------
    processed : ProcessedOCRResult
        Output of the OCR post-processing pipeline.

    Returns
    -------
    dict
        Structured JSON returned by the LLM or an error payload.
    """

    # ------------------------------------------------------------------
    # Validate OCR output
    # ------------------------------------------------------------------
    if (
        processed is None
        or not processed.cleaned_text
        or not processed.reconstructed_text.strip()
    ):
        return {
            "error": "No OCR text was provided for analysis."
        }

    # ------------------------------------------------------------------
    # Build prompt
    # ------------------------------------------------------------------
    prompt = _build_prompt(processed)

    if not prompt.strip():
        return {
            "error": "Failed to build prompt for analysis."
        }

    # ------------------------------------------------------------------
    # Run LLM
    # ------------------------------------------------------------------
    try:
        response = _run_analysis(prompt)

        if not isinstance(response, dict):
            return {
                "error": "Model returned an unexpected response type."
            }

        return response

    # ------------------------------------------------------------------
    # Invalid JSON
    # ------------------------------------------------------------------
    except json.JSONDecodeError:
        return {
            "error": "Model did not return valid JSON.",
            "prompt": prompt,
        }

    # ------------------------------------------------------------------
    # Ollama / Runtime errors
    # ------------------------------------------------------------------
    except Exception as exc:
        logger.exception("AI analysis failed")

        return {
            "error": str(exc),
        }