import json
import re
from functools import lru_cache

import ollama

from backend.prompts import SYSTEM_PROMPT

MAX_PROMPT_LINES = 25
MAX_PROMPT_CHARS = 1800
MODEL_NAME = "qwen3.5:4b"


def _build_prompt(ocr_text):
    if not ocr_text:
        return "OCR TEXT:\n\n"

    cleaned_lines = []
    seen = set()

    for line in ocr_text:
        text = re.sub(r"\s+", " ", str(line)).strip()

        if not text:
            continue

        lowered = text.lower()
        if lowered in seen:
            continue

        seen.add(lowered)
        cleaned_lines.append(text)

        if len(cleaned_lines) >= MAX_PROMPT_LINES:
            break

    prompt_text = "\n".join(cleaned_lines)
    if len(prompt_text) > MAX_PROMPT_CHARS:
        prompt_text = prompt_text[:MAX_PROMPT_CHARS]

    return f"OCR TEXT:\n\n{prompt_text}"


@lru_cache(maxsize=256)
def _run_analysis(prompt: str):
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


def analyze_product(ocr_text):
    prompt = _build_prompt(ocr_text)

    if not prompt.strip().endswith(":\n\n"):
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

    return {
        "error": "No OCR text was provided for analysis.",
    }