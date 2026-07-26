"""Utilities for extracting and normalizing packaging text fields from OCR output.

This module is intended to clean OCR-derived text and extract structured product
information such as batch numbers, manufacturing dates, expiry dates, MRP values,
and manufacturer names.
"""

from __future__ import annotations

import re
from typing import Iterable, List


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace into a single space and strip leading/trailing spaces.

    Args:
        text: Raw text to normalize.

    Returns:
        A whitespace-normalized string.
    """
    return re.sub(r"\s+", " ", text or "").strip()


def remove_duplicate_text(lines: Iterable[str]) -> List[str]:
    """Remove duplicate lines while preserving order.

    Args:
        lines: Iterable of OCR text lines.

    Returns:
        A list containing only the first occurrence of each normalized line.
    """
    seen: set[str] = set()
    unique_lines: List[str] = []

    for line in lines:
        cleaned = normalize_whitespace(str(line))
        if not cleaned:
            continue

        lowered = cleaned.lower()
        if lowered in seen:
            continue

        seen.add(lowered)
        unique_lines.append(cleaned)

    return unique_lines


def normalize_date(value: str) -> str:
    """Normalize recognized date strings to a consistent `MM/YYYY` format.

    Supported input patterns include:
    - `MM/YYYY`
    - `MM-YYYY`
    - `YYYY-MM-DD`
    - `DD/MM/YYYY`
    - `DD-MM-YYYY`

    Args:
        value: Raw date string.

    Returns:
        A normalized date string in `MM/YYYY` format, or an empty string if the
        value cannot be parsed.
    """
    text = normalize_whitespace(value or "")
    if not text:
        return ""

    date_formats = (
        "%m/%Y",
        "%m-%Y",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
    )

    for fmt in date_formats:
        try:
            parsed = re.sub(r"\s+", "", text)
            return __import__("datetime").datetime.strptime(parsed, fmt).strftime("%m/%Y")
        except Exception:
            continue

    return ""


def extract_batch_number(text: str) -> str:
    """Extract a batch/lot number from OCR text.

    Args:
        text: Raw OCR text.

    Returns:
        The first matched batch number or an empty string.
    """
    normalized = normalize_whitespace(text)
    patterns = [
        r"\b(?:Batch|Lot|BATCH|LOT)\s*[:#-]?\s*([A-Za-z0-9-]+)",
        r"\b(?:BNo|Batch No|Lot No)\s*[:#-]?\s*([A-Za-z0-9-]+)",
        r"\b([A-Za-z0-9]{4,}-[A-Za-z0-9]{3,})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return ""


def extract_manufacturing_date(text: str) -> str:
    """Extract a manufacturing date from OCR text.

    Args:
        text: Raw OCR text.

    Returns:
        A normalized manufacturing date in `MM/YYYY` format, or an empty string.
    """
    normalized = normalize_whitespace(text)
    pattern = r"\b(?:Mfg|Manufacturing|Made|Prod)\s*[:#-]?\s*([0-9]{1,2}[/-][0-9]{4}|[0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2}|[0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{4})\b"
    match = re.search(pattern, normalized, re.IGNORECASE)
    if not match:
        return ""
    return normalize_date(match.group(1))


def extract_expiry_date(text: str) -> str:
    """Extract an expiry date from OCR text.

    Args:
        text: Raw OCR text.

    Returns:
        A normalized expiry date in `MM/YYYY` format, or an empty string.
    """
    normalized = normalize_whitespace(text)
    pattern = r"\b(?:Expiry|Exp|Best Before|Use By|Valid Until)\s*[:#-]?\s*([0-9]{1,2}[/-][0-9]{4}|[0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2}|[0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{4})\b"
    match = re.search(pattern, normalized, re.IGNORECASE)
    if not match:
        return ""
    return normalize_date(match.group(1))


def extract_mrp(text: str) -> str:
    """Extract the MRP value from OCR text.

    Args:
        text: Raw OCR text.

    Returns:
        The matched MRP string, or an empty string.
    """
    normalized = normalize_whitespace(text)
    pattern = r"\b(?:MRP|Price|Rs|₹)\s*[:=]?\s*([0-9]+(?:\.[0-9]{1,2})?)"
    match = re.search(pattern, normalized, re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def extract_manufacturer(text: str) -> str:
    """Extract the manufacturer name from OCR text.

    Args:
        text: Raw OCR text.

    Returns:
        The matched manufacturer name or an empty string.
    """
    normalized = normalize_whitespace(text)
    patterns = [
        r"\b(?:Manufactured by|Mfd by|Manufacturer|Made by)\s*[:#-]?\s*([A-Za-z0-9 .,&'-]+?)(?=\s*(?:Expiry|Exp|Best Before|Use By|Valid Until|Batch|Lot|BNo|MRP|Price|Mfg|Manufacturing)\b|$)",
        r"\b(?:By)\s*[:#-]?\s*([A-Za-z0-9 .,&'-]+?)(?=\s*(?:Expiry|Exp|Best Before|Use By|Valid Until|Batch|Lot|BNo|MRP|Price|Mfg|Manufacturing)\b|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            manufacturer = normalize_whitespace(match.group(1))
            return manufacturer.strip(" ,;:-")

    return ""
