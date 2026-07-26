"""Extract structured product metadata from OCR results.

This service takes an :class:`backend.schemas.OCRResult` instance, merges and
normalizes OCR lines, applies regex-based extraction helpers, and returns a
validated :class:`backend.schemas.ProductInfo` model.
"""

from __future__ import annotations

import re
from typing import Iterable, List
from backend.schemas import OCRResult, ProductInfo
from backend.services.ocr_postprocessor import ProcessedOCRResult
from backend.utils.regex_utils import (
    extract_batch_number,
    extract_expiry_date,
    extract_manufacturing_date,
    extract_manufacturer,
    extract_mrp,
    normalize_whitespace,
    remove_duplicate_text,
)


def _collapse_ocr_lines(lines: Iterable[str]) -> List[str]:
    """Normalize OCR lines and remove repeated text.

    Args:
        lines: Raw OCR lines.

    Returns:
        A deduplicated list of cleaned OCR lines.
    """
    return remove_duplicate_text(lines)


def _merge_text(lines: Iterable[str]) -> str:
    """Merge OCR lines into a single paragraph for regex processing.

    Args:
        lines: OCR lines to merge.

    Returns:
        A normalized text block.
    """
    cleaned_lines = _collapse_ocr_lines(lines)
    return "\n".join(cleaned_lines)


def _identify_product_name(lines: Iterable[str]) -> str:
    """Infer a likely product name from OCR lines.

    The heuristic prefers the first meaningful line that does not look like a
    date, price, batch, or manufacturer label.

    Args:
        lines: OCR lines.

    Returns:
        A candidate product name string.
    """
    for line in _collapse_ocr_lines(lines):
        lowered = line.lower()

        if any(token in lowered for token in ("manufactured by", "mfg", "manufacturer", "expiry", "exp", "mrp", "batch", "lot", "ingredients", "composition", "price", "best before", "use by")):
            continue

        if re.search(r"\d{1,2}[/-]\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}", line):
            continue

        if re.search(r"\b(?:rs|inr|₹|mrp)\b", lowered):
            continue

        if len(line) < 3:
            continue

        return normalize_whitespace(line)

    return ""


def _infer_category(lines: Iterable[str]) -> str:
    """Infer a coarse product category from the OCR text."""
    combined = " ".join(_collapse_ocr_lines(lines)).lower()

    medicine_keywords = (
        "tablet",
        "capsule",
        "syrup",
        "medicine",
        "pharma",
        "drug",
        "mg",
        "ml",
        "dose",
        "paracetamol",
        "acetaminophen",
    )
    food_keywords = (
        "milk",
        "tea",
        "coffee",
        "chocolate",
        "snack",
        "rice",
        "oil",
        "flour",
        "biscuits",
        "bread",
        "food",
    )

    if any(keyword in combined for keyword in medicine_keywords):
        return "medicine"
    if any(keyword in combined for keyword in food_keywords):
        return "food"
    return ""


def _identify_ingredients(lines: Iterable[str]) -> List[str]:
    """Extract ingredient-related lines from OCR text.

    Args:
        lines: OCR lines.

    Returns:
        A list of ingredient strings.
    """
    ingredients: List[str] = []
    seen: set[str] = set()

    for line in _collapse_ocr_lines(lines):
        lowered = line.lower()
        if "ingredient" in lowered or "composition" in lowered:
            value = re.sub(r"^(ingredients?|composition)\s*[:\-]?\s*", "", line, flags=re.IGNORECASE)
            parts = re.split(r"[,;]", value)
            for part in parts:
                cleaned = normalize_whitespace(part)
                if not cleaned:
                    continue
                lowered_part = cleaned.lower()
                if lowered_part in seen:
                    continue
                seen.add(lowered_part)
                ingredients.append(cleaned)

    if not ingredients:
        for line in _collapse_ocr_lines(lines):
            if any(token in line.lower() for token in ("ingredient", "composition")):
                continue
            if "," in line:
                for part in re.split(r"[,;]", line):
                    cleaned = normalize_whitespace(part)
                    if not cleaned:
                        continue
                    lowered_part = cleaned.lower()
                    if lowered_part in seen:
                        continue
                    seen.add(lowered_part)
                    ingredients.append(cleaned)

    return ingredients

def extract_product_info(processed: ProcessedOCRResult,) -> ProductInfo:
    """Build a structured product profile from OCR output.

    Args:
        ocr_result: OCR text and confidence results returned by the OCR stage.

    Returns:
        A validated :class:`ProductInfo` object containing the extracted fields.
    """
    lines = processed.cleaned_text
    merged_text = processed.reconstructed_text

    manufacturer = extract_manufacturer(merged_text)
    if not manufacturer:
        for line in lines:
            lowered = line.lower()
            if any(token in lowered for token in ("manufactured by", "manufacturer", "mfd by", "made by")):
                manufacturer = normalize_whitespace(
                    re.sub(r"^(manufactured by|manufacturer|mfd by|made by)\s*[:#-]?\s*", "", line, flags=re.IGNORECASE)
                )
                manufacturer = re.sub(r"\b(?:expiry|exp|batch|lot|mrp|mfg|date)\b.*$", "", manufacturer, flags=re.IGNORECASE).strip(" -:")
                if manufacturer:
                    break

    if not manufacturer:
        for line in lines:
            lowered = line.lower()
            if "by" in lowered and not any(token in lowered for token in ("best before", "expiry", "exp", "mrp", "batch", "lot")):
                candidate = normalize_whitespace(line)
                if len(candidate) > 3:
                    manufacturer = candidate
                    break

    fields = processed.structured_fields
    expiry_date = (
        fields.expiry_date
        or extract_expiry_date(merged_text)
    )

    manufacturing_date = (
        fields.manufacturing_date
        or extract_manufacturing_date(merged_text)
    )

    batch_number = (
        fields.batch_number
        or extract_batch_number(merged_text)
    )

    mrp = (
        fields.mrp
        or extract_mrp(merged_text)
    )

    manufacturer = (
        fields.manufacturer
        or extract_manufacturer(merged_text)
    )
    product_name = (
        fields.product_name
        or _identify_product_name(lines)
    )
    if not product_name:
        for line in lines:
            if not any(token in line.lower() for token in ("manufactured by", "manufacturer", "mfd by", "expiry", "exp", "mrp", "batch", "lot", "ingredients", "composition")):
                product_name = normalize_whitespace(line)
                break

    category = _infer_category(lines)
    if not category and product_name:
        category = _infer_category([product_name])

    ingredients = (
        fields.ingredients
        if fields.ingredients
        else _identify_ingredients(lines)
    )

    confidence_score = str(processed.confidence)

    warnings: List[str] = []
    if not mrp:
        warnings.append("MRP not identified")
    if not manufacturing_date:
        warnings.append("Manufacturing date not identified")

    return ProductInfo(
        product_name=product_name,
        category=category,
        manufacturer=manufacturer,
        expiry_date=expiry_date,
        manufacturing_date=manufacturing_date,
        batch_number=batch_number,
        ingredients=ingredients,
        warnings=warnings,
        confidence=confidence_score,
    )
