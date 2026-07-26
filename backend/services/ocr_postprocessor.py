"""
OCR Post Processing Service.

This module transforms raw EasyOCR output into clean, structured,
LLM-friendly information.

Pipeline
--------
Raw OCR
    ↓
Cleaning
    ↓
OCR Correction
    ↓
Duplicate Removal
    ↓
Reading Order Reconstruction
    ↓
Structured Field Extraction
    ↓
Confidence Re-ranking
    ↓
Processed OCR Result
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict
from typing import List
from typing import Optional

from backend.schemas import OCRResult
from pathlib import Path
from rapidfuzz import process

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Part 1: Configuration
# -----------------------------------------------------------------------------

MIN_CONFIDENCE = 0.35

FUZZY_MATCH_THRESHOLD = 90

MAX_WORD_DISTANCE = 2

MAX_LINE_GAP = 25

MAX_VERTICAL_GROUPING = 18

ENABLE_MEDICINE_MATCHING = True

ENABLE_REGEX_EXTRACTION = True

ENABLE_FUZZY_DUPLICATES = True

ENABLE_TEXT_CORRECTION = True

# -----------------------------------------------------------------------------
# OCR Corrections
# -----------------------------------------------------------------------------

COMMON_CORRECTIONS = {

    "Lubrlcant": "Lubricant",
    "LubrIcant": "Lubricant",
    "Lubrlcant.": "Lubricant",

    "Eyc": "Eye",

    "Orops": "Drops",

    "Balch": "Batch",

    "Baldh": "Batch",

    "Mlg": "Mfg",

    "Mig": "Mfg",

    "Exo": "Exp",

    "Carboxymethyceluose": "Carboxymethylcellulose",

    "Carboxymelyceluose": "Carboxymethylcellulose",
}

# -----------------------------------------------------------------------------
# Common medicine keywords
# -----------------------------------------------------------------------------

MEDICINE_KEYWORDS = {

    "tablet",
    "capsule",
    "syrup",
    "drops",
    "eye",
    "ointment",
    "gel",
    "cream",
    "oral",
    "sterile",
    "solution",
    "suspension",
    "paracetamol",
    "cetirizine",
    "azithromycin",
    "crocin",
    "dolo",
    "stenlo",
}

# -----------------------------------------------------------------------------
# Regular Expressions
# -----------------------------------------------------------------------------

EXPIRY_PATTERN = re.compile(
    r"(?:EXP|EXPIRY|EXP DATE)?\s*[:\-]?\s*([0-9]{1,2}[/-][0-9]{2,4})",
    re.IGNORECASE,
)
MFG_PATTERN = re.compile(
    r"(?:MFG|MFD|MFG DATE|MFD DATE)?\s*[:\-]?\s*([0-9]{1,2}[/-][0-9]{2,4})",
    re.IGNORECASE,
)

MRP_PATTERN = re.compile(
    r"(?:MRP)\D*([0-9]+(?:\.[0-9]{1,2})?)",
    re.IGNORECASE,
)

DOSAGE_PATTERN = re.compile(
    r"\b[0-9]+(?:\.[0-9]+)?\s*(?:MG|ML|MCG|G)\b",
    re.IGNORECASE,
)
BATCH_PATTERN = re.compile(
    r"(?:BATCH|LOT|B\.?NO|BATCH NO)?[^A-Z0-9]*([A-Z0-9\-]{4,})",
    re.IGNORECASE,
)

# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class StructuredFields:
    """
    Structured information extracted from OCR.
    """

    product_name: str = ""

    manufacturer: str = ""

    dosage: str = ""

    expiry_date: str = ""

    manufacturing_date: str = ""

    batch_number: str = ""

    mrp: str = ""

    ingredients: List[str] = field(default_factory=list)

    warnings: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ProcessedOCRResult:
    """
    Final processed OCR output.
    """

    raw: OCRResult

    cleaned_text: List[str]

    reconstructed_text: str

    structured_fields: StructuredFields

    confidence: float = 0.0
# -----------------------------------------------------------------------------
# Part 2: Text Cleaning Utilities
# -----------------------------------------------------------------------------

WHITESPACE_PATTERN = re.compile(r"\s+")

DECORATIVE_PATTERN = re.compile(
    r"[★•●■□▪◦◆◇◉◎※¤§©®™`~^|<>]+"
)

MULTI_DASH_PATTERN = re.compile(r"[-_]{2,}")

PUNCT_ONLY_PATTERN = re.compile(r"^[^\w]+$")

OCR_CHAR_REPLACEMENTS = {
    "|": "I",
    "§": "S",
    "¥": "Y",
    "€": "E",
    "¢": "C",
}


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple spaces into one.
    """
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def remove_decorative_symbols(text: str) -> str:
    """
    Remove decorative characters commonly found
    on medicine packaging.
    """
    text = DECORATIVE_PATTERN.sub(" ", text)
    text = MULTI_DASH_PATTERN.sub("-", text)
    return normalize_whitespace(text)


def replace_common_characters(text: str) -> str:
    """
    Replace OCR character substitutions.
    """
    for source, target in OCR_CHAR_REPLACEMENTS.items():
        text = text.replace(source, target)

    return text


def apply_dictionary_corrections(text: str) -> str:
    """
    Replace common OCR spelling mistakes.
    """
    words = text.split()

    corrected = [
        COMMON_CORRECTIONS.get(word, word)
        for word in words
    ]

    return " ".join(corrected)


def preserve_measurements(text: str) -> str:
    """
    Normalize medicine measurements.
    """

    text = re.sub(
        r"(\d+)\s*MG\b",
        r"\1 MG",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"(\d+)\s*ML\b",
        r"\1 ML",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"(\d+)\s*MCG\b",
        r"\1 MCG",
        text,
        flags=re.IGNORECASE,
    )

    return text


def preserve_percentages(text: str) -> str:
    """
    Normalize percentage formatting.
    """

    return re.sub(
        r"(\d+(?:\.\d+)?)\s*%",
        r"\1%",
        text,
    )


def preserve_dates(text: str) -> str:
    """
    Normalize OCR date separators.
    """

    text = text.replace("\\", "/")
    text = text.replace(".", "/")

    text = re.sub(
        r"(\d{2})-(\d{2,4})",
        r"\1/\2",
        text,
    )

    return text


def remove_empty_lines(lines: list[str]) -> list[str]:
    """
    Remove blank OCR entries.
    """

    return [
        line
        for line in lines
        if line.strip()
    ]

def remove_garbage(lines: list[str]) -> list[str]:
    """
    Remove useless OCR fragments.
    """

    cleaned: list[str] = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if PUNCT_ONLY_PATTERN.fullmatch(line):
            continue

        if len(line) == 1 and not line.isdigit():
            continue

        cleaned.append(line)

    return cleaned


def clean_line(text: str) -> str:
    """
    Complete cleaning pipeline for a single OCR line.
    """

    if not text:
        return ""

    text = normalize_whitespace(text)

    text = remove_decorative_symbols(text)

    text = replace_common_characters(text)

    if ENABLE_TEXT_CORRECTION:
        text = apply_dictionary_corrections(text)

    text = preserve_measurements(text)

    text = preserve_percentages(text)

    text = preserve_dates(text)

    text = normalize_whitespace(text)

    return text.strip()


def clean_text_lines(lines: list[str]) -> list[str]:
    """
    Clean an entire OCR result.
    """

    cleaned = [
        clean_line(line)
        for line in lines
    ]

    cleaned = remove_empty_lines(cleaned)

    cleaned = remove_garbage(cleaned)

    return cleaned

# -----------------------------------------------------------------------------
# Part 3: OCR Error Correction Engine
# -----------------------------------------------------------------------------

# Common OCR confusion pairs
OCR_CONFUSIONS = {
    "rn": "m",
    "vv": "w",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "mi": "ml",
}

# Words frequently found on medicine labels
PHARMA_TERMS = {
    "lubricant",
    "eye",
    "drops",
    "tablet",
    "capsule",
    "ointment",
    "sterile",
    "solution",
    "suspension",
    "gel",
    "cream",
    "manufacturer",
    "batch",
    "expiry",
    "exp",
    "mfg",
    "date",
    "dosage",
    "contains",
    "composition",
    "carboxymethylcellulose",
}


def normalize_token(token: str) -> str:
    """
    Normalize a token before comparison.
    """
    token = token.lower()
    token = re.sub(r"[^\w]", "", token)
    return token


def correct_known_word(word: str) -> str:
    """
    Correct a word using the predefined dictionary.
    """

    key = word.strip()

    if key in COMMON_CORRECTIONS:
        return COMMON_CORRECTIONS[key]

    return word


def repair_character_confusions(word: str) -> str:
    """
    Repair common OCR character confusions.
    """

    repaired = word

    for wrong, correct in OCR_CONFUSIONS.items():
        repaired = repaired.replace(wrong, correct)

    return repaired

def repair_numeric_context(word: str) -> str:
    """
    Repair OCR mistakes only when the token appears
    to represent a numeric value.

    Examples:
        1O  -> 10
        O.5 -> 0.5
        2O24 -> 2024
        5O0MG -> 500MG

    This avoids corrupting normal words such as
    'Stenlo', 'Lubricant', or 'VelDrop'.
    """

    if not word:
        return word

    # Only attempt repair if the token already contains digits
    if not any(ch.isdigit() for ch in word):
        return word

    repaired = word

    # Letter O inside numeric tokens → zero
    repaired = re.sub(r"(?<=\d)[Oo](?=\d)", "0", repaired)
    repaired = re.sub(r"^[Oo](?=\d)", "0", repaired)
    repaired = re.sub(r"(?<=\d)[Oo]$", "0", repaired)

    # Letter I or l inside numeric tokens → one
    repaired = re.sub(r"(?<=\d)[Il](?=\d)", "1", repaired)
    repaired = re.sub(r"^[Il](?=\d)", "1", repaired)
    repaired = re.sub(r"(?<=\d)[Il]$", "1", repaired)

    return repaired

def normalize_medicine_word(word: str) -> str:
    """
    Normalize capitalization of medicine words.
    """

    if not word:
        return word

    if word.lower() in PHARMA_TERMS:
        return word.title()

    return word


def fix_exp_keywords(text: str) -> str:
    """
    Normalize expiry keywords.
    """

    text = re.sub(
        r"\bExo\b",
        "Exp",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bExpr\b",
        "Exp",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\bExp Date\b",
        "EXP Date",
        text,
        flags=re.IGNORECASE,
    )

    return text


def fix_batch_keywords(text: str) -> str:
    """
    Normalize batch keywords.
    """

    replacements = {
        "Balch": "Batch",
        "Baldh": "Batch",
        "Baldh": "Batch",
        "Balch No": "Batch No",
        "Batch N": "Batch No",
    }

    for src, dst in replacements.items():
        text = re.sub(
            re.escape(src),
            dst,
            text,
            flags=re.IGNORECASE,
        )

    return text


def fix_mfg_keywords(text: str) -> str:
    """
    Normalize manufacturing keywords.
    """

    replacements = {
        "Mlg": "Mfg",
        "Mig": "Mfg",
        "Miq": "Mfg",
        "Mlg Date": "Mfg Date",
    }

    for src, dst in replacements.items():
        text = re.sub(
            re.escape(src),
            dst,
            text,
            flags=re.IGNORECASE,
        )

    return text


def correct_line(line: str) -> str:
    """
    Perform OCR correction on a single line.
    """

    words = []

    for token in line.split():

        token = correct_known_word(token)
        token = repair_numeric_context(token)
        token = repair_character_confusions(token)

        token = normalize_medicine_word(token)

        words.append(token)

    line = " ".join(words)

    line = fix_exp_keywords(line)

    line = fix_batch_keywords(line)

    line = fix_mfg_keywords(line)

    line = normalize_whitespace(line)

    return line


def correct_text(lines: list[str]) -> list[str]:
    """
    Apply OCR correction to every OCR line.
    """

    return [
        correct_line(line)
        for line in lines
    ]

# -----------------------------------------------------------------------------
# Part 4: Fuzzy Duplicate Removal
# -----------------------------------------------------------------------------

from rapidfuzz import fuzz


def normalize_duplicate_key(text: str) -> str:
    """
    Normalize text before duplicate comparison.
    """

    text = text.lower()

    text = re.sub(r"[^\w\s]", " ", text)

    text = normalize_whitespace(text)

    return text


def are_similar(
    left: str,
    right: str,
    threshold: int = FUZZY_MATCH_THRESHOLD,
) -> bool:
    """
    Determine whether two OCR strings should be
    considered duplicates.
    """

    left = normalize_duplicate_key(left)
    right = normalize_duplicate_key(right)

    if left == right:
        return True

    similarity = fuzz.ratio(left, right)

    return similarity >= threshold


def merge_duplicate_lines(
    text: list[str],
    confidence: list[float],
    bounding_boxes: list,
) -> OCRResult:
    """
    Merge OCR duplicates using fuzzy matching.

    Highest confidence detection is preserved.
    """

    if not text:
        return OCRResult()

    merged_text = []
    merged_confidence = []
    merged_boxes = []

    used = set()

    for i in range(len(text)):

        if i in used:
            continue

        best_index = i

        best_conf = confidence[i]

        for j in range(i + 1, len(text)):

            if j in used:
                continue

            if are_similar(text[i], text[j]):

                used.add(j)

                if confidence[j] > best_conf:

                    best_conf = confidence[j]

                    best_index = j

        merged_text.append(text[best_index])

        merged_confidence.append(confidence[best_index])

        merged_boxes.append(
            bounding_boxes[best_index]
        )

    return OCRResult(
        text=merged_text,
        confidence=merged_confidence,
        bounding_boxes=merged_boxes,
    )
def box_center(box):
    """
    Compute center of OCR bounding box.
    """

    xs = [p[0] for p in box]
    ys = [p[1] for p in box]

    return (
        sum(xs) / len(xs),
        sum(ys) / len(ys),
    )


def remove_spatial_duplicates(result: OCRResult) -> OCRResult:
    """
    Remove duplicate OCR detections occupying
    almost identical locations.
    """

    keep_text = []
    keep_conf = []
    keep_boxes = []

    used = set()

    for i, box_i in enumerate(result.bounding_boxes):

        if i in used:
            continue

        cx_i, cy_i = box_center(box_i)

        best = i

        for j in range(i + 1, len(result.bounding_boxes)):

            if j in used:
                continue

            cx_j, cy_j = box_center(
                result.bounding_boxes[j]
            )

            distance = (
                abs(cx_i - cx_j)
                + abs(cy_i - cy_j)
            )

            if distance < 12:

                used.add(j)

                if result.confidence[j] > result.confidence[best]:
                    best = j

        keep_text.append(result.text[best])
        keep_conf.append(result.confidence[best])
        keep_boxes.append(result.bounding_boxes[best])

    return OCRResult(
        text=keep_text,
        confidence=keep_conf,
        bounding_boxes=keep_boxes,
    )
def deduplicate(result: OCRResult) -> OCRResult:
    """
    Complete duplicate removal pipeline.
    """

    if not ENABLE_FUZZY_DUPLICATES:
        return result

    merged = merge_duplicate_lines(
        result.text,
        result.confidence,
        result.bounding_boxes,
    )

    merged = remove_spatial_duplicates(
        merged
    )

    return merged
# -----------------------------------------------------------------------------
# Part 5: Reading Order Reconstruction
# -----------------------------------------------------------------------------

#from operator import itemgetter   --Not used to be deleted


def box_top(box) -> float:
    """
    Return the top-most y coordinate.
    """
    return min(point[1] for point in box)


def box_left(box) -> float:
    """
    Return the left-most x coordinate.
    """
    return min(point[0] for point in box)


def sort_reading_order(result: OCRResult) -> OCRResult:
    """
    Sort OCR detections from top-to-bottom and
    left-to-right.
    """

    if not result.text:
        return result

    entries = list(
        zip(
            result.text,
            result.confidence,
            result.bounding_boxes,
        )
    )

    entries.sort(
        key=lambda item: (
            round(box_top(item[2]) / MAX_VERTICAL_GROUPING),
            box_left(item[2]),
        )
    )

    return OCRResult(
        text=[e[0] for e in entries],
        confidence=[e[1] for e in entries],
        bounding_boxes=[e[2] for e in entries],
    )
def group_into_lines(result: OCRResult) -> list[list[int]]:
    """
    Group OCR detections into text lines based
    on their vertical positions.
    """

    if not result.text:
        return []

    groups: list[list[int]] = []

    current_group = [0]

    current_y = box_top(result.bounding_boxes[0])

    for index in range(1, len(result.text)):

        y = box_top(result.bounding_boxes[index])

        if abs(y - current_y) <= MAX_LINE_GAP:
            current_group.append(index)
        else:
            groups.append(current_group)
            current_group = [index]
            current_y = y

    groups.append(current_group)

    return groups

def reconstruct_text(result: OCRResult) -> str:
    """
    Reconstruct text in natural reading order.
    """

    result = sort_reading_order(result)

    groups = group_into_lines(result)

    lines = []

    for group in groups:

        words = []

        for index in group:
            words.append(result.text[index])

        lines.append(" ".join(words))

    return "\n".join(lines)


def merge_fragmented_words(text: str) -> str:
    """
    Merge common fragmented pharmaceutical words.
    """

    replacements = {

        "Carboxy methyl cellulose":
            "Carboxymethylcellulose",

        "Eye Drops":
            "Eye Drops",

        "Batch No":
            "Batch No",

        "EXP Date":
            "EXP Date",

        "Mfg Date":
            "Mfg Date",
    }

    for source, target in replacements.items():

        text = text.replace(source, target)

    return text
def build_reconstructed_text(result: OCRResult) -> str:
    """
    Complete reading-order reconstruction.
    """

    reconstructed = reconstruct_text(result)

    reconstructed = merge_fragmented_words(
        reconstructed
    )

    reconstructed = normalize_whitespace(
        reconstructed
    )

    return reconstructed
# -----------------------------------------------------------------------------
# Part 6: Structured Field Extraction
# -----------------------------------------------------------------------------

PRODUCT_HINTS = {
    "tablet",
    "capsule",
    "drops",
    "syrup",
    "ointment",
    "gel",
    "cream",
    "solution",
    "eye",
}


def extract_dosage(text: str) -> str:
    """
    Extract dosage such as:
    500 MG
    10 ML
    0.5%
    """

    match = DOSAGE_PATTERN.search(text)

    return match.group(0) if match else ""


def extract_expiry(text: str) -> str:

    patterns = [

        r"EXP(?:IRY)?\s*DATE[: ]*([0-9]{2}[/-][0-9]{4})",

        r"EXP[: ]*([0-9]{2}[/-][0-9]{4})",

        r"([0-9]{2}[/-][0-9]{4})",

        r"([A-Z]{3}\s*[0-9]{4})",
        r"(20[2-9][0-9])",
    ]

    for pattern in patterns:

        m = re.search(
            pattern,
            text,
            re.I,
        )

        if m:
            return m.group(1)

    return ""


def extract_mfg(text: str) -> str:
    """
    Extract manufacturing date.
    """

    match = MFG_PATTERN.search(text)

    return match.group(1) if match else ""


def extract_batch(text: str) -> str:
    """
    Extract batch number.
    """

    match = BATCH_PATTERN.search(text)

    return match.group(1) if match else ""


def extract_mrp(text: str) -> str:
    """
    Extract MRP.
    """

    match = MRP_PATTERN.search(text)

    return match.group(1) if match else ""


def extract_product_name(lines: list[str]) -> str:
    """
    Guess product name.

    Usually highest-confidence line
    without numbers.
    """

    for line in lines:

        lower = line.lower()

        if any(
            keyword in lower
            for keyword in PRODUCT_HINTS
        ):
            continue

        if len(line) < 3:
            continue

        if any(ch.isdigit() for ch in line):
            continue

        return line

    return ""


def extract_structured_fields(
    reconstructed_text: str,
    cleaned_lines: list[str],
) -> StructuredFields:
    """
    Extract structured OCR information.
    """

    fields = StructuredFields()

    fields.product_name = extract_product_name(
        cleaned_lines
    )

    joined = "\n".join(cleaned_lines)

    joined = joined.replace("\n", " ")

    fields.expiry_date = extract_expiry(joined)

    fields.manufacturing_date = extract_mfg(
        reconstructed_text
    )

    fields.batch_number = extract_batch(
        reconstructed_text
    )

    fields.mrp = extract_mrp(
        reconstructed_text
    )

    fields.dosage = extract_dosage(
        reconstructed_text
    )

    return fields

# Part 6.1 : Dictionary loading
_dictionary: list[str] | None = None


def load_dictionary():
    global _dictionary

    if _dictionary is None:

        path = (
            Path(__file__).parent.parent
            / "data"
            / "medicine_dictionary.txt"
        )

        if path.exists():

            _dictionary = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            _dictionary = []

    return _dictionary


def correct_medicine_name(name: str) -> str:

    medicines = load_dictionary()

    if not medicines:
        return name

    match = process.extractOne(name, medicines)

    if match and match[1] >= 90:
        return match[0]

    return name

def process_ocr(result: OCRResult) -> ProcessedOCRResult:
    """
    Complete OCR post-processing pipeline.
    """

    cleaned = clean_text_lines(result.text)

    corrected = correct_text(cleaned)

    corrected_result = OCRResult(
        text=corrected,
        confidence=result.confidence,
        bounding_boxes=result.bounding_boxes,
    )

    corrected_result = deduplicate(corrected_result)

    reconstructed = build_reconstructed_text(
        corrected_result
    )

    fields = extract_structured_fields(
        reconstructed,
        corrected_result.text,
    )

    fields.product_name = correct_medicine_name(
        fields.product_name
    )

    confidence = (
        sum(corrected_result.confidence) / len(corrected_result.confidence)
        if corrected_result.confidence
        else 0.0
    )

    return ProcessedOCRResult(
        raw=result,
        cleaned_text=corrected_result.text,
        reconstructed_text=reconstructed,
        structured_fields=fields,
        confidence=round(confidence, 3),
    )