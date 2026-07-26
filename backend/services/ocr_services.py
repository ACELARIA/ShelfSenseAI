"""
Production-grade OCR service for ShelfSense AI.

Features
--------
- Cached EasyOCR reader
- Multi-stage preprocessing
- Multi-orientation OCR
- Confidence filtering
- Duplicate removal
- Reading-order sorting
- Medicine-aware text cleaning
- CPU optimized
- Safe exception handling

Public API
----------
extract_text(image_source) -> OCRResult
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple
from typing import Union

import cv2
import easyocr
import numpy as np

from backend.schemas import OCRResult

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# OCR Configuration
# -----------------------------------------------------------------------------

MIN_CONFIDENCE = 0.35

MIN_TEXT_LENGTH = 2

BEAMSEARCH = "beamsearch"

GREEDY = "greedy"

MAG_RATIO = 1.5

WIDTH_THRESHOLD = 0.7

MIN_SIZE = 25

CONTRAST_THRESHOLD = 0.15

ADJUST_CONTRAST = 0.8

CLAHE_CLIP_LIMIT = 2.0

CLAHE_GRID_SIZE = (8, 8)

DENOISE_STRENGTH = 10

SHARPEN_KERNEL = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ],
    dtype=np.float32,
)

# -----------------------------------------------------------------------------
# Common OCR corrections
# -----------------------------------------------------------------------------

OCR_REPLACEMENTS = {
    "OCT0BER": "OCTOBER",
    "0CT": "OCT",
    "JANUARYY": "JANUARY",
    "MFG,": "MFG",
    "EXP,": "EXP",
    "MRP.": "MRP",
    "0MG": "0 MG",
    "1MG": "1 MG",
    "5MG": "5 MG",
    "500MG": "500 MG",
    "250MG": "250 MG",
    "650MG": "650 MG",
}

# Decorative symbols removed from OCR

DECORATIVE_PATTERN = re.compile(
    r"[^\w\s:/%.,()\-+]"
)

# Garbage detections

GARBAGE_PATTERN = re.compile(
    r"^[\W_]+$"
)

# Normalize duplicate keys

NORMALIZE_PATTERN = re.compile(
    r"[^A-Za-z0-9]+"
)

# -----------------------------------------------------------------------------
# EasyOCR Reader
# -----------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_reader() -> easyocr.Reader:
    """
    Create a cached EasyOCR reader.

    The reader is created only once during the application's lifetime.
    """

    try:
        import torch

        gpu = torch.cuda.is_available()

    except Exception:
        gpu = False

    logger.info("Initializing EasyOCR (GPU=%s)", gpu)

    return easyocr.Reader(
        ["en"],
        gpu=gpu,
        verbose=False,
    )

# -----------------------------------------------------------------------------
# Image Loading
# -----------------------------------------------------------------------------


def _load_image(
    image_source: Union[str, Path, np.ndarray]
) -> np.ndarray:
    """
    Load an image from disk or numpy array.

    Raises
    ------
    ValueError
        If image cannot be loaded.
    """

    if isinstance(image_source, np.ndarray):
        image = image_source.copy()

    elif isinstance(image_source, (str, Path)):
        image = cv2.imread(
            str(image_source),
            cv2.IMREAD_COLOR,
        )

    else:
        raise ValueError("Unsupported image source.")

    if image is None:
        raise ValueError("Unable to load image.")

    return image

# -----------------------------------------------------------------------------
# Image Enhancement
# -----------------------------------------------------------------------------


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_GRID_SIZE,
    )

    return clahe.apply(gray)


def _adaptive(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        11,
    )


def _otsu(gray: np.ndarray) -> np.ndarray:
    _, result = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return result


def _denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(
        gray,
        None,
        DENOISE_STRENGTH,
        7,
        21,
    )


def _sharpen(gray: np.ndarray) -> np.ndarray:
    return cv2.filter2D(
        gray,
        -1,
        SHARPEN_KERNEL,
    )    

def _close(gray):

    kernel = np.ones((2, 2), np.uint8)

    return cv2.morphologyEx(
        gray,
        cv2.MORPH_CLOSE,
        kernel,
    )
# -----------------------------------------------------------------------------
# Image Variant Generation
# -----------------------------------------------------------------------------

def _rotate(image: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotate image by 90°, 180° or 270°.
    """

    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)

    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


def _generate_variants(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate multiple enhanced versions of the image.

    Running OCR on multiple variants dramatically improves
    recognition across different lighting conditions,
    medicine packages, glossy labels and faded text.
    """

    gray = _to_gray(image)
    if gray.shape[1] < 1000:

        gray = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )

    clahe = _clahe(gray)

    denoise = _denoise(gray)

    sharpen = _sharpen(gray)

    adaptive = _adaptive(gray)

    otsu = _otsu(gray)

    closed = _close(gray)

    return {

        "original": image,

        "gray": gray,

        "clahe": clahe,

        "denoise": denoise,

        "sharpen": sharpen,

        "adaptive": adaptive,

        "otsu": otsu,

        "closed": closed,

        "rot90": _rotate(gray, 90),

        "rot180": _rotate(gray, 180),

        "rot270": _rotate(gray, 270),
    }


# -----------------------------------------------------------------------------
# OCR Execution
# -----------------------------------------------------------------------------

def _read_variant(
    reader: easyocr.Reader,
    image: np.ndarray,
    decoder: str,
):
    """
    Execute OCR using a specified decoder.

    Parameters
    ----------
    reader
        Cached EasyOCR reader.

    image
        Preprocessed image.

    decoder
        beamsearch or greedy
    """

    return reader.readtext(

        image,

        detail=1,

        paragraph=False,

        decoder=decoder,

        min_size=MIN_SIZE,

        contrast_ths=CONTRAST_THRESHOLD,

        adjust_contrast=ADJUST_CONTRAST,

        width_ths=WIDTH_THRESHOLD,

        mag_ratio=2.0,
    )


def _run_easyocr(
    reader: easyocr.Reader,
    image: np.ndarray,
):
    """
    Run OCR with Beam Search first.

    If nothing useful is detected,
    automatically retry using Greedy decoding.
    """

    try:

        results = _read_variant(
            reader,
            image,
            BEAMSEARCH,
        )

        if results:
            return results

        return _read_variant(
            reader,
            image,
            GREEDY,
        )

    except Exception as exc:

        logger.exception(
            "EasyOCR failed: %s",
            exc,
        )

        return []


# -----------------------------------------------------------------------------
# OCR Extraction
# -----------------------------------------------------------------------------

def _extract_variant_results(
    reader: easyocr.Reader,
    image: np.ndarray,
):
    """
    Convert EasyOCR output into
    simple tuples for downstream processing.
    """

    detections = []

    results = _run_easyocr(
        reader,
        image,
    )

    for box, text, confidence in results:

        detections.append(

            (
                box,
                str(text),
                float(confidence),
            )

        )

    return detections


def _collect_all_results(
    reader: easyocr.Reader,
    variants: Dict[str, np.ndarray],
):
    """
    Perform OCR on every generated variant.

    Returns one merged list.
    """

    merged = []

    for variant_name, variant_image in variants.items():

        logger.debug(
            "Running OCR on %s",
            variant_name,
        )

        merged.extend(

            _extract_variant_results(
                reader,
                variant_image,
            )

        )

    return merged
# -----------------------------------------------------------------------------
# Text Cleaning
# -----------------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def _apply_replacements(text: str) -> str:
    """Apply common OCR corrections."""
    for old, new in OCR_REPLACEMENTS.items():
        text = re.sub(
            rf"\b{re.escape(old)}\b",
            new,
            text,
            flags=re.IGNORECASE,
        )
    return text


def _preserve_dosages(text: str) -> str:
    """
    Normalize medicine dosage units.

    Examples:
        500MG  -> 500 MG
        10ML   -> 10 ML
        2GM    -> 2 GM
    """

    text = re.sub(
        r"(\d+)\s*(MG|ML|GM|MCG|KG|G|L)\b",
        r"\1 \2",
        text,
        flags=re.IGNORECASE,
    )

    return text


def _preserve_percentages(text: str) -> str:
    """
    Normalize percentage formatting.

    Example:
        5 % -> 5%
    """

    return re.sub(
        r"(\d+)\s*%",
        r"\1%",
        text,
    )


def _preserve_dates(text: str) -> str:
    """
    Normalize expiry/manufacturing dates.

    Examples:
        EXP12/26
        EXP : 12/26
        MFG01/24
    """

    text = re.sub(
        r"(EXP|MFG|PKD)\s*:?\s*",
        lambda m: m.group(1).upper() + " ",
        text,
        flags=re.IGNORECASE,
    )

    return text


def _preserve_batch_numbers(text: str) -> str:
    """
    Normalize batch numbers.

    Example:
        BatchNo123
        Batch No 123
    """

    text = re.sub(
        r"BATCH\s*NO",
        "BATCH NO",
        text,
        flags=re.IGNORECASE,
    )

    return text


def _remove_decorations(text: str) -> str:
    """
    Remove decorative symbols while preserving useful characters.
    """

    return DECORATIVE_PATTERN.sub(" ", text)


def _clean_text(text: str) -> str:
    """
    Complete OCR cleaning pipeline.
    """

    text = _normalize_whitespace(text)

    text = _apply_replacements(text)

    text = _remove_decorations(text)

    text = _preserve_dosages(text)

    text = _preserve_percentages(text)

    text = _preserve_dates(text)

    text = _preserve_batch_numbers(text)

    text = _normalize_whitespace(text)

    return text


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def _is_valid_detection(
    text: str,
    confidence: float,
) -> bool:
    """
    Determine whether an OCR detection should be kept.
    """

    if confidence < MIN_CONFIDENCE:
        return False

    if not text:
        return False

    if len(text) < MIN_TEXT_LENGTH:
        return False

    if GARBAGE_PATTERN.fullmatch(text):
        return False

    return True


# -----------------------------------------------------------------------------
# Duplicate Handling
# -----------------------------------------------------------------------------

def _normalize_key(text: str) -> str:
    """
    Normalize text used for duplicate comparison.
    """

    key = NORMALIZE_PATTERN.sub(
        "",
        text.upper(),
    )

    return key


def _merge_duplicate_results(
    detections: Iterable[Tuple],
):
    """
    Keep only the highest-confidence duplicate.
    """

    best = {}

    for box, text, conf in detections:

        cleaned = _clean_text(text)

        if not _is_valid_detection(
            cleaned,
            conf,
        ):
            continue

        key = _normalize_key(cleaned)

        if not key:
            continue

        if key not in best:

            best[key] = (
                box,
                cleaned,
                conf,
            )

            continue

        if conf > best[key][2]:

            best[key] = (
                box,
                cleaned,
                conf,
            )

    return list(best.values())
# -----------------------------------------------------------------------------
# Bounding Box Utilities
# -----------------------------------------------------------------------------

def _box_top_left(box: List[List[float]]) -> Tuple[float, float]:
    """
    Return (y, x) coordinates of the upper-left corner.
    """

    if not box:
        return (0.0, 0.0)

    x = min(point[0] for point in box)
    y = min(point[1] for point in box)

    return (y, x)


def _box_center(box: List[List[float]]) -> Tuple[float, float]:
    """
    Compute bounding box center.
    """

    xs = [p[0] for p in box]
    ys = [p[1] for p in box]

    return (
        float(sum(xs) / len(xs)),
        float(sum(ys) / len(ys)),
    )


def _box_height(box: List[List[float]]) -> float:
    """
    Approximate bounding-box height.
    """

    ys = [p[1] for p in box]

    return max(ys) - min(ys)


# -----------------------------------------------------------------------------
# Reading Order Reconstruction
# -----------------------------------------------------------------------------

def _sort_reading_order(
    detections: List[Tuple],
) -> List[Tuple]:
    """
    Sort OCR results in natural reading order.

    Top-to-bottom.
    Left-to-right.

    Nearby text lines are grouped together.
    """

    if not detections:
        return []

    average_height = np.mean(
        [
            max(_box_height(box), 1)
            for box, _, _ in detections
        ]
    )

    line_threshold = average_height * 0.60

    detections.sort(
        key=lambda item: _box_top_left(item[0])
    )

    grouped_lines = []

    current_line = []

    current_y = None

    for detection in detections:

        box, text, confidence = detection

        y, x = _box_top_left(box)

        if current_y is None:

            current_y = y

            current_line.append(detection)

            continue

        if abs(y - current_y) <= line_threshold:

            current_line.append(detection)

        else:

            grouped_lines.append(current_line)

            current_line = [detection]

            current_y = y

    if current_line:

        grouped_lines.append(current_line)

    ordered = []

    for line in grouped_lines:

        line.sort(
            key=lambda item: _box_top_left(item[0])[1]
        )

        ordered.extend(line)

    return ordered


# -----------------------------------------------------------------------------
# OCRResult Construction
# -----------------------------------------------------------------------------

def _build_result(
    detections: List[Tuple],
) -> OCRResult:
    """
    Convert processed detections into OCRResult.
    """

    texts: List[str] = []

    confidences: List[float] = []

    boxes: List[List[List[float]]] = []

    for box, text, confidence in detections:

        texts.append(text)

        confidences.append(
            round(float(confidence), 3)
        )

        boxes.append(box)

    return OCRResult(

        text=texts,

        confidence=confidences,

        bounding_boxes=boxes,
    )


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def _average_confidence(
    confidences: List[float],
) -> float:
    """
    Compute average OCR confidence.
    """

    if not confidences:

        return 0.0

    return float(sum(confidences) / len(confidences))


def _log_summary(result: OCRResult) -> None:
    """
    Log OCR statistics.
    """

    logger.info(

        "OCR finished | %d detections | Avg confidence %.2f",

        len(result.text),

        _average_confidence(
            result.confidence
        ),
    )
# -----------------------------------------------------------------------------
# OCR Processing Pipeline
# -----------------------------------------------------------------------------

def _process_image(
    image_source: Union[str, Path, np.ndarray],
) -> OCRResult:
    """
    Complete OCR processing pipeline.

    Steps
    -----
    1. Load image
    2. Generate enhanced variants
    3. Run EasyOCR on each variant
    4. Merge all detections
    5. Remove duplicates
    6. Sort into reading order
    7. Build OCRResult
    """

    image = _load_image(image_source)

    variants = _generate_variants(image)

    reader = _get_reader()

    logger.info(
        "Running OCR on %d image variants.",
        len(variants),
    )

    detections = _collect_all_results(
        reader,
        variants,
    )

    logger.info(
        "Collected %d raw detections.",
        len(detections),
    )

    detections = _merge_duplicate_results(
        detections,
    )

    logger.info(
        "Remaining detections after deduplication: %d",
        len(detections),
    )

    detections = _sort_reading_order(
        detections,
    )

    result = _build_result(
        detections,
    )

    _log_summary(result)

    return result


# -----------------------------------------------------------------------------
# OCR Validation Helpers
# -----------------------------------------------------------------------------

def _is_supported_image(
    image_source: Union[str, Path, np.ndarray],
) -> bool:
    """
    Check if the supplied image source is supported.
    """

    if isinstance(image_source, np.ndarray):

        if image_source.size == 0:
            return False

        return True

    if isinstance(image_source, Path):
        suffix = image_source.suffix.lower()

    elif isinstance(image_source, str):
        suffix = Path(image_source).suffix.lower()

    else:
        return False

    return suffix in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }


def _empty_result() -> OCRResult:
    """
    Return an empty OCRResult.
    """

    return OCRResult(
        text=[],
        confidence=[],
        bounding_boxes=[],
    )


# -----------------------------------------------------------------------------
# Performance Helpers
# -----------------------------------------------------------------------------

def warmup_reader() -> None:
    """
    Optional warm-up function.

    Call once during FastAPI startup to eliminate
    the EasyOCR initialization delay.
    """

    try:
        _get_reader()
        logger.info("EasyOCR warmed up.")

    except Exception:

        logger.exception(
            "Unable to warm up EasyOCR."
        )
# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def extract_text(
    image_source: Union[str, Path, np.ndarray],
) -> OCRResult:
    """
    Extract text from an image using a robust multi-stage OCR pipeline.

    Parameters
    ----------
    image_source
        Path to an image or numpy array.

    Returns
    -------
    OCRResult

    Notes
    -----
    This function never raises an exception.
    On any failure an empty OCRResult is returned.
    """

    try:

        if not _is_supported_image(image_source):

            logger.warning(
                "Unsupported image type: %s",
                type(image_source),
            )

            return _empty_result()

        return _process_image(image_source)

    except FileNotFoundError:

        logger.exception(
            "Image file not found."
        )

    except ValueError as exc:

        logger.exception(
            "Invalid image: %s",
            exc,
        )

    except cv2.error:

        logger.exception(
            "OpenCV processing failed."
        )

    except Exception:

        logger.exception(
            "Unexpected OCR failure."
        )

    return _empty_result()


# -----------------------------------------------------------------------------
# Development Test
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    TEST_IMAGE = Path("sample.jpg")

    if TEST_IMAGE.exists():

        result = extract_text(TEST_IMAGE)

        print("\nDetected Text\n")

        for text, confidence in zip(
            result.text,
            result.confidence,
        ):

            print(
                f"{confidence:.2f} : {text}"
            )

    else:

        logger.info(
            "Place a sample image named sample.jpg beside this file "
            "to perform a standalone OCR test."
        )