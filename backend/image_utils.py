from pathlib import Path

import cv2
import numpy as np

from PIL import Image

from backend.config import ALLOWED_EXTENSIONS


def validate_extension(filename):

    extension = Path(filename).suffix.lower()

    return extension in ALLOWED_EXTENSIONS


def get_image_metadata(image_path):

    with Image.open(image_path) as image:
        width, height = image.size

    return width, height


def preprocess_image(image_path):

    if isinstance(image_path, np.ndarray):
        image = image_path
    else:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    height, width = gray.shape[:2]
    long_side = max(height, width)
    scale = max(1.0, 1800 / long_side)

    if scale > 1.0:
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
    elif long_side < 1200:
        gray = cv2.resize(
            gray,
            (max(width * 2, 1), max(height * 2, 1)),
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10)

    glare_mask = cv2.inRange(gray, 220, 255)
    glare_ratio = float(np.count_nonzero(glare_mask)) / float(glare_mask.size)
    if glare_ratio > 0.01:
        gray = cv2.addWeighted(gray, 0.85, cv2.GaussianBlur(gray, (7, 7), 0), 0.15, 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)

    return gray.astype(np.uint8)