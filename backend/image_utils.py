from pathlib import Path

import cv2

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

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    return image