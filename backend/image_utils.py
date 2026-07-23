from pathlib import Path

import cv2

from PIL import Image

from backend.config import ALLOWED_EXTENSIONS


def validate_extension(filename):

    extension = Path(filename).suffix.lower()

    return extension in ALLOWED_EXTENSIONS


def get_image_metadata(image_path):

    image = Image.open(image_path)

    width, height = image.size

    return width, height


def preprocess_image(image_path):

    image = cv2.imread(str(image_path))

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    processed_path = image_path

    cv2.imwrite(
        str(processed_path),
        gray
    )

    return processed_path