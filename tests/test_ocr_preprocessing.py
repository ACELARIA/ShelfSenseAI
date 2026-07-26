import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.image_utils import preprocess_image


def test_preprocess_image_upscales_small_inputs_for_ocr(tmp_path):
    image_path = tmp_path / "small_label.png"
    synthetic = np.zeros((40, 60, 3), dtype=np.uint8)
    synthetic[:] = 255
    cv2.putText(
        synthetic,
        "SHELF",
        (6, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(image_path), synthetic)

    processed = preprocess_image(image_path)

    assert processed.dtype == np.uint8
    assert processed.shape[0] >= 120
    assert processed.shape[1] >= 180


def test_preprocess_image_preserves_orientation_targets_for_rotation(tmp_path):
    image_path = tmp_path / "rotated_label.png"
    synthetic = np.zeros((80, 120, 3), dtype=np.uint8)
    synthetic[:] = 255
    cv2.putText(
        synthetic,
        "ROTATED",
        (12, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    rotated = cv2.rotate(synthetic, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(str(image_path), rotated)

    processed = preprocess_image(image_path)

    assert processed.dtype == np.uint8
    assert processed.shape[0] >= 120
    assert processed.shape[1] >= 120
