from functools import lru_cache
from pathlib import Path

import easyocr


@lru_cache(maxsize=1)
def _get_reader():
    try:
        import torch

        use_gpu = torch.cuda.is_available()
    except Exception:
        use_gpu = False

    return easyocr.Reader(
        ['en'],
        gpu=use_gpu,
        verbose=False,
    )


def extract_text(image_source):
    source = image_source
    if isinstance(image_source, (str, Path)):
        source = str(image_source)

    results = _get_reader().readtext(
        source,
        detail=1,
        paragraph=False,
        decoder='greedy',
        min_size=20,
        contrast_ths=0.3,
    )

    extracted = []
    confidence = []

    for _, text, conf in results:
        cleaned_text = text.strip()
        if not cleaned_text:
            continue

        extracted.append(cleaned_text)
        confidence.append(round(float(conf), 3))

    return extracted, confidence