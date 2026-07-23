import easyocr

reader = easyocr.Reader(
    ['en'],
    gpu=True
)


def extract_text(image_path):

    results = reader.readtext(str(image_path))

    extracted = []
    confidence = []

    for item in results:

        bbox, text, conf = item

        extracted.append(text)

        confidence.append(round(float(conf), 3))

    return extracted, confidence