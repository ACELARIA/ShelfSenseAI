import shutil

from fastapi import APIRouter
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from backend.ocr import extract_text

from backend.image_utils import preprocess_image

from backend.config import (
    APP_NAME,
    VERSION,
    UPLOAD_DIR,
)

from backend.image_utils import (
    validate_extension,
    get_image_metadata,
)

router = APIRouter()


@router.get("/")
def home():
    return {
        "message": "ShelfSense AI Backend Running"
    }


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "application": APP_NAME,
        "version": VERSION
    }


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    if not validate_extension(file.filename):

        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported."
        )

    destination = UPLOAD_DIR / file.filename

    with destination.open("wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    width, height = get_image_metadata(destination)

    processed = preprocess_image(destination)

    text, confidence = extract_text(processed)

    return {

        "success": True,

        "filename": file.filename,

        "file_size": destination.stat().st_size,

        "width": width,

        "height": height,

        "content_type": file.content_type,

        "message": "OCR completed successfully.",

        "ocr": {

            "text": text,

            "confidence": confidence

        }

    }