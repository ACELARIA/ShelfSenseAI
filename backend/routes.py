import shutil
import time
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from backend.intelligence import evaluate_product

from backend.ai import analyze_product
from backend.config import APP_NAME, UPLOAD_DIR, VERSION
from backend.image_utils import (
    get_image_metadata,
    preprocess_image,
    validate_extension,
)
from backend.ocr import extract_text

router = APIRouter()


@router.get("/")
def home():
    return {"message": "ShelfSense AI Backend Running"}


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "application": APP_NAME,
        "version": VERSION,
    }


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    start = time.time()

    # Validate file extension
    if not validate_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are supported."
        )

    # Create a unique filename
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    destination = UPLOAD_DIR / filename

    try:
        # Save uploaded image
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Get image metadata
        width, height = get_image_metadata(destination)

        # Preprocess image
        processed = preprocess_image(destination)

        # OCR
        text, confidence = extract_text(processed)

        # AI Analysis
        analysis = analyze_product(text)
        insights = evaluate_product(analysis)

        total_time = round(time.time() - start, 2)

        return {
            "success": True,
            "processing_time_seconds": total_time,

            "file": {
                "original_name": file.filename,
                "saved_name": filename,
                "size_bytes": destination.stat().st_size,
                "content_type": file.content_type,
                "width": width,
                "height": height,
            },

            "ocr": {
                "text": text,
                "confidence": confidence,
            },

            "analysis": analysis,
            "insights": insights

        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )

    finally:
        file.file.close()