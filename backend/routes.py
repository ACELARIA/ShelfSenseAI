import shutil
import time
import uuid
import traceback

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.services.ai_services import analyze_product
from backend.config import APP_NAME, UPLOAD_DIR, VERSION
from backend.image_utils import (
    get_image_metadata,
    preprocess_image,
    validate_extension,
)
from backend.services.ocr_services import extract_text
from backend.services.ocr_postprocessor import process_ocr
from backend.services.field_extractor import extract_product_info
from backend.services.recommendation_service import RecommendationService

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

    if not validate_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are supported."
        )

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    destination = UPLOAD_DIR / filename

    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        width, height = get_image_metadata(destination)
        processed_image = preprocess_image(destination)

        ocr_result = extract_text(processed_image)

        processed_ocr = process_ocr(ocr_result)

        product_info = extract_product_info(processed_ocr)

        ai_analysis = analyze_product(processed_ocr)


        recommendations = RecommendationService.recommend(
            product_info=product_info,
            ai_analysis=ai_analysis,
            user_profile=None,
        )

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
                "raw_text": ocr_result.text,
                "cleaned_text": processed_ocr.cleaned_text,
                "reconstructed_text": processed_ocr.reconstructed_text,
                "confidence": ocr_result.confidence,
                "bounding_boxes": ocr_result.bounding_boxes,
            },
            "structured_ocr": {
                "product_name": processed_ocr.structured_fields.product_name,
                "manufacturer": processed_ocr.structured_fields.manufacturer,
                "dosage": processed_ocr.structured_fields.dosage,
                "expiry_date": processed_ocr.structured_fields.expiry_date,
                "manufacturing_date": processed_ocr.structured_fields.manufacturing_date,
                "batch_number": processed_ocr.structured_fields.batch_number,
                "mrp": processed_ocr.structured_fields.mrp,
                "ingredients": processed_ocr.structured_fields.ingredients,
            },
            "analysis": ai_analysis,
            "insights": recommendations.model_dump(),
            "product_info": product_info.model_dump(),
        }
    except Exception as e:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {e}"
        )
#    except Exception as e:
#        raise HTTPException(
#            status_code=500,
#            detail=f"Processing failed: {str(e)}"
#        )

    finally:
        file.file.close()