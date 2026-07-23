from pydantic import BaseModel


class UploadResponse(BaseModel):
    success: bool
    filename: str
    file_size: int
    width: int
    height: int
    content_type: str
    message: str


class OCRResponse(BaseModel):
    success: bool
    filename: str
    extracted_text: list[str]
    confidence: list[float]