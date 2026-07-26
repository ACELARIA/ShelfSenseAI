from pydantic import BaseModel, Field


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


class OCRResult(BaseModel):
    text: list[str] = Field(default_factory=list)
    confidence: list[float] = Field(default_factory=list)
    bounding_boxes: list[list[list[float]]] = Field(default_factory=list)


class ProductInfo(BaseModel):
    product_name: str = ""
    category: str = ""
    manufacturer: str = ""
    expiry_date: str = ""
    manufacturing_date: str = ""
    batch_number: str = ""
    ingredients: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: str = ""


class Identification(BaseModel):
    status: str = "Identified"
    confidence: str = "0%"


class Quality(BaseModel):
    expiry_status: str = "Unknown"
    manufacturing_date: str = ""
    expiry_date: str = ""
    batch_number: str = ""


class Inventory(BaseModel):
    status: str = "Available"
    priority: str = "Normal"
    restock_required: bool = False


class StorageGuide(BaseModel):
    recommended: str = "Refer to package instructions"
    assessment: str = "Compliant"


class AIRecommendation(BaseModel):
    decision: str = "Suitable"
    risk_level: str = "Low"
    summary: str = ""


class RecommendationResult(BaseModel):
    identification: Identification = Field(default_factory=Identification)
    quality: Quality = Field(default_factory=Quality)
    inventory: Inventory = Field(default_factory=Inventory)
    storage: StorageGuide = Field(default_factory=StorageGuide)
    ai_recommendation: AIRecommendation = Field(default_factory=AIRecommendation)