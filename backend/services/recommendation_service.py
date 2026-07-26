"""Recommendation service for converting structured product and profile data into a typed result.

This module preserves the existing service-style API while moving the reasoning
into a single, explicit recommendation boundary.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from backend.schemas import ProductInfo, RecommendationResult


class RecommendationService:
    """Generate recommendation output from product and profile context."""

    @staticmethod
    def recommend(
        product_info: ProductInfo,
        ai_analysis: Mapping[str, Any] | None = None,
        user_profile: Mapping[str, Any] | None = None,
    ) -> RecommendationResult:
        """Build a recommendation result from structured product metadata.

        Args:
            product_info: Structured product information extracted from OCR.
            ai_analysis: Optional AI analysis payload returned by the model.
            user_profile: Optional user profile dictionary with constraints.

        Returns:
            Typed recommendation output with storage and risk guidance.
        """
        result = RecommendationResult()

        normalized_product = product_info.model_dump()
        if ai_analysis:
            normalized_product.update(
                {
                    key: value
                    for key, value in ai_analysis.items()
                    if value not in (None, "")
                }
            )

        category = str(normalized_product.get("category") or "").lower()
        expiry_date = str(normalized_product.get("expiry_date") or "").strip()
        manufacturing_date = str(normalized_product.get("manufacturing_date") or "").strip()
        batch_number = str(normalized_product.get("batch_number") or "").strip()
        profile = dict(user_profile or {})

        if category == "medicine":
            result.storage.recommended = "Store below 25°C"
        elif category == "food":
            result.storage.recommended = "Keep in a cool and dry place"
        else:
            result.storage.recommended = "Refer to package instructions"

        result.quality.manufacturing_date = manufacturing_date
        result.quality.expiry_date = expiry_date
        result.quality.batch_number = batch_number
        result.identification.confidence = f"{normalized_product.get('confidence', '') or '0%'}"
        if not expiry_date:
            result.quality.expiry_status = "Unknown"
            result.inventory.status = "Available"
            result.inventory.priority = "Normal"
            result.inventory.restock_required = False
            result.ai_recommendation.decision = "Review required"
            result.ai_recommendation.risk_level = "Unknown"
            result.ai_recommendation.summary = "Expiry date not detected."
            return result

        try:
            parsed_date = datetime.strptime(expiry_date, "%m/%Y")
            today = datetime.today()
            months_left = (
                (parsed_date.year - today.year) * 12
                + parsed_date.month
                - today.month
            )

            if months_left < 0:
                result.quality.expiry_status = "Expired"
                result.inventory.status = "Unavailable"
                result.inventory.priority = "High"
                result.inventory.restock_required = True
                result.ai_recommendation.decision = "Remove"
                result.ai_recommendation.risk_level = "Critical"
                result.ai_recommendation.summary = "Remove product immediately."
            elif months_left <= 3:
                result.quality.expiry_status = "Near expiry"
                result.inventory.status = "Available"
                result.inventory.priority = "High"
                result.inventory.restock_required = True
                result.ai_recommendation.decision = "Prioritize"
                result.ai_recommendation.risk_level = "High"
                result.ai_recommendation.summary = "Prioritize selling this product."
            elif months_left <= 6:
                result.quality.expiry_status = "Valid"
                result.inventory.status = "Available"
                result.inventory.priority = "Medium"
                result.inventory.restock_required = False
                result.ai_recommendation.decision = "Monitor"
                result.ai_recommendation.risk_level = "Medium"
                result.ai_recommendation.summary = "Monitor expiry."
            else:
                result.quality.expiry_status = "Valid"
                result.inventory.status = "Available"
                result.inventory.priority = "Normal"
                result.inventory.restock_required = False
                result.ai_recommendation.decision = "Suitable"
                result.ai_recommendation.risk_level = "Low"
                result.ai_recommendation.summary = "Product is within expiry and appears suitable based on the extracted information."

        except Exception:
            result.quality.expiry_status = "Unknown"
            result.inventory.status = "Available"
            result.inventory.priority = "Normal"
            result.inventory.restock_required = False
            result.ai_recommendation.decision = "Review required"
            result.ai_recommendation.risk_level = "Unknown"
            result.ai_recommendation.summary = "Unable to evaluate expiry."

        if profile:
            allergies = profile.get("allergies") or []
            if isinstance(allergies, (list, tuple)) and allergies:
                result.ai_recommendation.summary = (
                    f"{result.ai_recommendation.summary} User profile indicates allergies: "
                    f"{', '.join(map(str, allergies))}."
                )

            if profile.get("diabetes"):
                result.ai_recommendation.summary = (
                    f"{result.ai_recommendation.summary} User profile indicates diabetes; "
                    "review sugar content carefully."
                )

            if profile.get("hypertension"):
                result.ai_recommendation.summary = (
                    f"{result.ai_recommendation.summary} User profile indicates hypertension; "
                    "review sodium content carefully."
                )

        result.storage.assessment = "Compliant" if result.ai_recommendation.risk_level in {"Low", "Medium"} else "Needs review"
        return result

    @staticmethod
    def evaluate_product(
        product_info: ProductInfo,
        ai_analysis: Mapping[str, Any] | None = None,
        user_profile: Mapping[str, Any] | None = None,
    ) -> RecommendationResult:
        """Backward-compatible wrapper preserving the existing service call style."""
        return RecommendationService.recommend(
            product_info=product_info,
            ai_analysis=ai_analysis,
            user_profile=user_profile,
        )


def recommend_product(
    product_info: ProductInfo,
    ai_analysis: Mapping[str, Any] | None = None,
    user_profile: Mapping[str, Any] | None = None,
) -> RecommendationResult:
    """Convenience wrapper for creating a recommendation result."""
    return RecommendationService.recommend(
        product_info=product_info,
        ai_analysis=ai_analysis,
        user_profile=user_profile,
    )
