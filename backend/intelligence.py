from datetime import datetime
from unicodedata import category


def evaluate_product(product):

    expiry = product.get("expiry_date", "")

    result = {
        "inventory_status": "Available",
        "risk_level": "Unknown",
        "recommendation": "",
        "priority": "Normal"
    }
    if category == "medicine":
        result["storage"] = "Store below 25°C"

    elif category == "food":
        result["storage"] = "Keep in a cool and dry place"

    else:
        result["storage"] = "Refer to package instructions"
    if expiry == "":
        result["risk_level"] = "Unknown"
        result["recommendation"] = "Expiry date not detected."
        return result

    try:
        exp_date = datetime.strptime(expiry, "%m/%Y")

        today = datetime.today()

        months_left = (
            (exp_date.year - today.year) * 12
            + exp_date.month
            - today.month
        )

        if months_left < 0:

            result["inventory_status"] = "Expired"
            result["risk_level"] = "Critical"
            result["priority"] = "High"
            result["recommendation"] = "Remove product immediately."

        elif months_left <= 3:

            result["risk_level"] = "High"
            result["priority"] = "High"
            result["recommendation"] = "Prioritize selling this product."

        elif months_left <= 6:

            result["risk_level"] = "Medium"
            result["priority"] = "Medium"
            result["recommendation"] = "Monitor expiry."

        else:

            result["risk_level"] = "Low"
            result["recommendation"] = "Safe for sale."

    except Exception:

        result["risk_level"] = "Unknown"
        result["recommendation"] = "Unable to evaluate expiry."

    return result