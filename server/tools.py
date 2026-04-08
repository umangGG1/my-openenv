"""
Tool implementations for the Real Estate CS Environment.
"""

import json
import os
import random
from typing import Any, Dict, List, Optional

_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "properties.json")
with open(_DB_PATH) as f:
    _ALL_PROPERTIES: List[Dict] = json.load(f)


def _randomize_properties(seed: int) -> List[Dict]:
    rng = random.Random(seed)
    props = []
    for p in _ALL_PROPERTIES:
        prop = dict(p)
        prop["price"] = int(prop["price"] * rng.uniform(0.97, 1.03))  # ±3% jitter (was ±5%)
        if prop["available"] and rng.random() < 0.10:                  # 10% unavailable
            prop["available"] = False
            prop["available_dates"] = []
        props.append(prop)
    return props


def get_property_db(episode_seed: int) -> List[Dict]:
    return _randomize_properties(episode_seed)


def search_properties(
    db: List[Dict],
    bedrooms: Optional[int] = None,
    max_price: Optional[float] = None,
    min_price: Optional[float] = None,
    location: Optional[str] = None,
    property_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Search properties matching criteria. Returns up to 6 results."""
    results = []
    for prop in db:
        if not prop["available"]:
            continue
        if bedrooms is not None and prop["bedrooms"] != bedrooms:
            continue
        # 5% price grace on both ends
        if max_price is not None and prop["price"] > max_price * 1.05:
            continue
        if min_price is not None and prop["price"] < min_price * 0.95:
            continue
        if location is not None and location.lower() not in prop["neighborhood"].lower():
            continue
        if property_type is not None and property_type.lower() not in ("any", "all") \
                and property_type.lower() not in prop["type"].lower():
            continue
        results.append({
            "id": prop["id"],
            "address": prop["address"],
            "neighborhood": prop["neighborhood"],
            "type": prop["type"],
            "bedrooms": prop["bedrooms"],
            "bathrooms": prop["bathrooms"],
            "price": prop["price"],
            "sqft": prop["sqft"],
            "school_rating": prop["school_rating"],
        })

    results.sort(key=lambda x: (-x["school_rating"], x["price"]))
    results = results[:6]

    if not results:
        return {
            "status": "no_results",
            "message": "No available properties match the given criteria. Try relaxing filters (e.g. remove location, widen price range).",
            "properties": [],
            "tip": "Try calling search_properties without a location filter to see all available properties.",
        }
    return {"status": "success", "count": len(results), "properties": results}


def get_property_details(db: List[Dict], property_id: str) -> Dict[str, Any]:
    """Return full details for a specific property ID."""
    for prop in db:
        if prop["id"] == property_id:
            return {"status": "success", "property": prop}
    return {
        "status": "error",
        "message": f"Property '{property_id}' not found. Use IDs returned by search_properties.",
    }


def schedule_viewing(
    db: List[Dict],
    property_id: str,
    viewing_date: Optional[str],
    viewing_time: Optional[str],
    client_name: Optional[str],
    client_phone: Optional[str],
) -> Dict[str, Any]:
    """Schedule a viewing for a property."""
    prop = next((p for p in db if p["id"] == property_id), None)
    if prop is None:
        return {"status": "error", "message": f"Property '{property_id}' not found."}
    if not prop["available"]:
        return {"status": "error", "message": f"Property {property_id} is not available."}
    if not client_name:
        return {"status": "error", "message": "client_name is required to schedule a viewing."}

    note = ""
    if viewing_date and prop["available_dates"] and viewing_date not in prop["available_dates"]:
        note = f" Closest available slot: {prop['available_dates'][0]}."

    return {
        "status": "success",
        "confirmation": {
            "property_id": property_id,
            "address": prop["address"],
            "neighborhood": prop["neighborhood"],
            "viewing_date": viewing_date or (prop["available_dates"][0] if prop["available_dates"] else "TBD"),
            "viewing_time": viewing_time or "10:00",
            "client_name": client_name,
            "client_phone": client_phone or "not provided",
            "confirmation_code": f"VIEW-{property_id}-{(viewing_date or 'TBD').replace('-', '')}",
        },
        "note": note,
    }


def check_mortgage_estimate(
    home_price: float,
    down_payment: float,
    annual_income: float,
    interest_rate: float = 7.0,
    loan_term_years: int = 30,
) -> Dict[str, Any]:
    """
    Estimate monthly mortgage payment and affordability.

    Args:
        home_price: Purchase price in USD
        down_payment: Down payment amount in USD
        annual_income: Gross annual income in USD
        interest_rate: Annual interest rate percentage (default 7.0)
        loan_term_years: Loan term in years (default 30)
    """
    if down_payment >= home_price:
        return {"status": "error", "message": "Down payment cannot exceed home price."}
    if annual_income <= 0:
        return {"status": "error", "message": "annual_income must be positive."}

    loan_amount = home_price - down_payment
    monthly_rate = (interest_rate / 100) / 12
    n_payments = loan_term_years * 12

    if monthly_rate == 0:
        monthly_payment = loan_amount / n_payments
    else:
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** n_payments) / \
                          ((1 + monthly_rate) ** n_payments - 1)

    monthly_income = annual_income / 12
    dti_ratio = monthly_payment / monthly_income  # debt-to-income

    if dti_ratio <= 0.28:
        affordability = "excellent"
        advice = "This payment is comfortably within standard lending guidelines."
    elif dti_ratio <= 0.36:
        affordability = "good"
        advice = "This payment is within acceptable lending guidelines."
    elif dti_ratio <= 0.43:
        affordability = "marginal"
        advice = "This approaches the upper limit lenders typically allow (43% DTI)."
    else:
        affordability = "challenging"
        advice = "Monthly payment may exceed what lenders typically approve. Consider a larger down payment."

    down_pct = (down_payment / home_price) * 100
    pmi_note = "" if down_pct >= 20 else " PMI (~$50-200/mo) likely required (down payment < 20%)."

    return {
        "status": "success",
        "estimate": {
            "home_price": home_price,
            "down_payment": down_payment,
            "down_payment_pct": round(down_pct, 1),
            "loan_amount": round(loan_amount, 0),
            "interest_rate_pct": interest_rate,
            "monthly_payment": round(monthly_payment, 0),
            "monthly_income": round(monthly_income, 0),
            "dti_ratio_pct": round(dti_ratio * 100, 1),
            "affordability": affordability,
            "advice": advice + pmi_note,
        },
    }
