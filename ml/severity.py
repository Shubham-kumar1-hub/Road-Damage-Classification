from __future__ import annotations


def estimate_severity(damage_type: str, confidence: float) -> str:
    if damage_type == "no_damage":
        return "none"
    if damage_type == "pothole" and confidence >= 0.70:
        return "high"
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.60:
        return "medium"
    return "low"

