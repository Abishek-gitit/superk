"""
recommendation.py
------------------
Rule-based Recovery Recommendation Engine.

Given a predicted damage class, this module returns:
- Safety status (Safe / Caution / Unsafe)
- A list of recommended recovery / engineering actions

This is intentionally simple (rule-based / lookup table) so it can later
be swapped for an LLM-generated engineering report (see README ->
Future Enhancements) without changing the calling code's interface.
"""

from typing import Dict, List, TypedDict


class DamageInfo(TypedDict):
    safety_status: str
    color: str          # used by the UI to color-code the status
    recommendations: List[str]


# ---------------------------------------------------------------------------
# Core lookup table: damage class -> safety status + recommended actions
# ---------------------------------------------------------------------------
RECOMMENDATION_TABLE: Dict[str, DamageInfo] = {
    "No Damage": {
        "safety_status": "Safe",
        "color": "green",
        "recommendations": [
            "Building is safe for occupancy.",
            "Continue periodic (routine) structural inspection.",
            "No immediate action required.",
        ],
    },
    "Minor Damage": {
        "safety_status": "Safe with Monitoring",
        "color": "yellow",
        "recommendations": [
            "Repair surface cracks and cosmetic damage.",
            "Inspect walls, roof, and cladding for hidden issues.",
            "Safe for continued use, but monitor for progression.",
            "Schedule a follow-up inspection within 30 days.",
        ],
    },
    "Major Damage": {
        "safety_status": "Restricted / Unsafe for Normal Use",
        "color": "orange",
        "recommendations": [
            "Structural engineering inspection required before re-entry.",
            "Restrict occupancy until cleared by a licensed engineer.",
            "Repair or shore up damaged structural members (columns, beams).",
            "Barricade the area to prevent unauthorized access.",
        ],
    },
    "Destroyed": {
        "safety_status": "Unsafe - Do Not Enter",
        "color": "red",
        "recommendations": [
            "Building is unsafe. Evacuate the area immediately.",
            "Do not attempt re-entry under any circumstances.",
            "Recommend demolition or complete reconstruction.",
            "Notify local disaster management / emergency authorities.",
        ],
    },
}


def get_recommendation(damage_class: str) -> DamageInfo:
    """
    Return the safety status and recommended actions for a given
    predicted damage class.

    Args:
        damage_class: One of "No Damage", "Minor Damage",
                       "Major Damage", "Destroyed".

    Returns:
        A DamageInfo dict with keys: safety_status, color, recommendations.
    """
    # Fallback in case the model returns an unexpected label
    default: DamageInfo = {
        "safety_status": "Unknown",
        "color": "gray",
        "recommendations": ["Damage class not recognized. Manual inspection recommended."],
    }
    return RECOMMENDATION_TABLE.get(damage_class, default)
