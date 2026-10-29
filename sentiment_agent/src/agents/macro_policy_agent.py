"""Macro Policy Agent — classifies RBI/macro headlines."""

import re
from src.config.settings import MACRO_KEYWORDS

# Rule-based classifier until LLM is wired
POLICY_PATTERNS = {
    "rate_action": ["repo rate", "reverse repo", "interest rate", "mpc", "rate hike", "rate cut"],
    "liquidity": ["liquidity", "ltro", "tltro", "msf", "liquidity adjustment"],
    "npa_provisioning": ["npa", "provision", "asset quality", "slippage"],
    "pca_framework": ["pca", "prompt corrective action"],
    "capital_adequacy": ["crar", "capital adequacy", "basel", "tier 1", "tier 2"],
    "regulatory_norms": ["norms", "guidelines", "circular", "directive", "compliance"],
    "budget_fiscal": ["budget", "finance minister", "fiscal", "allocation", "subsidy"],
}

# Sector sensitivity mapping
SECTOR_IMPACT = {
    "rate_action": {"psu": -0.5, "nbfc": -0.3, "reason": "rate hikes hurt rate-sensitive lenders"},
    "liquidity": {"psu": -0.2, "nbfc": -0.4, "reason": "tight liquidity squeezes NBFC funding"},
    "npa_provisioning": {"psu": -0.4, "nbfc": -0.3, "reason": "higher provisioning dents earnings"},
    "pca_framework": {"psu": -0.6, "nbfc": 0.0, "reason": "PCA restricts PSU bank operations"},
    "capital_adequacy": {"psu": -0.3, "nbfc": -0.3, "reason": "capital norms affect all banks"},
    "regulatory_norms": {"psu": -0.2, "nbfc": -0.2, "reason": "new compliance burden"},
    "budget_fiscal": {"psu": 0.1, "nbfc": 0.0, "reason": "fiscal support can aid PSU banks"},
}


def run(headline: str) -> dict:
    """
    Classify a macro headline and return policy type + sector impact estimate.

    Returns:
        {
            "policy_type": str,
            "affected_universe": ["psu"|"nbfc"],
            "sector_magnitude": {"psu": float, "nbfc": float},
            "reason": str,
            "confidence": float,
            "llm_used": False
        }
    """
    text = headline.lower()

    # Rule-based classification
    matched_type = None
    max_matches = 0
    for policy_type, keywords in POLICY_PATTERNS.items():
        matches = sum(1 for kw in keywords if kw in text)
        if matches > max_matches:
            max_matches = matches
            matched_type = policy_type

    if matched_type is None:
        return {
            "policy_type": "unknown",
            "affected_universe": ["psu", "nbfc"],
            "sector_magnitude": {"psu": 0.0, "nbfc": 0.0},
            "reason": "No macro pattern matched",
            "confidence": 0.0,
            "llm_used": False,
        }

    sector_impact = SECTOR_IMPACT.get(matched_type, {"psu": 0.0, "nbfc": 0.0, "reason": ""})
    affected = ["psu"] if sector_impact.get("psu", 0) != 0 else []
    if sector_impact.get("nbfc", 0) != 0:
        affected.append("nbfc")

    confidence = min(max_matches * 0.3, 0.95)

    return {
        "policy_type": matched_type,
        "affected_universe": affected,
        "sector_magnitude": {
            "psu": sector_impact.get("psu", 0.0),
            "nbfc": sector_impact.get("nbfc", 0.0),
        },
        "reason": sector_impact.get("reason", ""),
        "confidence": confidence,
        "llm_used": False,
    }
