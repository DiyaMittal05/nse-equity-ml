"""Rule-based content moderation and PII detection."""

import re
from src.config.settings import BLOCKLIST


# Simple financial domain keywords
FINANCIAL_KEYWORDS = [
    "stock", "share", "market", "bank", "finance", "rbi", "repo", "rate",
    "npa", "loan", "deposit", "capital", "equity", "bond", "yield",
    "sentiment", "impact", "ticker", "portfolio", "sector", "index",
    "nifty", "sensex", "bse", "nse", "psu", "nbfc",
]

# Simple PII patterns
PII_PATTERNS = [
    r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",  # PAN
    r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
    r"\b\d{2}/\d{2}/\d{4}\b",  # DOB
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Email
]


class ContentFilter:
    def __init__(self):
        self.blocklist = set(w.lower() for w in BLOCKLIST)
        self.financial_keywords = set(FINANCIAL_KEYWORDS)
        self.pii_patterns = [re.compile(p) for p in PII_PATTERNS]

    def check(self, query: str, headline: str = "") -> dict:
        """
        Run moderation checks.
        Returns: {"status": "PASS"|"REJECT"|"FLAG", "reasons": [...]}
        """
        text = f"{query} {headline}".lower()
        words = set(text.split())
        reasons = []

        # Blocklist check
        blocked = words & self.blocklist
        if blocked:
            reasons.append(f"off-topic keywords: {', '.join(blocked)}")

        # PII check
        for pattern in self.pii_patterns:
            if pattern.search(text):
                reasons.append("potential PII detected")
                break

        # Financial domain check
        if not (words & self.financial_keywords):
            # Also check if any ticker-like patterns exist
            if not re.search(r"\b[A-Z]{3,8}\b", text.upper()):
                reasons.append("no financial domain context detected")

        if reasons:
            # PII is always reject; off-topic is reject; weak domain is flag
            if "potential PII detected" in reasons:
                return {"status": "REJECT", "reasons": reasons}
            if "off-topic keywords" in str(reasons):
                return {"status": "REJECT", "reasons": reasons}
            return {"status": "FLAG", "reasons": reasons}

        return {"status": "PASS", "reasons": []}
