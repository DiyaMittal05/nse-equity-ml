"""Standalone Sentiment Impact Agent — Stages A→E, self-contained."""

import re
from src.registry import KBLoader
from src.config.settings import PSU_TICKERS, NBFC_TICKERS, MACRO_KEYWORDS, SENTIMENT_BACKEND


from src.agents.sentiment_model import TrainedSentiment, FinBERTSentiment


# ---------------------------------------------------------------------------
# Stage A: Base Sentiment
# ---------------------------------------------------------------------------

_SENTIMENT = None

def _get_sentiment():
    """TF-IDF model by default (FinBERT if SENTIMENT_BACKEND=finbert); keyword stub as last resort."""
    global _SENTIMENT
    if _SENTIMENT is None:
        if SENTIMENT_BACKEND == "finbert":
            try:
                _SENTIMENT = FinBERTSentiment()
                return _SENTIMENT
            except Exception as e:  # transformers/torch missing or model download failed
                print(f"[sentiment] FinBERT unavailable ({type(e).__name__}: {e}); using TF-IDF model")
        try:
            _SENTIMENT = TrainedSentiment()
        except FileNotFoundError:
            # Fallback to keyword stub if model not trained yet
            _SENTIMENT = BaseSentiment()
    return _SENTIMENT


class BaseSentiment:
    """Stub sentiment classifier. Replace with trained model."""

    # Simple keyword-based fallback for now
    POSITIVE_KWS = [
        "profit", "profits", "growth", "grew", "grown", "growing",
        "beat", "beats", "beating",
        "strong", "strength", "strengthen", "strengthened",
        "surge", "surges", "surged", "soar", "soars", "soared",
        "gain", "gains", "gained", "rally", "rallies", "rallied",
        "upgrade", "upgraded", "upgrades", "upgrading",
        "buy", "outperform", "bullish", "bull",
        "dividend", "dividends", "bonus", "bonuses",
        "merger", "merge", "merged", "merging",
        "acquisition", "acquire", "acquires", "acquired", "acquiring",
        "expansion", "expands", "expanded", "expand",
        "deal", "deals", "partnership", "partnerships",
        "launch", "launches", "launched", "launching",
        "record", "all-time", "historic", "historic high",
        "approval", "approved", "approves",
        "recovery", "recovers", "recovered",
    ]
    NEGATIVE_KWS = [
        "loss", "losses", "lost", "losing", "lose",
        "decline", "declines", "declined", "declining",
        "weak", "weakness", "weaken", "weakened", "weakening",
        "crash", "crashes", "crashed", "crashing",
        "fall", "falls", "fell", "fallen", "falling",
        "drop", "drops", "dropped", "dropping",
        "plunge", "plunges", "plunged", "plunging",
        "downgrade", "downgraded", "downgrades", "downgrading",
        "sell", "underperform", "bearish", "bear",
        "default", "defaults", "defaulted",
        "npa", "npas", "provision", "provisions", "provisioning",
        "write-off", "write off", "writeoff",
        "penalty", "penalties", "fined", "fine",
        "fraud", "scam", "scams", "scandal",
        "layoff", "layoffs", "restructuring", "restructured",
        "slump", "slumps", "slumped", "crisis",
        "crash", "crashes", "crashed",
        "concern", "concerns", "worried", "worry", "fear", "fears",
        "down", "slide", "slides", "sliding",
    ]

    def predict(self, headline: str) -> tuple[int, float]:
        """Return (direction, confidence): direction = +1/-1/0, confidence = 0..1"""
        text = headline.lower()
        pos = sum(1 for w in self.POSITIVE_KWS if w in text)
        neg = sum(1 for w in self.NEGATIVE_KWS if w in text)
        total = pos + neg
        if total == 0:
            return 0, 0.0
        net = pos - neg
        conf = min(1.0, total * 0.15)
        if net > 0:
            return +1, conf
        if net < 0:
            return -1, conf
        return 0, conf


# ---------------------------------------------------------------------------
# Stage B: Ticker Resolution
# ---------------------------------------------------------------------------

_RESOLVER = None

def _get_resolver() -> KBLoader:
    global _RESOLVER
    if _RESOLVER is None:
        _RESOLVER = KBLoader()
    return _RESOLVER


def resolve_tickers(headline: str, universe: str | None = None) -> list[dict]:
    """Find companies mentioned in headline, optionally filter by universe."""
    kb = _get_resolver()
    comps = kb.resolve(headline)
    if universe:
        comps = [c for c in comps if c.get("universe") == universe]
    return comps


# ---------------------------------------------------------------------------
# Stage C: Relevance Filtering
# ---------------------------------------------------------------------------

# Generic words that cause false positives when they match company aliases
GENERIC_WORDS = {
    "bank", "finance", "capital", "india", "investment", "commercial",
    "union", "central", "state", "national", "overseas", "rural",
    "credit", "micro", "gold", "sme", "rbi", "repo", "rate",
}

MARKET_MOVING_KWS = {
    "profit": 1.0, "profits": 1.0, "loss": 1.0, "losses": 1.0,
    "merger": 1.0, "merge": 0.9, "acquisition": 1.0, "acquire": 0.9, "acquires": 0.9, "acquired": 0.9,
    "default": 1.0, "defaults": 1.0, "npa": 0.9, "npas": 0.9,
    "provision": 0.8, "provisions": 0.8, "provisioning": 0.8,
    "dividend": 0.7, "dividends": 0.7, "bonus": 0.6,
    "rbi": 0.8, "repo rate": 0.9, "mpc": 0.8, "budget": 0.7,
    "upgrade": 0.8, "upgraded": 0.8, "downgrade": 0.8, "downgraded": 0.8,
    "buy": 0.6, "sell": 0.6, "outperform": 0.7, "underperform": 0.7,
    "strong": 0.5, "strength": 0.5, "weak": 0.5, "weakness": 0.5,
    "surge": 0.6, "surges": 0.6, "surged": 0.6,
    "crash": 0.9, "crashes": 0.9, "crashed": 0.9,
    "rally": 0.6, "rallies": 0.6, "rallied": 0.6,
    "drop": 0.6, "drops": 0.6, "dropped": 0.6,
    "plunge": 0.8, "plunges": 0.8, "plunged": 0.8,
    "rise": 0.5, "rises": 0.5, "rose": 0.5, "risen": 0.5,
    "fall": 0.5, "falls": 0.5, "fell": 0.5, "fallen": 0.5,
    "growth": 0.6, "grows": 0.6, "grew": 0.6, "grown": 0.6,
    "decline": 0.6, "declines": 0.6, "declined": 0.6,
    "expansion": 0.6, "expands": 0.6, "expanded": 0.6,
    "launch": 0.4, "launches": 0.4, "launched": 0.4,
    "deal": 0.5, "deals": 0.5,
    "partnership": 0.4, "partnerships": 0.4,
    "record": 0.4, "all-time": 0.4, "historic": 0.4, "massive": 0.4,
    "slight": -0.2, "marginally": -0.2, "modest": -0.2, "small": -0.2,
}


def is_generic_company_reference(headline: str) -> bool:
    """Check if headline uses company name in generic sense (e.g., 'river bank')."""
    text = headline.lower()
    # If headline contains only generic financial nouns with no market-moving keywords,
    # it is likely a generic reference
    has_market = any(kw in text for kw in MARKET_MOVING_KWS)
    return not has_market


def compute_relevance(headline: str, company: dict, resolver_hits: int = 1) -> dict:
    """Compute relevance score for a company in a headline."""
    text = headline.lower()

    # Market-moving keyword score
    headline_score = 0.0
    matched_kws = []
    for kw, weight in MARKET_MOVING_KWS.items():
        if kw in text:
            headline_score += weight
            matched_kws.append(kw)

    norm_headline = min(1.0, headline_score / 2.0)
    norm_resolver = min(1.0, resolver_hits / 1.2)

    final = 0.6 * norm_headline + 0.4 * norm_resolver

    # Penalty for generic references
    if is_generic_company_reference(headline):
        final *= 0.3

    # Penalty if alias is a generic word by itself
    name = company.get("name", "").lower()
    if name.split()[0] in GENERIC_WORDS and len(name.split()) == 1:
        final *= 0.5

    return {
        "ticker": company["ticker"],
        "relevance": round(min(1.0, final), 4),
        "headline_score": round(norm_headline, 4),
        "resolver_score": round(norm_resolver, 4),
        "matched_keywords": matched_kws,
        "is_relevant": final >= 0.35,
    }


# ---------------------------------------------------------------------------
# Stage D: Macro Sector Routing
# ---------------------------------------------------------------------------


def detect_macro(headline: str) -> tuple[bool, list[str]]:
    """Detect if headline is macro/sector-wide and which sectors affected."""
    text = headline.lower()
    is_macro = any(kw in text for kw in MACRO_KEYWORDS)
    sectors = []
    if is_macro:
        # Simple heuristic: all financial sectors affected by macro
        sectors = ["Public Sector Bank", "Consumer Finance", "Gold Loan",
                   "Diversified Finance", "Microfinance", "Commercial Vehicle Finance",
                   "Rural Finance", "Infrastructure Finance", "Credit Cards", "SME Finance"]
    return is_macro, sectors


# ---------------------------------------------------------------------------
# Stage E: Impact Scoring
# ---------------------------------------------------------------------------


def get_keyword_magnitude(headline: str) -> float:
    """Amplify impact based on strength keywords."""
    text = headline.lower()
    mag = 1.0
    if any(w in text for w in ["record", "all-time", "historic", "massive"]):
        mag += 0.3
    if any(w in text for w in ["slight", "marginal", "modest", "small"]):
        mag -= 0.2
    if re.search(r"\b\d+\s*bps\b", text):
        # Rate change magnitude
        match = re.search(r"\b(\d+)\s*bps\b", text)
        if match:
            bps = int(match.group(1))
            mag += min(0.3, bps / 100)
    return max(0.5, min(1.5, mag))


def compute_impact(headline: str, company: dict, sentiment_dir: int,
                   sentiment_conf: float, relevance: float,
                   macro_context: dict | None = None) -> dict:
    """Compute final impact score for one company."""
    kb = _get_resolver()
    sub_sector = company.get("sub_sector", "Diversified Finance")
    beta = kb.sector_beta(sub_sector)
    keyword_mag = get_keyword_magnitude(headline)

    raw = sentiment_dir * sentiment_conf * relevance * beta * keyword_mag
    impact_score = min(1.0, abs(raw))

    # Apply macro adjustment if present
    macro_adj = 0.0
    if macro_context and macro_context.get("policy_type") != "unknown":
        universe = company.get("universe", "nbfc")
        macro_adj = macro_context.get("sector_magnitude", {}).get(universe, 0.0)
        raw += macro_adj

    direction_label = "NEUTRAL"
    if raw > 0.05:
        direction_label = "BULL"
    elif raw < -0.05:
        direction_label = "BEAR"

    return {
        "ticker": company["ticker"],
        "name": company.get("name"),
        "sector": company.get("sector"),
        "sub_sector": sub_sector,
        "direction": direction_label,
        "impact": round(raw, 3),
        "impact_score": round(impact_score, 3),
        "sentiment": round(sentiment_dir * sentiment_conf, 3),
        "relevance": round(relevance, 3),
        "sector_beta": beta,
        "keyword_magnitude": keyword_mag,
        "macro_adjustment": macro_adj,
    }


# ---------------------------------------------------------------------------
# Agent Entry Point
# ---------------------------------------------------------------------------


def run(headline: str, universe: str, tickers: list[str] | None = None,
        macro_context: dict | None = None) -> list[dict]:
    """
    Full A→E pipeline for a headline on a given universe.

    Args:
        headline: news headline text
        universe: "psu" or "nbfc"
        tickers: optional explicit ticker list
        macro_context: optional macro policy context

    Returns:
        List of impact dicts per ticker
    """
    allowed = set(PSU_TICKERS if universe == "psu" else NBFC_TICKERS)

    # Stage A: Base sentiment
    sentiment = _get_sentiment()
    direction, conf = sentiment.predict(headline)

    # Stage D: Macro check
    is_macro, affected_sectors = detect_macro(headline)
    is_macro = is_macro or (macro_context is not None and macro_context.get("policy_type") != "unknown")

    # Stage B: Resolve tickers
    if tickers:
        companies = []
        kb = _get_resolver()
        for t in tickers:
            comp = kb.get(t)
            if comp:
                companies.append(comp)
    else:
        companies = resolve_tickers(headline, universe=universe)

    # If macro and no specific tickers resolved, expand to full universe
    if is_macro and not companies:
        kb = _get_resolver()
        companies = [kb.get(t) for t in allowed]
        companies = [c for c in companies if c]

    results = []
    for comp in companies:
        if comp["ticker"] not in allowed:
            continue

        # Stage C: Relevance
        rel = compute_relevance(headline, comp)
        if not rel["is_relevant"] and not is_macro:
            continue

        # Stage E: Impact
        imp = compute_impact(
            headline, comp, direction, conf,
            rel["relevance"], macro_context
        )
        imp["confidence"] = _tag_confidence(rel["relevance"])
        results.append(imp)

    # Sort by absolute impact
    results.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return results


def _tag_confidence(relevance: float) -> str:
    if relevance > 0.7:
        return "high"
    if relevance > 0.3:
        return "medium"
    return "low"
