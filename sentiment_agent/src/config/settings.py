"""Central configuration for the agentic pipeline."""

import os
from pathlib import Path

# Project root...
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Sibling data directories (existing projects)
DATA_PSU = PROJECT_ROOT.parent / "Data_analysis_psu"
DATA_NBFC = PROJECT_ROOT.parent / "Data_analysis_nbfc"

# Local data
LOCAL_DATA = PROJECT_ROOT / "data"

# KB path (local, self-contained)
KB_PATH = LOCAL_DATA / "kb_psu_nbfc.json"

# Load .env if present
_ENV_FILE = PROJECT_ROOT / ".env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

# Headline sentiment model: "tfidf" (default) or "finbert" (ProsusAI/finbert, needs the [finbert] extra).
# FinBERT misreads Indian banking terms (e.g. "NPAs jump" scores positive), so it is opt-in.
SENTIMENT_BACKEND = os.getenv("SENTIMENT_BACKEND", "tfidf").lower()

# Ticker lists...
PSU_TICKERS = [
    "SBIN.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS",
    "UNIONBANK.NS", "BANKINDIA.NS", "INDIANB.NS", "CENTRALBK.NS",
    "IOB.NS", "MAHABANK.NS", "PSB.NS", "UCOBANK.NS",
]

# NBFC tickers — read dynamically from data dir, but here is the core list
NBFC_TICKERS = [
    "BAJFINANCE.NS", "CHOLAFIN.NS", "SHRIRAMFIN.NS", "M&MFIN.NS",
    "LTF.NS", "SBICARD.NS", "MUTHOOTFIN.NS", "IIFL.NS",
    "TATACAP.NS", "ABCAPITAL.NS", "MUTHOOTMF.NS",
    "FEDFINA.NS", "MANAPPURAM.NS", "CREDITACC.NS", "PAISALO.NS",
    "MONEYBOXX.NS", "UGROCAP.NS", "SBFC.NS", "INDOSTAR.NS",
    "MASFIN.NS", "SUNDARMFIN.NS", "EDELWEISS.NS", "BAIDFIN.NS",
    "JIOFIN.NS", "FUSION.NS", "FIVESTAR.NS", "POONAWALLA.NS",
    "AFSL.NS", "PFS.NS", "NIYOGIN.BO",
]

# Macro keywords that trigger cross-universe routing
MACRO_KEYWORDS = [
    "rbi", "repo rate", "reverse repo", "mpc", "monetary policy",
    "budget", "finance minister", "nirmala sitharaman",
    "npa", "provision", "pca", "prompt corrective action",
    "liquidity", "crar", "capital adequacy", "basel",
    "interest rate", "fd rates", "loan rates",
]

# Off-topic blocklist for moderation
BLOCKLIST = [
    "recipe", "cooking", "cricket", "ipl", "bollywood", "movie",
    "weather", "travel", "hotel", "restaurant", "food",
]

# LLM API keys — populated later
OPENAI_API_KEY = ""
ANTHROPIC_API_KEY = ""
GOOGLE_NEWS_API_KEY = ""

# Local LMStudio endpoint (fallback)
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
