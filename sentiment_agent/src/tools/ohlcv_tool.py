"""OHLCV data fetcher for PSU and NBFC tickers."""

import pandas as pd
from pathlib import Path
from src.config.settings import DATA_PSU, DATA_NBFC


def _resolve_path(ticker: str, universe: str) -> Path | None:
    """Resolve the correct CSV path for a ticker and universe."""
    if universe == "psu":
        base = DATA_PSU / "cleaned"
        candidates = [
            base / f"{ticker}_ohlcv.csv",
            base / f"{ticker}.csv",
            DATA_PSU / "data" / f"{ticker}.csv",  # raw Yahoo download
        ]
    elif universe == "nbfc":
        base = DATA_NBFC / "cleaned"
        candidates = [
            base / f"{ticker}_ohlcv.csv",
            base / f"{ticker}.csv",
            DATA_NBFC / "data" / f"{ticker}.csv",  # raw Yahoo download
        ]
    else:
        # Try both
        for u in ["psu", "nbfc"]:
            p = _resolve_path(ticker, u)
            if p and p.exists():
                return p
        return None

    for c in candidates:
        if c.exists():
            return c
    return None


def run(ticker: str, universe: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """
    Load OHLCV data for a ticker.

    Args:
        ticker: e.g. "SBIN.NS"
        universe: "psu" or "nbfc"
        start_date: optional YYYY-MM-DD filter
        end_date: optional YYYY-MM-DD filter

    Returns:
        DataFrame with columns [date, open, high, low, close, volume]
    """
    path = _resolve_path(ticker, universe)
    if path is None:
        available = list((DATA_PSU / "cleaned").glob("*.csv")) + list((DATA_NBFC / "cleaned").glob("*.csv"))
        raise FileNotFoundError(f"No OHLCV data for {ticker} in universe {universe}. Checked: {available[:5]}...")

    df = pd.read_csv(path)

    # Normalize column names (some raw files prefix them with the ticker, e.g. "AFSL.NS.Close")
    df.columns = [c.strip().rsplit(".", 1)[-1].lower() if c.strip() else c for c in df.columns]

    # Ensure date column exists
    date_col = next((c for c in df.columns if c in ("date", "datetime", "timestamp")), None)
    if date_col is None:
        # Try first column
        date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])

    # Filter dates
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    return df.sort_values("date").reset_index(drop=True)
