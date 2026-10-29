"""Technical indicator fetcher for PSU and NBFC tickers."""

import pandas as pd
from pathlib import Path
from src.config.settings import DATA_PSU, DATA_NBFC


def _resolve_path(ticker: str, universe: str) -> Path | None:
    """Resolve the correct indicators CSV path."""
    if universe == "psu":
        base = DATA_PSU / "indicators"
        candidate = base / f"{ticker}_indicators.csv"
    elif universe == "nbfc":
        base = DATA_NBFC / "indicators"
        candidate = base / f"{ticker}_indicators.csv"
    else:
        for u in ["psu", "nbfc"]:
            p = _resolve_path(ticker, u)
            if p and p.exists():
                return p
        return None

    if candidate.exists():
        return candidate
    return None


def run(ticker: str, universe: str, indicator: str, date: str | None = None) -> float | pd.Series:
    """
    Load technical indicator data for a ticker.

    Args:
        ticker: e.g. "SBIN.NS"
        universe: "psu" or "nbfc"
        indicator: column name like "RSI", "MACD", "BB_Upper", etc.
        date: optional YYYY-MM-DD to get a specific value

    Returns:
        Scalar float if date given, else full time series as pd.Series
    """
    path = _resolve_path(ticker, universe)
    if path is None:
        raise FileNotFoundError(f"No indicator data for {ticker} in universe {universe}")

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Find date column
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime", "timestamp")), None)
    if date_col is None:
        date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])

    # Find indicator column (case-insensitive fuzzy match)
    indicator_col = next(
        (c for c in df.columns if c.upper() == indicator.upper()),
        None
    )
    if indicator_col is None:
        # Try partial match
        indicator_col = next(
            (c for c in df.columns if indicator.upper() in c.upper()),
            None
        )
    if indicator_col is None:
        raise ValueError(f"Indicator '{indicator}' not found in {path}. Columns: {list(df.columns)}")

    if date:
        row = df[df["date"] == pd.to_datetime(date)]
        if row.empty:
            raise ValueError(f"No data for {date} in {path}")
        return float(row[indicator_col].iloc[0])

    return df.set_index("date")[indicator_col]
