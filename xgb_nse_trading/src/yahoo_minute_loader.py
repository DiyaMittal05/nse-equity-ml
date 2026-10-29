"""
Step 1 (minute): Yahoo Finance 1-minute loader
===============================================
Fetch 1-minute OHLCV for an NSE stock from Yahoo Finance (no API key).

Yahoo only serves 1-minute bars for the last 30 days, at most 7 days per request.
Every download is merged into a saved CSV per symbol, so running this regularly
builds up a history longer than 30 days.
"""

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'ohlcv' / 'minute'
TIMEZONE = 'Asia/Kolkata'
YAHOO_MAX_DAYS = 30
DAYS_PER_REQUEST = 7


class YahooMinuteLoader:
    """Load 1-minute OHLCV from Yahoo Finance, keeping a growing local history"""

    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = Path(cache_dir)

    def cache_path(self, symbol):
        return self.cache_dir / f"{symbol}_1m.csv"

    def _read_cache(self, symbol):
        path = self.cache_path(symbol)
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col='time')
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(TIMEZONE)
        return df

    def _download(self, symbol):
        """Download the last ~30 days in 7-day chunks"""
        import yfinance as yf

        end = pd.Timestamp.now(tz=TIMEZONE).normalize() + pd.Timedelta(days=1)
        earliest = end - pd.Timedelta(days=YAHOO_MAX_DAYS - 1)
        parts = []
        chunk_end = end
        while chunk_end > earliest:
            chunk_start = max(chunk_end - pd.Timedelta(days=DAYS_PER_REQUEST), earliest)
            data = yf.download(symbol, start=chunk_start.date(), end=chunk_end.date(),
                               interval='1m', progress=False, auto_adjust=False)
            if data is not None and len(data):
                parts.append(data)
            chunk_end = chunk_start

        if not parts:
            return None
        df = pd.concat(parts)
        # yfinance returns (field, ticker) columns for a single ticker in recent versions
        df.columns = [(c[0] if isinstance(c, tuple) else c).lower() for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_convert(TIMEZONE)
        return df[['open', 'high', 'low', 'close', 'volume']]

    def load(self, symbol, refresh=True):
        """
        Load 1-minute OHLCV data

        Args:
            symbol: Yahoo ticker (e.g., 'SBIN.NS')
            refresh: Download the latest 30 days and merge them into the saved history

        Returns:
            pd.DataFrame: [open, high, low, close, volume] with tz-aware DatetimeIndex 'time'
        """
        cached = self._read_cache(symbol)

        fresh = None
        if refresh:
            try:
                fresh = self._download(symbol)
            except Exception as e:
                logger.error(f"Yahoo download failed for {symbol}: {e}")

        frames = [f for f in (cached, fresh) if f is not None and len(f)]
        if not frames:
            logger.error(f"No 1-minute data for {symbol}")
            return None

        # Newer download wins where the two overlap
        df = pd.concat(frames)
        df = df[~df.index.duplicated(keep='last')].sort_index()
        df.index.name = 'time'

        # Market hours only; drop incomplete bars
        df = df.between_time('09:15', '15:29')
        df = df.dropna(subset=['open', 'high', 'low', 'close'])

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.cache_path(symbol))

        days = pd.Index(df.index.date).nunique()
        logger.info(f"✓ Loaded {symbol}: {len(df)} 1-minute bars over {days} trading days "
                    f"({df.index[0]} to {df.index[-1]})")
        logger.info(f"  Saved history: {self.cache_path(symbol)}")

        return df


if __name__ == '__main__':
    df = YahooMinuteLoader().load('SBIN.NS')

    if df is not None:
        print(df.tail())
        print(f"\nShape: {df.shape}")
