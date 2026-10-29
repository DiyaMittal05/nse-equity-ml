"""
Step 1 (alternative): Angel One SmartAPI loader
================================================
Fetch minute-level OHLCV data from Angel One SmartAPI for any NSE stock.
Needs an Angel One API key (see .env.example); main.py uses yahoo_minute_loader instead.
"""

import pandas as pd
import numpy as np
import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Angel One's public instrument list (symbol -> token for every exchange)
SCRIP_MASTER_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
SCRIP_MASTER_PATH = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'ohlcv' / 'angel_scrip_master.json'

# Angel One allows at most 30 days of ONE_MINUTE candles per request
MAX_DAYS_PER_REQUEST = 30


class DataLoader:
    """Load minute-level OHLCV data from Angel One SmartAPI"""

    def __init__(self, api_key=None, client_code=None, password=None, totp_secret=None):
        """
        Initialize Angel One connection

        Args:
            api_key: Angel One API key (or from env ANGEL_ONE_API_KEY)
            client_code: Client code (or from env ANGEL_ONE_CLIENT_CODE)
            password: Login PIN (or from env ANGEL_ONE_PASSWORD)
            totp_secret: TOTP secret shown when enabling TOTP on Angel One
                         (or from env ANGEL_ONE_TOTP_SECRET); the 6-digit code is generated from it
        """
        # Pick up credentials from xgb_nse_trading/.env if present (see .env.example)
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parent.parent / '.env')
        except ImportError:
            pass
        
        self.api_key = api_key or os.getenv('ANGEL_ONE_API_KEY')
        self.client_code = client_code or os.getenv('ANGEL_ONE_CLIENT_CODE')
        self.password = password or os.getenv('ANGEL_ONE_PASSWORD')
        self.totp_secret = totp_secret or os.getenv('ANGEL_ONE_TOTP_SECRET')

        self.conn = None
        self.authenticated = False
        self._scrip_master = None

        try:
            from SmartApi import SmartConnect
            self.SmartConnect = SmartConnect
        except ImportError:
            logger.error("SmartAPI not installed. Run: pip install smartapi-python")
            raise

    def authenticate(self):
        """Authenticate with Angel One"""
        if not all([self.api_key, self.client_code, self.password, self.totp_secret]):
            logger.error("Missing credentials. Set env vars or pass to __init__")
            return False

        try:
            import pyotp
            totp = pyotp.TOTP(self.totp_secret).now()

            self.conn = self.SmartConnect(api_key=self.api_key)
            session_data = self.conn.generateSession(self.client_code, self.password, totp)

            if not session_data.get('status'):
                logger.error(f"Auth failed: {session_data.get('message')}")
                return False

            logger.info(f"✓ Authenticated as {self.client_code}")
            self.authenticated = True
            return True

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    def _load_scrip_master(self):
        """Load Angel One scrip master, downloading and caching it on first use"""
        if self._scrip_master is not None:
            return self._scrip_master

        if not SCRIP_MASTER_PATH.exists():
            import requests
            logger.info(f"Downloading Angel One scrip master to {SCRIP_MASTER_PATH}")
            SCRIP_MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(SCRIP_MASTER_URL, timeout=120)
            r.raise_for_status()
            SCRIP_MASTER_PATH.write_bytes(r.content)

        with open(SCRIP_MASTER_PATH, 'r') as f:
            self._scrip_master = json.load(f)
        return self._scrip_master

    def get_token_for_symbol(self, symbol):
        """
        Get Angel One instrument token for an NSE equity symbol

        Args:
            symbol: Stock symbol (e.g., 'SBIN')

        Returns:
            str: Token or None if not found
        """
        target = f"{symbol.upper()}-EQ"
        for row in self._load_scrip_master():
            if row.get('exch_seg') == 'NSE' and row.get('symbol') == target:
                logger.info(f"✓ Token found for {symbol}: {row['token']}")
                return row['token']

        logger.error(f"Symbol {symbol} ({target}) not found in Angel One scrip master")
        return None

    def load(self, symbol, days=30, start_date=None, end_date=None):
        """
        Fetch minute-level OHLCV data

        Args:
            symbol: NSE stock symbol (e.g., 'SBIN')
            days: Number of days to fetch (default: 30)
            start_date: Start date as 'YYYY-MM-DD' (overrides days)
            end_date: End date as 'YYYY-MM-DD' (default: today)

        Returns:
            pd.DataFrame: OHLCV data with DatetimeIndex
        """
        if not self.authenticated:
            if not self.authenticate():
                return None

        # Get token
        token = self.get_token_for_symbol(symbol)
        if not token:
            return None

        # Calculate date range
        if end_date is None:
            end = datetime.now()
        else:
            end = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=15, minute=30)

        if start_date:
            start = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start = end - timedelta(days=days)

        logger.info(f"Fetching {symbol} ({start.date()} to {end.date()})")

        try:
            candles = []
            chunk_start = start
            while chunk_start < end:
                chunk_end = min(chunk_start + timedelta(days=MAX_DAYS_PER_REQUEST), end)
                response = self.conn.getCandleData({
                    'exchange': 'NSE',
                    'symboltoken': token,
                    'interval': 'ONE_MINUTE',
                    'fromdate': chunk_start.strftime('%Y-%m-%d %H:%M'),
                    'todate': chunk_end.strftime('%Y-%m-%d %H:%M'),
                })

                if not response or not response.get('status'):
                    logger.error(f"API error: {response.get('message') if response else 'empty response'}")
                    return None

                candles.extend(response.get('data') or [])
                chunk_start = chunk_end
                time.sleep(0.4)  # stay under SmartAPI's historical-data rate limit

            if not candles:
                logger.error(f"No data for {symbol}")
                return None

            logger.info(f"✓ Received {len(candles)} bars")

            # Each candle is [timestamp, open, high, low, close, volume]
            df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])

            # Set datetime index
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()

            # Filter to NSE trading hours
            df = df.between_time('09:15', '15:30')

            logger.info(f"✓ Final shape: {df.shape}")
            logger.info(f"  Price: ₹{df['close'].min():.2f} - ₹{df['close'].max():.2f}")
            logger.info(f"  Volume: {df['volume'].mean():.0f} avg")

            return df

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None


if __name__ == '__main__':
    # Example usage
    loader = DataLoader()
    df = loader.load(symbol='SBIN', days=30)

    if df is not None:
        print(df.head())
        print(f"\nShape: {df.shape}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
