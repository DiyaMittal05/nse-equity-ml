"""
Step 1 (daily): CSV Data Loader
================================
Load daily OHLCV for PSU banks and NBFCs from the R analysis outputs
(Data_analysis_psu/ and Data_analysis_nbfc/ next to this project)
"""

import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Checked in order: cleaned output of r_analysis/02_clean_align.R first, raw Yahoo download as fallback
SEARCH_PATHS = [
    REPO_ROOT / 'Data_analysis_psu' / 'cleaned' / '{symbol}_ohlcv.csv',
    REPO_ROOT / 'Data_analysis_nbfc' / 'cleaned' / '{symbol}_ohlcv.csv',
    REPO_ROOT / 'Data_analysis_psu' / 'data' / '{symbol}.csv',
    REPO_ROOT / 'Data_analysis_nbfc' / 'data' / '{symbol}.csv',
]


class CSVDataLoader:
    """Load daily OHLCV from the R pipeline's CSV files"""

    @staticmethod
    def find_csv(symbol):
        """
        Find the CSV for a symbol

        Args:
            symbol: Yahoo-style ticker (e.g., 'SBIN.NS', 'BAJFINANCE.NS')

        Returns:
            Path or None
        """
        for template in SEARCH_PATHS:
            path = Path(str(template).format(symbol=symbol))
            if path.exists():
                return path
        return None

    def load(self, symbol, start_date=None, end_date=None):
        """
        Load daily OHLCV data

        Args:
            symbol: Yahoo-style ticker (e.g., 'SBIN.NS')
            start_date: Optional 'YYYY-MM-DD' lower bound
            end_date: Optional 'YYYY-MM-DD' upper bound

        Returns:
            pd.DataFrame: [open, high, low, close, volume] with DatetimeIndex named 'time'
        """
        path = self.find_csv(symbol)
        if path is None:
            logger.error(f"No CSV for {symbol}. Looked in: {[str(p).format(symbol=symbol) for p in SEARCH_PATHS]}")
            return None

        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index.name = 'time'
            # Some raw Yahoo files prefix columns with the ticker ("AFSL.NS.Close")
            df.columns = [c.rsplit('.', 1)[-1].lower() for c in df.columns]
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = df.sort_index()

            # Yahoo leaves NaN rows and zero-volume rows on some holidays; they are not real sessions
            n_before = len(df)
            df = df.dropna()
            df = df[df['volume'] > 0]
            if len(df) < n_before:
                logger.info(f"  Dropped {n_before - len(df)} NaN / zero-volume rows")

            if start_date:
                df = df[df.index >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df.index <= pd.Timestamp(end_date)]

            logger.info(f"✓ Loaded {symbol} from {path.relative_to(REPO_ROOT)}")
            logger.info(f"✓ Final shape: {df.shape} ({df.index[0].date()} to {df.index[-1].date()})")
            logger.info(f"  Price: ₹{df['close'].min():.2f} - ₹{df['close'].max():.2f}")

            return df

        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None


if __name__ == '__main__':
    df = CSVDataLoader().load('SBIN.NS')

    if df is not None:
        print(df.tail())
        print(f"\nShape: {df.shape}")
