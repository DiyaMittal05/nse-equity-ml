"""
Step 4: Feature Engineer
=========================
Engineer technical + FII-based features for daily or 1-minute bars.
Every feature only uses data up to and including the current bar, and none
depends on the price level (returns, ratios to price).
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NSE session: 09:15 to 15:30 IST
SESSION_OPEN_MIN = 9 * 60 + 15
SESSION_CLOSE_MIN = 15 * 60 + 30

# Look-back windows (in bars) per bar size
WINDOWS = {
    'daily':  {'returns': [1, 5, 20], 'ma': [5, 20, 50], 'stats': [10, 20, 60], 'suffix': 'd'},
    'minute': {'returns': [1, 5, 15], 'ma': [5, 15, 30], 'stats': [10, 20, 60], 'suffix': 'm'},
}


class FeatureEngineer:
    """Build trading features from OHLCV and FII data"""

    def __init__(self, df, bars='daily'):
        """
        Initialize with merged data

        Args:
            df: Merged OHLCV+FII DataFrame
            bars: 'daily' or 'minute'
        """
        if bars not in WINDOWS:
            raise ValueError(f"bars must be one of {list(WINDOWS)}, got {bars!r}")
        self.df = df.copy()
        self.bars = bars
        self.windows = WINDOWS[bars]
        self.features = pd.DataFrame(index=df.index)

    def build(self):
        """
        Build all features

        Returns:
            pd.DataFrame: Feature matrix (N, n_features)
        """
        logger.info(f"Building features ({self.bars} bars)...")

        self._add_returns()
        self._add_moving_averages()
        self._add_rsi()
        self._add_atr()
        self._add_bollinger_bands()
        self._add_vwap()
        self._add_buy_pressure()
        self._add_time_features()
        self._add_volume_features()
        self._add_rolling_stats()
        self._add_macd()
        self._add_fii_features()

        # Drop NaN from warmup (and any inf from division by zero)
        self.features = self.features.replace([np.inf, -np.inf], np.nan).dropna()

        logger.info(f"✓ Built {len(self.features.columns)} features")
        logger.info(f"  Shape: {self.features.shape}")

        return self.features

    def _add_returns(self):
        """Add return features (1 day, 1 week, 1 month / 1, 5, 15 minutes)"""
        sfx = self.windows['suffix']
        for n in self.windows['returns']:
            self.features[f'ret_{n}{sfx}'] = self.df['close'].pct_change(n)

    def _add_moving_averages(self):
        """Add moving average distance features"""
        for window in self.windows['ma']:
            ma = self.df['close'].rolling(window).mean()
            self.features[f'dist_ma{window}'] = (self.df['close'] - ma) / ma

    def _add_rsi(self, window=14):
        """Add Relative Strength Index"""
        delta = self.df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        rs = avg_gain / avg_loss
        self.features['rsi'] = 100 - (100 / (1 + rs))

    def _add_atr(self, window=14):
        """Add Average True Range as a fraction of price (scale-free across price levels)"""
        hl = self.df['high'] - self.df['low']
        hpc = (self.df['high'] - self.df['close'].shift(1)).abs()
        lpc = (self.df['low'] - self.df['close'].shift(1)).abs()

        true_range = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        self.features['atr_pct'] = true_range.rolling(window).mean() / self.df['close']

    def _add_bollinger_bands(self, window=20):
        """Add Bollinger Band position"""
        mid = self.df['close'].rolling(window).mean()
        std = self.df['close'].rolling(window).std()
        self.features['bb_pos'] = (self.df['close'] - mid) / (2 * std)

    def _add_vwap(self, window=20):
        """Add VWAP deviation: session VWAP (resets daily) on minute bars, rolling 20-day on daily bars"""
        typical = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        pv = typical * self.df['volume']

        if self.bars == 'minute':
            session = pd.Series(self.df.index.date, index=self.df.index)
            vwap = pv.groupby(session).cumsum() / self.df['volume'].groupby(session).cumsum()
        else:
            # A session VWAP on daily bars would just be the close
            vwap = pv.rolling(window).sum() / self.df['volume'].rolling(window).sum()

        self.features['vwap_dev'] = (self.df['close'] - vwap) / vwap

    def _add_buy_pressure(self):
        """Add intrabar buy pressure"""
        self.features['buy_pressure'] = (
            (self.df['close'] - self.df['low']) /
            (self.df['high'] - self.df['low'] + 1e-9)
        )

    def _add_time_features(self):
        """Add time-of-day features (minute bars) or calendar features (daily bars)"""
        if self.bars == 'minute':
            minute_of_day = self.df.index.hour * 60 + self.df.index.minute
            self.features['minute_of_day'] = minute_of_day
            self.features['time_since_open'] = minute_of_day - SESSION_OPEN_MIN
            self.features['time_to_close'] = SESSION_CLOSE_MIN - minute_of_day
            self.features['is_opening_30min'] = (self.features['time_since_open'] < 30).astype(int)
            self.features['is_closing_30min'] = (self.features['time_to_close'] <= 30).astype(int)
        else:
            self.features['day_of_week'] = self.df.index.day_of_week
            # Last business day of the month (calendar month-end often falls on a weekend)
            self.features['is_month_end'] = (self.df.index + pd.offsets.BMonthEnd(0) == self.df.index).astype(int)

    def _add_volume_features(self):
        """Add volume-based features"""
        avg_vol = self.df['volume'].rolling(20).mean()
        self.features['vol_ratio'] = self.df['volume'] / (avg_vol + 1e-9)
        self.features['vol_trend'] = np.sign(self.df['volume'].diff())

    def _add_rolling_stats(self):
        """Add rolling mean and volatility of bar returns (raw price levels drift over time)"""
        sfx = self.windows['suffix']
        bar_ret = self.df['close'].pct_change()
        for window in self.windows['stats']:
            self.features[f'ret_mean_{window}{sfx}'] = bar_ret.rolling(window).mean()
            self.features[f'volatility_{window}{sfx}'] = bar_ret.rolling(window).std()

    def _add_macd(self, fast=12, slow=26, signal=9):
        """Add MACD indicator, normalised by price"""
        ema_fast = self.df['close'].ewm(span=fast).mean()
        ema_slow = self.df['close'].ewm(span=slow).mean()

        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()

        self.features['macd'] = macd / self.df['close']
        self.features['macd_signal'] = macd_signal / self.df['close']

    def _add_fii_features(self):
        """Add FII-based features (trend and 3-day mean are computed per FII day by DataMerger)"""
        if 'net_fii' not in self.df.columns:
            logger.warning("FII data not available, skipping FII features")
            return

        self.features['fii_net_b'] = self.df['net_fii'] / 1e9
        self.features['fii_trend'] = self.df['fii_trend']
        self.features['fii_intensity_3d'] = self.df['fii_3d'] / 1e9
        self.features['is_heavy_fii_selling'] = (self.df['net_fii'] < -1e9).astype(int)


if __name__ == '__main__':
    from csv_loader import CSVDataLoader
    from fii_loader import FIILoader
    from data_merger import DataMerger

    loader = CSVDataLoader()
    df_ohlcv = loader.load(symbol='SBIN.NS')

    fii_loader = FIILoader()
    df_fii = fii_loader.load()

    merger = DataMerger()
    df_merged = merger.merge(df_ohlcv, df_fii)

    if df_merged is not None:
        engineer = FeatureEngineer(df_merged)
        X = engineer.build()

        print(X.head())
        print(f"\nShape: {X.shape}")
