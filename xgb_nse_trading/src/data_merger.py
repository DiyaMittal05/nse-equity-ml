"""
Step 3: Data Merger
===================
Merge OHLCV bars with daily FII flows
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Oldest FII value a bar may use (covers long weekends / holiday runs)
MAX_FII_STALENESS = pd.Timedelta(days=5)


class DataMerger:
    """Merge OHLCV and FII data"""
    
    @staticmethod
    def merge(df_ohlcv, df_fii):
        """
        Merge OHLCV bars with daily FII data
        
        Args:
            df_ohlcv: OHLCV DataFrame with DatetimeIndex
            df_fii: FII DataFrame with 'date' column
            
        Returns:
            pd.DataFrame: Merged data with FII columns aligned to each bar's date
        """
        if df_ohlcv is None or df_fii is None:
            logger.error("Input data cannot be None")
            return None
        
        try:
            # Create copy to avoid modifying original
            df = df_ohlcv.copy()
            
            # Extract date from bar index (as datetime64 so it matches the FII 'date' dtype)
            df['date'] = pd.to_datetime(df.index.date).astype('datetime64[ns]')

            fii = df_fii[['date', 'net_fii', 'inflow', 'outflow']].copy()
            fii['date'] = pd.to_datetime(fii['date']).astype('datetime64[ns]')
            fii = fii.sort_values('date')
            
            # Day-level FII features, computed on the FII table itself so "3 days" means
            # 3 FII days whatever the bar size (each uses only that day and earlier)
            fii['fii_trend'] = np.sign(fii['net_fii'].diff())
            fii['fii_3d'] = fii['net_fii'].rolling(3).mean()

            # No look-ahead: day D's FII figures are published after the close, so each bar gets the
            # most recent FII day strictly BEFORE its own date (holidays fall back to the prior day).
            # Values older than MAX_FII_STALENESS are left NaN rather than carried forward forever.
            index_name = df.index.name or 'time'
            df = df.rename_axis(index_name).reset_index()
            df = pd.merge_asof(
                df.sort_values('date', kind='stable'),  # keep intraday bar order
                fii,
                on='date',
                direction='backward',
                allow_exact_matches=False,
                tolerance=MAX_FII_STALENESS,
            )
            df = df.set_index(index_name).sort_index()

            # Drop date column (keep index)
            df = df.drop('date', axis=1)
            
            logger.info(f"✓ Merged data shape: {df.shape}")
            logger.info(f"  Columns: {list(df.columns)}")
            
            return df
        
        except Exception as e:
            logger.error(f"Merge error: {e}")
            return None
    
    @staticmethod
    def validate_merge(df):
        """
        Validate merged data quality
        
        Args:
            df: Merged DataFrame
            
        Returns:
            bool: True if valid
        """
        # Check for NaN in critical columns (FII only if it was merged in)
        critical_cols = [c for c in ['open', 'close', 'volume', 'net_fii'] if c in df.columns]
        nan_count = df[critical_cols].isna().sum().sum()

        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in critical columns {critical_cols}")
            return False
        
        logger.info(f"✓ Validation passed")
        return True


if __name__ == '__main__':
    from csv_loader import CSVDataLoader
    from fii_loader import FIILoader
    
    loader = CSVDataLoader()
    df_ohlcv = loader.load(symbol='SBIN.NS')
    
    fii_loader = FIILoader()
    df_fii = fii_loader.load()
    
    if df_ohlcv is not None and df_fii is not None:
        merger = DataMerger()
        df_merged = merger.merge(df_ohlcv, df_fii)
        
        print(df_merged.head())
        print(f"\nShape: {df_merged.shape}")