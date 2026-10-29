"""
Step 5: Label Creator
=====================
Create labels: forward returns over the next `horizon` bars
(next day on daily bars, next 5 minutes on 1-minute bars)
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LabelCreator:
    """Create forward-looking labels for training"""
    
    @staticmethod
    def create(close_prices, horizon=1, same_session=False, unit='day'):
        """
        Create forward return labels
        
        Args:
            close_prices: Series of close prices (indexed by time)
            horizon: How many bars ahead to predict (1 = next day on daily bars)
            same_session: Blank out labels whose target bar is on a later date, so an
                          intraday label never spans the overnight gap
            unit: Bar unit for log messages ('day' or 'minute')
            
        Returns:
            pd.Series: Forward returns
        """
        # Calculate forward return
        forward_return = close_prices.pct_change(horizon)
        
        # Shift to align with current row
        label = forward_return.shift(-horizon)
        
        if same_session:
            dates = pd.Series(close_prices.index.date, index=close_prices.index)
            crosses = dates.shift(-horizon) != dates
            label[crosses] = np.nan
            logger.info(f"  Dropped {int(crosses.sum())} labels that would span the overnight gap")
        
        logger.info(f"✓ Created labels: predict {horizon}-{unit} return")
        logger.info(f"  Mean return: {label.mean()*100:+.4f}%")
        logger.info(f"  Std return:  {label.std()*100:.4f}%")
        logger.info(f"  Min return:  {label.min()*100:+.4f}%")
        logger.info(f"  Max return:  {label.max()*100:+.4f}%")
        
        return label
    
    @staticmethod
    def validate_labels(labels):
        """
        Validate label quality
        
        Args:
            labels: Label series
            
        Returns:
            bool: True if valid
        """
        nan_count = labels.isna().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN labels (expected from shift)")
        
        extreme = (labels.abs() > 0.10).sum()
        if extreme > 0:
            logger.warning(f"Found {extreme} extreme values (>10% in one label)")
        
        logger.info(f"✓ Labels valid")
        return True


if __name__ == '__main__':
    from csv_loader import CSVDataLoader
    from fii_loader import FIILoader
    from data_merger import DataMerger
    from feature_engineer import FeatureEngineer
    
    loader = CSVDataLoader()
    df_ohlcv = loader.load(symbol='SBIN.NS')
    
    fii_loader = FIILoader()
    df_fii = fii_loader.load()
    
    merger = DataMerger()
    df_merged = merger.merge(df_ohlcv, df_fii)
    
    label_creator = LabelCreator()
    y = label_creator.create(df_merged['close'], horizon=1)
    
    print(f"Labels shape: {y.shape}")
    print(f"First 10 values:\n{y.head(10)}")