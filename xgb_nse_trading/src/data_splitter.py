"""
Step 6: Data Splitter
=====================
Time-safe train/test split and feature scaling
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSplitter:
    """Split and scale data for model training"""
    
    @staticmethod
    def split(X, y, test_size=0.3, shuffle=False):
        """
        Time-safe train/test split
        
        Args:
            X: Feature DataFrame
            y: Label Series
            test_size: Proportion for test (default: 0.3)
            shuffle: Whether to shuffle (default: False, time-series safe)
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, scaler)
        """
        # Align X and y
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        logger.info(f"Data shape: {X.shape}")
        logger.info(f"Label shape: {y.shape}")
        
        # Time-based split (NO shuffling for time series)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"✓ Train/test split (70/30):")
        logger.info(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        logger.info(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Scale features (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
        
        logger.info(f"✓ Features scaled (StandardScaler)")
        logger.info(f"  Train mean: {X_train_scaled.mean().mean():.6f}")
        logger.info(f"  Train std:  {X_train_scaled.std().mean():.6f}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == '__main__':
    from csv_loader import CSVDataLoader
    from fii_loader import FIILoader
    from data_merger import DataMerger
    from feature_engineer import FeatureEngineer
    from label_creator import LabelCreator
    
    loader = CSVDataLoader()
    df_ohlcv = loader.load(symbol='SBIN.NS')
    
    fii_loader = FIILoader()
    df_fii = fii_loader.load()
    
    merger = DataMerger()
    df_merged = merger.merge(df_ohlcv, df_fii)
    
    engineer = FeatureEngineer(df_merged)
    X = engineer.build()
    
    label_creator = LabelCreator()
    y = label_creator.create(df_merged['close'])
    
    splitter = DataSplitter()
    X_train, X_test, y_train, y_test, scaler = splitter.split(X, y)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")