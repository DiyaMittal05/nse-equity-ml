"""
Step 2: FII Loader
==================
Fetch daily Foreign Institutional Investor flows from SEBI/Moneycontrol
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Written by scraper_fii_dii.py
DEFAULT_FII_CSV = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'fii' / 'fii_dii_historical.csv'


class FIILoader:
    """Load daily FII flow data"""

    def __init__(self, csv_file=DEFAULT_FII_CSV):
        """
        Initialize FII loader
        
        Args:
            csv_file: Path to FII historical data CSV
        """
        self.csv_file = csv_file
    
    def load_from_csv(self, csv_file=None):
        """
        Load FII/DII data from CSV file
        
        Args:
            csv_file: Path to CSV with columns: date, fii_cm, dii_cm, ...
            
        Returns:
            pd.DataFrame: FII/DII data
        """
        file_path = csv_file or self.csv_file
        
        if not Path(file_path).exists():
            logger.error(f"FII file not found: {file_path}. Run src/scraper_fii_dii.py to create it")
            return None
        
        try:
            df = pd.read_csv(file_path, parse_dates=['date'])
            # Rename to match old interface
            df = df.rename(columns={
                'fii_cm': 'net_fii',
                'dii_cm': 'net_dii',
            })
            # net_fii in CSV is in crores; convert to rupees for model
            df['net_fii'] = df['net_fii'] * 1e7
            df['net_dii'] = df['net_dii'] * 1e7
            # Create inflow/outflow approximations from net
            df['inflow'] = df['net_fii'].clip(lower=0)
            df['outflow'] = (-df['net_fii']).clip(lower=0)
            logger.info(f"✓ Loaded {len(df)} days of FII/DII data")
            return df
        except Exception as e:
            logger.error(f"Error loading FII: {e}")
            return None
    def create_sample_data(self, days=30):
        """
        Create sample FII data for testing
        
        Args:
            days: Number of days to generate
            
        Returns:
            pd.DataFrame: Sample FII data
        """
        import numpy as np
        
        dates = pd.date_range(
            end=datetime.now(),
            periods=days,
            freq='B'  # Business days only
        )
        
        # Generate realistic FII flows
        np.random.seed(42)
        flows = np.random.normal(100, 200, size=len(dates)) * 1e7  # In rupees
        
        df = pd.DataFrame({
            'date': dates.date,
            'net_fii': flows.astype(int),
            'inflow': np.maximum(flows, 0).astype(int),
            'outflow': np.maximum(-flows, 0).astype(int)
        })
        
        logger.info(f"✓ Generated {len(df)} days of sample FII data")
        return df
    
    def load(self, days=None, csv_file=None):
        """
        Load FII data from CSV (never falls back to sample data)

        Args:
            days: Number of most recent days to keep (None = all)
            csv_file: Path to CSV (if None, use the default scraped CSV)

        Returns:
            pd.DataFrame: FII data with columns [date, net_fii, inflow, outflow],
                          or None if the CSV is missing or unreadable
        """
        df = self.load_from_csv(csv_file)

        if df is not None:
            if days:
                # Keep only requested days
                df = df.tail(days)
            logger.info(f"✓ FII data: {len(df)} days, date range {df['date'].min()} to {df['date'].max()}")
        
        return df


if __name__ == '__main__':
    loader = FIILoader()
    df = loader.load(days=30)
    
    if df is not None:
        print(df.head())
        print(f"\nShape: {df.shape}")
        print(f"Net FII avg: ₹{df['net_fii'].mean()/1e9:.2f}B")