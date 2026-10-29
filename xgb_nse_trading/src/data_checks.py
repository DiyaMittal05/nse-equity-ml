"""
Data-quality checks run between pipeline steps
===============================================
Each check raises DataQualityError on data that would silently corrupt the
model (bad prices, misaligned labels, train/test overlap) and logs a ✓ otherwise.
"""

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityError(ValueError):
    """Raised when a pipeline stage produces data that must not be used"""


def check_ohlcv(df, min_rows=250):
    """Step 1: prices are positive, consistent and in time order"""
    problems = []
    if len(df) < min_rows:
        problems.append(f"only {len(df)} rows (need {min_rows})")
    if not df.index.is_monotonic_increasing:
        problems.append("index not sorted by time")
    if df.index.has_duplicates:
        problems.append(f"{df.index.duplicated().sum()} duplicate timestamps")
    if df[['open', 'high', 'low', 'close', 'volume']].isna().any().any():
        problems.append("NaN in OHLCV")
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        problems.append("non-positive prices")
    bad_range = (df['high'] < df[['open', 'close', 'low']].max(axis=1)) | (df['low'] > df[['open', 'close']].min(axis=1))
    if bad_range.sum() > 0:
        problems.append(f"{bad_range.sum()} bars where high/low do not bracket open/close")

    _raise_or_pass("OHLCV", problems)

    # Large one-day moves are kept (they can be real) but surfaced
    big_moves = (df['close'].pct_change().abs() > 0.2).sum()
    if big_moves:
        logger.warning(f"  {big_moves} day(s) with >20% close-to-close move; check for unadjusted splits")


def check_features(X):
    """Step 4: finite values, no constant columns"""
    problems = []
    if len(X) == 0:
        problems.append("no rows left after warm-up")
    elif not np.isfinite(X.to_numpy(dtype=float)).all():
        problems.append("non-finite feature values")
    constant = [c for c in X.columns if X[c].nunique() <= 1]
    if constant:
        logger.warning(f"  Constant features (carry no information): {constant}")

    _raise_or_pass("features", problems)


def check_aligned(X, y):
    """Step 5: one label per feature row, same timestamps, no NaN labels"""
    problems = []
    if len(X) != len(y):
        problems.append(f"X has {len(X)} rows but y has {len(y)}")
    elif not X.index.equals(y.index):
        problems.append("X and y timestamps differ")
    if y.isna().any():
        problems.append(f"{y.isna().sum()} NaN labels")

    _raise_or_pass("X/y alignment", problems)


def check_split(X_train, X_test):
    """Step 6: every training row is strictly earlier than every test row"""
    problems = []
    if len(X_train) == 0 or len(X_test) == 0:
        problems.append("empty train or test set")
    elif X_train.index.max() >= X_test.index.min():
        problems.append(f"train ends {X_train.index.max()} but test starts {X_test.index.min()}")

    _raise_or_pass("train/test split", problems)


def _raise_or_pass(stage, problems):
    if problems:
        raise DataQualityError(f"{stage} check failed: " + "; ".join(problems))
    logger.info(f"✓ Data check passed: {stage}")
