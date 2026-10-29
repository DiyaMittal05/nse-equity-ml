"""
Step 10: Backtest Engine
=========================
Walk-forward backtesting with detailed metrics
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def strategy_returns(y_true, y_pred, cost=0.0):
    """
    Daily returns of trading sign(prediction), net of costs

    Args:
        y_true: Realised next-day returns
        y_pred: Predicted next-day returns
        cost: Cost per unit of position change (0.001 = 10 bps); a long->short flip pays it twice

    Returns:
        np.array: Net strategy returns
    """
    position = np.sign(np.asarray(y_pred))
    turnover = np.abs(np.diff(position, prepend=0.0))
    return np.asarray(y_true) * position - cost * turnover


class WalkForwardBacktest:
    """Expanding-window walk-forward backtest"""

    def __init__(self, gap_days=1, cost=0.001, explain=True, bars_per_day=1, horizon=1):
        """
        Initialize walk-forward backtest

        Args:
            gap_days: Trading days dropped between train and test so no training label
                      overlaps the test period (gap_days * bars_per_day must be >= horizon)
            cost: Transaction cost per unit of position change (default 10 bps)
            explain: Compute SHAP per window to see what drove each window's predictions
            bars_per_day: 1 for daily bars, 375 for 1-minute bars
            horizon: Label horizon in bars; trades are evaluated every `horizon` bars so
                     their holding periods do not overlap
        """
        self.gap_days = gap_days
        self.gap_rows = max(gap_days * bars_per_day, horizon)
        self.horizon = horizon
        # Non-overlapping trades per year, for annualising Sharpe
        self.periods_per_year = 252 * bars_per_day / horizon
        self.cost = cost
        self.explain = explain
        self.results = []
        self.predictions = []
        self.shap_by_window = {}

    def run(self, X, y, n_windows=5):
        """
        Run walk-forward backtest

        The data is cut into n_windows + 1 equal blocks. Window k trains on everything
        before block k+1 (minus the gap) and tests on block k+1, so all n_windows windows
        have training data and none trains on its own test period.

        Args:
            X: Feature DataFrame
            y: Label Series (aligned with X)
            n_windows: Number of test windows

        Returns:
            pd.DataFrame: Results per window
        """
        logger.info(f"Starting walk-forward backtest ({n_windows} windows, cost {self.cost*1e4:.0f} bps)...")

        n_samples = len(X)
        block_size = n_samples // (n_windows + 1)

        for window in range(n_windows):
            logger.info(f"\nWindow {window + 1}/{n_windows}")

            # Define splits
            test_start = (window + 1) * block_size
            test_end = test_start + block_size if window < n_windows - 1 else n_samples

            # No look-ahead: only train on data strictly before test, minus the gap
            train_end = test_start - self.gap_rows

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) < 100 or len(test_idx) < 50:
                logger.warning(f"  Insufficient data, skipping")
                continue

            # Get train/test
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # Scale (fit on train only)
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

            # Train model
            model = XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = model.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Trading metrics (net of costs)
            # One trade every `horizon` bars, so holding periods do not overlap
            trade_idx = np.arange(0, len(y_test), self.horizon)
            y_trade, pred_trade = y_test.values[trade_idx], y_pred[trade_idx]
            net_returns = strategy_returns(y_trade, pred_trade, self.cost)
            sharpe = (net_returns.mean() / net_returns.std()) * np.sqrt(self.periods_per_year) if net_returns.std() > 0 else 0
            win_rate = (net_returns > 0).mean()
            hit_rate = (np.sign(pred_trade) == np.sign(y_trade)).mean()

            # Max drawdown
            cumulative = np.cumprod(1 + net_returns)
            running_max = np.maximum.accumulate(cumulative)
            max_dd = ((cumulative - running_max) / running_max).min()

            result = {
                'window': window,
                'train_start': X_train.index[0].date(),
                'test_start': X_test.index[0].date(),
                'test_end': X_test.index[-1].date(),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'trades': len(trade_idx),
                'rmse': rmse,
                'sharpe': sharpe,
                'win_rate': win_rate,
                'hit_rate': hit_rate,
                'max_dd': max_dd,
                'avg_return': net_returns.mean(),
                'total_return': cumulative[-1] - 1,
            }
            self.results.append(result)
            self.predictions.append(pd.DataFrame({
                'window': window, 'y_true': y_trade, 'y_pred': pred_trade, 'net_return': net_returns,
            }, index=X_test.index[trade_idx]))

            if self.explain:
                self.shap_by_window[window] = self._window_shap(model, X_test_scaled, y_test, y_pred)

            logger.info(f"  Test {result['test_start']} to {result['test_end']} ({len(trade_idx)} trades) | RMSE: {rmse:.6f} | "
                        f"Sharpe: {sharpe:.2f} | Hit rate: {hit_rate:.1%} | Max DD: {max_dd:.1%}")

        results_df = pd.DataFrame(self.results)

        logger.info(f"\n{'='*60}")
        logger.info(f"WALK-FORWARD SUMMARY (net of {self.cost*1e4:.0f} bps costs)")
        logger.info(f"{'='*60}")
        logger.info(f"Average Sharpe:     {results_df['sharpe'].mean():.2f}")
        logger.info(f"Average Hit Rate:   {results_df['hit_rate'].mean():.1%}")
        logger.info(f"Average Win Rate:   {results_df['win_rate'].mean():.1%}")
        logger.info(f"Average Max DD:     {results_df['max_dd'].mean():.1%}")
        logger.info(f"Average Return/Trade: {results_df['avg_return'].mean()*100:+.4f}%")

        return results_df

    def _window_shap(self, model, X_test, y_test, y_pred):
        """Mean |SHAP| per feature for this window, overall and on wrong-direction days"""
        from src.feature_importance import FeatureImportance

        analyzer = FeatureImportance(model, X_test, y_test)
        diag = analyzer.diagnose_failures(y_pred)
        return diag

    def predictions_frame(self):
        """All out-of-sample predictions across windows"""
        return pd.concat(self.predictions) if self.predictions else pd.DataFrame()

    def shap_frame(self):
        """Per-window SHAP diagnosis, long format"""
        frames = [df.assign(window=w) for w, df in self.shap_by_window.items() if df is not None]
        return pd.concat(frames) if frames else pd.DataFrame()


if __name__ == '__main__':
    # Example usage (run after previous steps)
    pass
