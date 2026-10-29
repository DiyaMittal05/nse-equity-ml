"""
Step 7: Model Trainer
=====================
Train XGBoost model with early stopping and evaluation
"""

import pandas as pd
import numpy as np
import logging
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train XGBoost model"""
    
    def __init__(self, n_estimators=300, max_depth=5, learning_rate=0.05, horizon=1, bars_per_day=1):
        """
        Initialize model with hyperparameters
        
        Args:
            n_estimators: Number of trees
            max_depth: Max depth per tree
            learning_rate: Learning rate
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        # Quick test-set Sharpe uses one trade every `horizon` bars (non-overlapping)
        self.horizon = horizon
        self.periods_per_year = 252 * bars_per_day / horizon
        self.model = None
    
    def train(self, X_train, y_train, X_test, y_test, val_frac=0.15):
        """
        Train XGBoost model with early stopping on a validation slice
        
        The last `val_frac` of the (time-ordered) training data is held out for early
        stopping, so the test set is never used to choose the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (evaluation only)
            y_test: Test labels (evaluation only)
            val_frac: Share of the training data, taken from its end, used for early stopping
            
        Returns:
            XGBRegressor: Trained model
        """
        logger.info(f"Training XGBoost with {self.n_estimators} trees...")
        
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            n_jobs=-1,
            early_stopping_rounds=50  # constructor arg since xgboost 1.6; removed from fit() in 2.0
        )

        # Time-ordered train/validation split for early stopping
        split = int(len(X_train) * (1 - val_frac))
        X_fit, X_val = X_train.iloc[:split], X_train.iloc[split:]
        y_fit, y_val = y_train.iloc[:split], y_train.iloc[split:]
        
        # Train with early stopping
        self.model.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        logger.info(f"✓ Training complete (best iteration {self.model.best_iteration} "
                    f"on {len(X_val)}-row validation slice)")
        
        # Evaluate
        self._evaluate(X_train, y_train, X_test, y_test)
        
        return self.model
    
    def _evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
        """
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Trading Sharpe ratio (gross of costs; see the walk-forward backtest for net results)
        idx = np.arange(0, len(y_test), self.horizon)
        strategy_returns = y_test.values[idx] * np.sign(y_pred_test[idx])
        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(self.periods_per_year)
        win_rate = (strategy_returns > 0).mean()
        
        logger.info(f"\nMetrics:")
        logger.info(f"  Train RMSE: {train_rmse:.6f}")
        logger.info(f"  Test RMSE:  {test_rmse:.6f}")
        logger.info(f"  Test R²:    {test_r2:.4f}")
        logger.info(f"  Sharpe:     {sharpe:.2f} (gross)")
        logger.info(f"  Win Rate:   {win_rate:.1%}")
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            np.array: Predictions
        """
        if self.model is None:
            logger.error("Model not trained yet")
            return None
        
        return self.model.predict(X)
    
    def save(self, path):
        """
        Save trained model
        
        Args:
            path: File path to save to
        """
        if self.model is None:
            logger.error("No model to save")
            return False
        
        joblib.dump(self.model, path)
        logger.info(f"✓ Model saved to {path}")
        return True
    
    def load(self, path):
        """
        Load pre-trained model
        
        Args:
            path: File path to load from
        """
        self.model = joblib.load(path)
        logger.info(f"✓ Model loaded from {path}")
        return self.model


if __name__ == '__main__':
    # Example usage (run after previous steps)
    pass