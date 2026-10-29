"""
Step 8: Hyperparameter Tuner
=============================
Grid search and Bayesian optimization for hyperparameter tuning

Both searches only look at the training data, split in time order
(TimeSeriesSplit / a trailing validation slice). The test set is used once,
after the search, to report the chosen parameters.
"""

import pandas as pd
import numpy as np
import logging
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Tune XGBoost hyperparameters"""

    @staticmethod
    def grid_search(X_train, y_train, X_test, y_test, n_splits=3):
        """
        Grid search for best hyperparameters

        Args:
            X_train, y_train: Training data (searched with time-ordered CV)
            X_test, y_test: Test data (reported once, not used for selection)
            n_splits: Number of TimeSeriesSplit folds

        Returns:
            dict: Best parameters
        """
        logger.info("Starting grid search...")

        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0]
        }

        base_model = XGBRegressor(
            n_estimators=200,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )

        # Each fold trains on the past and validates on the following block
        grid = GridSearchCV(
            base_model,
            param_grid,
            cv=TimeSeriesSplit(n_splits=n_splits),
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_train)

        logger.info(f"\n✓ Grid search complete")
        logger.info(f"  Best params: {grid.best_params_}")
        logger.info(f"  Best CV score: {-grid.best_score_:.6f}")

        # Report best model on the untouched test set
        best_model = grid.best_estimator_
        y_pred_test = best_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        logger.info(f"  Test RMSE: {test_rmse:.6f}")

        return grid.best_params_

    @staticmethod
    def bayesian_search(X_train, y_train, X_test, y_test, n_trials=30, val_frac=0.2):
        """
        Bayesian optimization for hyperparameters

        Args:
            X_train, y_train: Training data; its last `val_frac` is the validation slice
            X_test, y_test: Test data (reported once, not used for selection)
            n_trials: Number of trials
            val_frac: Share of training data (from its end) used to score trials

        Returns:
            dict: Best parameters
        """
        try:
            import optuna
        except ImportError:
            logger.error("Optuna not installed. Run: pip install optuna")
            return None

        logger.info(f"Starting Bayesian optimization ({n_trials} trials)...")

        split = int(len(X_train) * (1 - val_frac))
        X_fit, X_val = X_train.iloc[:split], X_train.iloc[split:]
        y_fit, y_val = y_train.iloc[:split], y_train.iloc[split:]

        def objective(trial):
            """Objective function to minimize (validation MSE)"""
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            }

            model = XGBRegressor(**params, random_state=42, n_jobs=-1, tree_method='hist')
            model.fit(X_fit, y_fit, verbose=False)

            y_pred = model.predict(X_val)
            return mean_squared_error(y_val, y_pred)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"\n✓ Bayesian search complete")
        logger.info(f"  Best trial: {study.best_trial.number}")
        logger.info(f"  Best params: {study.best_params}")
        logger.info(f"  Best validation MSE: {study.best_value:.6f}")

        # Refit on all training data and report on the untouched test set
        best_model = XGBRegressor(**study.best_params, random_state=42, n_jobs=-1, tree_method='hist')
        best_model.fit(X_train, y_train, verbose=False)
        test_rmse = np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))
        logger.info(f"  Test RMSE: {test_rmse:.6f}")

        return study.best_params


if __name__ == '__main__':
    # Example usage (run after previous steps)
    pass
