"""
Step 9: Feature Importance
===========================
Analyze feature importance using SHAP values and XGBoost gain, and use SHAP
to diagnose the days where the model called the direction wrong
"""

import pandas as pd
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureImportance:
    """Analyze and visualize feature importance"""

    def __init__(self, model, X_test, y_test):
        """
        Initialize with trained model and test data

        Args:
            model: Trained XGBRegressor
            X_test: Test features (as fed to the model, i.e. scaled)
            y_test: Test labels
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.importance_gain = None
        self.importance_shap = None
        self.failure_diagnosis = None
        self._shap_values = None

    def xgboost_gain(self):
        """
        Get XGBoost feature importance (gain)

        Returns:
            pd.Series: Feature importance scores
        """
        importance = pd.Series(
            self.model.feature_importances_,
            index=self.X_test.columns
        ).sort_values(ascending=False)

        self.importance_gain = importance

        logger.info(f"✓ XGBoost Gain Importance:")
        logger.info(f"  Top 5:")
        for i, (feat, score) in enumerate(importance.head(5).items(), 1):
            logger.info(f"    {i}. {feat}: {score:.4f}")

        return importance

    def shap_values(self):
        """
        Calculate SHAP values for interpretability

        Returns:
            tuple: (shap_values, expected_value)
        """
        try:
            import shap
        except ImportError:
            logger.error("SHAP not installed. Run: pip install shap")
            return None, None

        logger.info("Calculating SHAP values...")

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_test)
        self._shap_values = shap_values

        # Mean absolute SHAP
        shap_importance = pd.Series(
            np.mean(np.abs(shap_values), axis=0),
            index=self.X_test.columns
        ).sort_values(ascending=False)

        self.importance_shap = shap_importance

        logger.info(f"✓ SHAP Importance:")
        logger.info(f"  Top 5:")
        for i, (feat, score) in enumerate(shap_importance.head(5).items(), 1):
            logger.info(f"    {i}. {feat}: {score:.6f}")

        return shap_values, explainer.expected_value

    def diagnose_failures(self, y_pred=None):
        """
        Compare SHAP attributions on days the model got the direction right vs wrong

        A feature whose mean |SHAP| is much larger on wrong days is one the model leaned
        on when it failed.

        Args:
            y_pred: Predictions for X_test (computed from the model if omitted)

        Returns:
            pd.DataFrame: per feature, mean |SHAP| on right / wrong days and their ratio
        """
        if self._shap_values is None:
            if self.shap_values()[0] is None:
                return None

        if y_pred is None:
            y_pred = self.model.predict(self.X_test)

        wrong = np.sign(y_pred) != np.sign(np.asarray(self.y_test))
        abs_shap = np.abs(self._shap_values)

        diag = pd.DataFrame({
            'mean_abs_shap_right': abs_shap[~wrong].mean(axis=0) if (~wrong).any() else np.nan,
            'mean_abs_shap_wrong': abs_shap[wrong].mean(axis=0) if wrong.any() else np.nan,
        }, index=self.X_test.columns)
        diag['wrong_to_right_ratio'] = diag['mean_abs_shap_wrong'] / diag['mean_abs_shap_right']
        diag = diag.sort_values('mean_abs_shap_wrong', ascending=False)
        diag.index.name = 'feature'

        self.failure_diagnosis = diag

        logger.info(f"✓ Failure diagnosis: direction wrong on {wrong.sum()}/{len(wrong)} test bars ({wrong.mean():.1%})")
        logger.info(f"  Features driving wrong calls (mean |SHAP| wrong vs right):")
        for feat, row in diag.head(5).iterrows():
            logger.info(f"    {feat:25s} {row['mean_abs_shap_wrong']:.6f} vs {row['mean_abs_shap_right']:.6f}")

        return diag

    def plot_importance_comparison(self, save_path='feature_importance.png'):
        """
        Plot XGBoost gain vs SHAP importance

        Args:
            save_path: Path to save plot
        """
        if self.importance_gain is None or self.importance_shap is None:
            logger.error("Run xgboost_gain() and shap_values() first")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # XGBoost Gain
        self.importance_gain.head(15).plot(kind='barh', ax=ax1, color='steelblue')
        ax1.set_xlabel('Gain Score')
        ax1.set_title('Top 15 Features - XGBoost Gain')
        ax1.invert_yaxis()

        # SHAP
        self.importance_shap.head(15).plot(kind='barh', ax=ax2, color='coral')
        ax2.set_xlabel('Mean |SHAP|')
        ax2.set_title('Top 15 Features - SHAP Importance')
        ax2.invert_yaxis()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved to {save_path}")
        plt.close()

    def summary(self):
        """
        Print summary of feature importance
        """
        if self.importance_gain is None:
            logger.error("Run xgboost_gain() first")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"FEATURE IMPORTANCE SUMMARY")
        logger.info(f"{'='*60}")

        logger.info(f"\nXGBoost Gain (Top 10):")
        for i, (feat, score) in enumerate(self.importance_gain.head(10).items(), 1):
            logger.info(f"  {i:2d}. {feat:25s} {score:.6f}")

        if self.importance_shap is not None:
            logger.info(f"\nSHAP Importance (Top 10):")
            for i, (feat, score) in enumerate(self.importance_shap.head(10).items(), 1):
                logger.info(f"  {i:2d}. {feat:25s} {score:.6f}")


if __name__ == '__main__':
    # Example usage (run after previous steps)
    pass
