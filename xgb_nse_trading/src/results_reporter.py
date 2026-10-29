"""
Step 11: Results Reporter
==========================
Generate comprehensive results report with plots and metrics
"""

import pandas as pd
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultsReporter:
    """Generate results report and visualizations"""
    
    def __init__(self, output_dir='results'):
        """
        Initialize reporter
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_backtest_results(self, backtest_df):
        """
        Save backtest results to CSV
        
        Args:
            backtest_df: DataFrame with backtest results per window
        """
        output_path = self.output_dir / 'backtest_results.csv'
        backtest_df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved backtest results to {output_path}")
    
    def save_feature_importance(self, importance_df):
        """
        Save feature importance to CSV
        
        Args:
            importance_df: Series with feature importance scores
        """
        output_path = self.output_dir / 'feature_importance.csv'
        importance_df.to_csv(output_path, header=['importance'])
        logger.info(f"✓ Saved feature importance to {output_path}")
    
    def plot_walk_forward_sharpe(self, backtest_df):
        """
        Plot Sharpe ratio across windows
        
        Args:
            backtest_df: DataFrame with backtest results
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(backtest_df['window'] + 1, backtest_df['sharpe'], 
                marker='o', linewidth=2, markersize=8, color='steelblue')
        ax.axhline(y=backtest_df['sharpe'].mean(), color='red', 
                   linestyle='--', label=f"Avg: {backtest_df['sharpe'].mean():.2f}")
        ax.set_xlabel('Window')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Walk-Forward Sharpe Ratio')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        output_path = self.output_dir / 'walk_forward_sharpe.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved plot to {output_path}")
        plt.close()
    
    def plot_metrics_comparison(self, backtest_df):
        """
        Plot all metrics across windows
        
        Args:
            backtest_df: DataFrame with backtest results
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Sharpe
        axes[0, 0].bar(backtest_df['window'] + 1, backtest_df['sharpe'], color='steelblue')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].set_title('Sharpe Ratio by Window')
        axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Win Rate
        axes[0, 1].bar(backtest_df['window'] + 1, backtest_df['win_rate'] * 100, color='green')
        axes[0, 1].set_ylabel('Win Rate (%)')
        axes[0, 1].set_title('Win Rate by Window')
        axes[0, 1].set_ylim([0, 100])
        axes[0, 1].grid(True, alpha=0.3)
        
        # Max Drawdown
        axes[1, 0].bar(backtest_df['window'] + 1, backtest_df['max_dd'] * 100, color='red')
        axes[1, 0].set_ylabel('Max Drawdown (%)')
        axes[1, 0].set_title('Maximum Drawdown by Window')
        axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Average Return
        axes[1, 1].bar(backtest_df['window'] + 1, backtest_df['avg_return'] * 100, color='orange')
        axes[1, 1].set_ylabel('Avg Net Return per Trade (%)')
        axes[1, 1].set_title('Average Return by Window')
        axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'metrics_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved plot to {output_path}")
        plt.close()
    
    def generate_summary_report(self, backtest_df, feature_importance_df=None):
        """
        Generate text summary report
        
        Args:
            backtest_df: DataFrame with backtest results
            feature_importance_df: Optional feature importance series
        """
        report_path = self.output_dir / 'summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("XGBOOST NSE TRADING MODEL - RESULTS REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Backtest Summary
            f.write("WALK-FORWARD BACKTEST RESULTS (net of transaction costs)\n")
            f.write("-"*70 + "\n")
            f.write(f"Number of Windows:        {len(backtest_df)}\n")
            f.write(f"\nAverage Metrics:\n")
            f.write(f"  Sharpe Ratio:           {backtest_df['sharpe'].mean():+.2f}\n")
            f.write(f"  Hit Rate (direction):   {backtest_df['hit_rate'].mean():.1%}\n")
            f.write(f"  Win Rate:               {backtest_df['win_rate'].mean():.1%}\n")
            f.write(f"  Max Drawdown:           {backtest_df['max_dd'].mean():.1%}\n")
            f.write(f"  Avg Return (per trade): {backtest_df['avg_return'].mean()*100:+.4f}%\n")
            
            f.write(f"\nSharpe Ratio by Window:\n")
            for idx, row in backtest_df.iterrows():
                f.write(f"  Window {int(row['window'])+1} ({row['test_start']} to {row['test_end']}): {row['sharpe']:+.2f}\n")
            
            # Feature Importance
            if feature_importance_df is not None:
                f.write("\n" + "="*70 + "\n")
                f.write("TOP 15 FEATURES BY IMPORTANCE\n")
                f.write("-"*70 + "\n")
                for i, (feat, score) in enumerate(feature_importance_df.head(15).items(), 1):
                    f.write(f"  {i:2d}. {feat:25s} {score:.6f}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        logger.info(f"✓ Saved report to {report_path}")
    
    def save_diagnostics(self, shap_importance=None, failure_diagnosis=None,
                         predictions=None, shap_by_window=None):
        """
        Save SHAP importance, failure diagnosis and out-of-sample predictions to CSV
        
        Args:
            shap_importance: Series of mean |SHAP| per feature (hold-out model)
            failure_diagnosis: DataFrame from FeatureImportance.diagnose_failures()
            predictions: Out-of-sample predictions from the walk-forward backtest
            shap_by_window: Per-window failure diagnosis from the walk-forward backtest
        """
        outputs = {
            'shap_importance.csv': shap_importance,
            'failure_diagnosis.csv': failure_diagnosis,
            'walk_forward_predictions.csv': predictions,
            'walk_forward_shap.csv': shap_by_window,
        }
        for name, data in outputs.items():
            if data is not None and len(data) > 0:
                data.to_csv(self.output_dir / name)
                logger.info(f"✓ Saved {name}")
    
    def generate_all(self, backtest_df, feature_importance_df=None, **diagnostics):
        """
        Generate all reports and plots
        
        Args:
            backtest_df: DataFrame with backtest results
            feature_importance_df: Optional feature importance series
            **diagnostics: Optional inputs for save_diagnostics()
        """
        logger.info(f"\n{'='*70}")
        logger.info("GENERATING RESULTS REPORT")
        logger.info(f"{'='*70}\n")
        
        self.save_backtest_results(backtest_df)
        self.plot_walk_forward_sharpe(backtest_df)
        self.plot_metrics_comparison(backtest_df)
        
        if feature_importance_df is not None:
            self.save_feature_importance(feature_importance_df)
        
        self.save_diagnostics(**diagnostics)
        self.generate_summary_report(backtest_df, feature_importance_df)
        
        logger.info(f"\n✓ All results saved to {self.output_dir}")


if __name__ == '__main__':
    # Example usage (run after previous steps)
    pass