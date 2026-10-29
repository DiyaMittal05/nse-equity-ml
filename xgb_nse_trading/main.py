"""
Main Pipeline Execution
=======================
Run the complete XGBoost trading model pipeline end-to-end.

  --bars daily   daily bars from the R CSVs (2019 onwards), predicts the next day's close
  --bars minute  1-minute bars from Yahoo Finance (last ~30 days, saved history grows with
                 each run), predicts the return over the next 5 minutes

Usage: python main.py [--symbol SBIN.NS] [--bars daily|minute]
"""

import argparse
import logging
import sys
import os

# Ensure src/ is on path if running from elsewhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Below this share of trading days with an FII value, FII features are skipped
# (the scraped FII history is much shorter than the 2019+ price history)
MIN_FII_COVERAGE = 0.8

# Per bar size: label horizon (bars), bars per trading day, cost per unit position change
MODES = {
    'daily':  {'horizon': 1, 'bars_per_day': 1,   'cost': 0.001,  'unit': 'day'},
    'minute': {'horizon': 5, 'bars_per_day': 375, 'cost': 0.0003, 'unit': 'minute'},
}


def main(symbol='SBIN.NS', bars='daily'):
    """Execute complete pipeline"""
    mode = MODES[bars]
    results_dir = os.path.join('data', 'results', bars)
    
    logger.info("="*70)
    logger.info(f"XGBOOST NSE {bars.upper()} TRADING MODEL - COMPLETE PIPELINE ({symbol})")
    logger.info("="*70)
    
    try:
        # Import all modules from src package
        import pandas as pd
        from src.csv_loader import CSVDataLoader
        from src.yahoo_minute_loader import YahooMinuteLoader
        from src.fii_loader import FIILoader
        from src.data_merger import DataMerger
        from src.feature_engineer import FeatureEngineer
        from src.label_creator import LabelCreator
        from src.data_splitter import DataSplitter
        from src.model_trainer import ModelTrainer
        from src.hyperparameter_tuner import HyperparameterTuner
        from src.feature_importance import FeatureImportance
        from src.backtest_engine import WalkForwardBacktest
        from src.results_reporter import ResultsReporter
        from src import data_checks
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBRegressor
        
        # Step 1: Load OHLCV data
        logger.info("\n[STEP 1] Loading OHLCV Data")
        logger.info("-"*70)
        loader = YahooMinuteLoader() if bars == 'minute' else CSVDataLoader()
        df_ohlcv = loader.load(symbol=symbol)
        if df_ohlcv is None:
            logger.error("Failed to load OHLCV data")
            return False
        data_checks.check_ohlcv(df_ohlcv)
        
        # Step 2: Load FII data
        logger.info("\n[STEP 2] Loading FII Data")
        logger.info("-"*70)
        fii_loader = FIILoader()
        df_fii = fii_loader.load()
        if df_fii is None:
            logger.error("Failed to load FII data")
            return False
        
        # Step 3: Merge data
        logger.info("\n[STEP 3] Merging Data")
        logger.info("-"*70)
        merger = DataMerger()
        df_merged = merger.merge(df_ohlcv, df_fii)
        if df_merged is None:
            logger.error("Failed to merge data")
            return False
        
        # Share of bars that got a (previous-day) FII value
        fii_coverage = df_merged['net_fii'].notna().mean()
        logger.info(f"  FII coverage: {fii_coverage:.1%} of bars")
        if fii_coverage < MIN_FII_COVERAGE:
            logger.warning(
                f"FII data covers only {fii_coverage:.1%} of bars "
                f"(need {MIN_FII_COVERAGE:.0%}); skipping FII features. Add longer FII history to use them."
            )
            df_merged = df_merged.drop(columns=['net_fii', 'inflow', 'outflow', 'fii_trend', 'fii_3d'])
        else:
            # Days before the FII history starts are dropped by the feature warm-up
            merger.validate_merge(df_merged)
        
        # Step 4: Engineer features
        logger.info("\n[STEP 4] Engineering Features")
        logger.info("-"*70)
        engineer = FeatureEngineer(df_merged, bars=bars)
        X = engineer.build()
        if X is None or len(X) == 0:
            logger.error("Failed to engineer features")
            return False
        data_checks.check_features(X)
        
        # Step 5: Create labels
        logger.info("\n[STEP 5] Creating Labels")
        logger.info("-"*70)
        label_creator = LabelCreator()
        y = label_creator.create(df_merged['close'], horizon=mode['horizon'],
                                 same_session=(bars == 'minute'), unit=mode['unit'])
        if y is None or len(y) == 0:
            logger.error("Failed to create labels")
            return False
        label_creator.validate_labels(y)

        # The last bar has features but no label yet: it is the row we forecast from
        X_latest = X.iloc[[-1]]
        
        # Align features and labels on timestamp: X lost its indicator warm-up rows,
        # y lost its last `horizon` rows. Keep only rows present in both.
        common_idx = X.index.intersection(y.dropna().index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        logger.info(f"✓ Aligned X and y on {len(common_idx)} timestamps")
        data_checks.check_aligned(X, y)

        # Step 6: Split and scale data
        logger.info("\n[STEP 6] Splitting and Scaling Data")
        logger.info("-"*70)
        splitter = DataSplitter()
        X_train, X_test, y_train, y_test, scaler = splitter.split(X, y)
        if X_train is None:
            logger.error("Failed to split data")
            return False
        data_checks.check_split(X_train, X_test)
        
        # Step 7: Train model
        logger.info("\n[STEP 7] Training Model")
        logger.info("-"*70)
        trainer = ModelTrainer(horizon=mode['horizon'], bars_per_day=mode['bars_per_day'])
        model = trainer.train(X_train, y_train, X_test, y_test)
        if model is None:
            logger.error("Failed to train model")
            return False
        
        # Step 8: Hyperparameter tuning (optional)
        logger.info("\n[STEP 8] Hyperparameter Tuning (Optional)")
        logger.info("-"*70)
        logger.info("Skipping hyperparameter tuning (can be enabled)")
        
        # Step 9: Feature importance
        logger.info("\n[STEP 9] Analyzing Feature Importance")
        logger.info("-"*70)
        importance_analyzer = FeatureImportance(model, X_test, y_test)
        importance_analyzer.xgboost_gain()
        importance_analyzer.shap_values()
        importance_analyzer.diagnose_failures()
        importance_analyzer.summary()
        if importance_analyzer.importance_shap is not None:
            os.makedirs(results_dir, exist_ok=True)
            importance_analyzer.plot_importance_comparison(os.path.join(results_dir, 'gain_vs_shap.png'))
        
        # Step 10: Walk-forward backtest
        logger.info("\n[STEP 10] Walk-Forward Backtest")
        logger.info("-"*70)
        backtest = WalkForwardBacktest(cost=mode['cost'], bars_per_day=mode['bars_per_day'],
                                       horizon=mode['horizon'])
        backtest_results = backtest.run(X, y, n_windows=5)
        if backtest_results is None or len(backtest_results) == 0:
            logger.error("Failed to run backtest")
            return False
        
        # Step 11: Generate report
        logger.info("\n[STEP 11] Generating Results Report")
        logger.info("-"*70)
        reporter = ResultsReporter(output_dir=results_dir)
        reporter.generate_all(
            backtest_results,
            importance_analyzer.importance_gain,
            shap_importance=importance_analyzer.importance_shap,
            failure_diagnosis=importance_analyzer.failure_diagnosis,
            predictions=backtest.predictions_frame(),
            shap_by_window=backtest.shap_frame(),
        )
        
        # Final summary
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*70)
        logger.info(f"\nKey Results:")
        logger.info(f"  Average Sharpe:    {backtest_results['sharpe'].mean():.2f} (net of costs)")
        logger.info(f"  Average Hit Rate:  {backtest_results['hit_rate'].mean():.1%}")
        logger.info(f"  Average Win Rate:  {backtest_results['win_rate'].mean():.1%}")
        logger.info(f"  Average Max DD:    {backtest_results['max_dd'].mean():.1%}")
        
        # Forecast: refit on all labelled history (same settings, tree count chosen
        # by early stopping in step 7) so the forecast uses the most recent data
        last_bar = X_latest.index[0]
        if bars == 'minute' and (15 * 60 + 30) - (last_bar.hour * 60 + last_bar.minute) <= mode['horizon']:
            logger.info(f"\nLast bar ({last_bar}) is within {mode['horizon']} minutes of the close, "
                        f"so there is no {mode['horizon']}-minute forecast until the next session.")
            logger.info(f"\nResults saved to: {results_dir}/")
            return True
        
        final_scaler = StandardScaler().fit(X)
        final_params = {**model.get_params(), 'n_estimators': model.best_iteration + 1, 'early_stopping_rounds': None}
        final_model = XGBRegressor(**final_params)
        final_model.fit(pd.DataFrame(final_scaler.transform(X), columns=X.columns, index=X.index), y)
        
        last_close = df_merged.loc[last_bar, 'close']
        pred_return = float(final_model.predict(pd.DataFrame(final_scaler.transform(X_latest), columns=X_latest.columns))[0])
        when = last_bar.date() if bars == 'daily' else last_bar.strftime('%Y-%m-%d %H:%M')
        target = 'Next-day' if bars == 'daily' else f"Next {mode['horizon']}-minute"
        logger.info(f"\n{target} forecast for {symbol} (from {when} close ₹{last_close:.2f}):")
        logger.info(f"  Predicted return: {pred_return*100:+.3f}%")
        logger.info(f"  Predicted close:  ₹{last_close * (1 + pred_return):.2f}")
        logger.info(f"\nResults saved to: {results_dir}/")
        
        return True
    
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all step files are in src/ and requirements are installed")
        return False
    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="XGBoost pipeline on PSU/NBFC stocks")
    parser.add_argument('--symbol', default='SBIN.NS', help="Yahoo ticker, e.g. SBIN.NS or BAJFINANCE.NS")
    parser.add_argument('--bars', default='daily', choices=list(MODES),
                        help="daily: R CSVs since 2019; minute: Yahoo 1-minute bars (last ~30 days)")
    args = parser.parse_args()
    success = main(symbol=args.symbol, bars=args.bars)
    sys.exit(0 if success else 1)
