"""Checks for look-ahead, alignment and evaluation in the daily pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data_merger import DataMerger
from src.feature_engineer import FeatureEngineer
from src.label_creator import LabelCreator
from src.backtest_engine import WalkForwardBacktest, strategy_returns
from src import data_checks


def make_ohlcv(n=600, seed=0, start='2020-01-01'):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n, name='time')
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    return pd.DataFrame({
        'open': open_,
        'high': np.maximum(open_, close) * 1.005,
        'low': np.minimum(open_, close) * 0.995,
        'close': close,
        'volume': rng.integers(1_000, 10_000, n).astype(float),
    }, index=idx)


def make_fii(dates, values):
    return pd.DataFrame({
        'date': pd.to_datetime(dates),
        'net_fii': values,
        'inflow': np.clip(values, 0, None),
        'outflow': np.clip(-np.asarray(values), 0, None),
    })


def test_fii_uses_previous_trading_day():
    ohlcv = make_ohlcv(5, start='2026-04-20')  # Mon..Fri
    fii = make_fii(ohlcv.index, [1.0, 2.0, 3.0, 4.0, 5.0])
    merged = DataMerger.merge(ohlcv, fii)

    assert isinstance(merged.index, pd.DatetimeIndex)
    assert np.isnan(merged['net_fii'].iloc[0])           # nothing published before day 1
    assert merged['net_fii'].iloc[1:].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_fii_not_carried_forward_when_stale():
    ohlcv = make_ohlcv(15, start='2026-04-20')
    fii = make_fii(['2026-04-17'], [9.0])                 # one value, then nothing
    merged = DataMerger.merge(ohlcv, fii)

    assert merged['net_fii'].iloc[0] == 9.0               # Monday sees Friday's figure
    assert merged['net_fii'].iloc[-1:].isna().all()       # weeks later it is not reused


def test_labels_are_next_day_return():
    close = pd.Series([100.0, 110.0, 99.0], index=pd.bdate_range('2026-01-01', periods=3))
    y = LabelCreator.create(close)
    assert y.iloc[0] == pytest.approx(0.10)
    assert y.iloc[1] == pytest.approx(-0.10)
    assert np.isnan(y.iloc[2])


def test_features_do_not_depend_on_price_level():
    ohlcv = make_ohlcv(300)
    scaled = ohlcv.copy()
    scaled[['open', 'high', 'low', 'close']] *= 10

    X = FeatureEngineer(ohlcv).build()
    X_scaled = FeatureEngineer(scaled).build()
    pd.testing.assert_frame_equal(X, X_scaled, rtol=1e-6)


def test_features_only_use_the_past():
    ohlcv = make_ohlcv(300)
    X_full = FeatureEngineer(ohlcv).build()
    X_cut = FeatureEngineer(ohlcv.iloc[:250]).build()
    pd.testing.assert_frame_equal(X_full.loc[X_cut.index], X_cut)


def test_walk_forward_trains_only_on_the_past():
    ohlcv = make_ohlcv(900)  # ~140-row blocks, enough for every window's minimum train size
    X = FeatureEngineer(ohlcv).build()
    y = LabelCreator.create(ohlcv['close']).loc[X.index].dropna()
    X = X.loc[y.index]

    bt = WalkForwardBacktest(gap_days=1, explain=False)
    results = bt.run(X, y, n_windows=5)

    assert len(results) == 5                              # no window skipped
    for _, row in results.iterrows():
        train_end = X.index[row['train_size'] - 1]
        assert train_end < pd.Timestamp(row['test_start'])
        # one row gap: the last training label (t -> t+1) ends before the test period
        assert X.index.get_loc(pd.Timestamp(row['test_start'])) - row['train_size'] == 1

    preds = bt.predictions_frame()
    assert not preds.index.has_duplicates                 # test periods do not overlap


def test_strategy_returns_charge_costs_on_position_changes():
    y_true = np.array([0.01, 0.01, -0.01, 0.01])
    y_pred = np.array([1.0, 1.0, -1.0, -1.0])
    net = strategy_returns(y_true, y_pred, cost=0.001)
    # entry (1 unit), hold, flip (2 units), hold
    np.testing.assert_allclose(net, [0.01 - 0.001, 0.01, 0.01 - 0.002, -0.01])


def test_data_checks_catch_misaligned_labels():
    X = pd.DataFrame({'a': [1.0, 2.0]}, index=pd.bdate_range('2026-01-01', periods=2))
    y = pd.Series([0.1, 0.2], index=pd.bdate_range('2026-01-02', periods=2))
    with pytest.raises(data_checks.DataQualityError):
        data_checks.check_aligned(X, y)


def test_data_checks_catch_train_test_overlap():
    idx = pd.bdate_range('2026-01-01', periods=4)
    X = pd.DataFrame({'a': range(4)}, index=idx)
    with pytest.raises(data_checks.DataQualityError):
        data_checks.check_split(X.iloc[:3], X.iloc[2:])


# ---------------------------------------------------------------------------
# Minute bars
# ---------------------------------------------------------------------------

def make_minute_ohlcv(days=6, seed=1, start='2026-10-05'):
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex([
        d + pd.Timedelta(hours=9, minutes=15 + m)
        for d in pd.bdate_range(start, periods=days) for m in range(375)
    ], name='time').tz_localize('Asia/Kolkata')
    close = 800 * np.exp(np.cumsum(rng.normal(0, 0.0005, len(idx))))
    open_ = close * (1 + rng.normal(0, 0.0002, len(idx)))
    return pd.DataFrame({
        'open': open_,
        'high': np.maximum(open_, close) * 1.0003,
        'low': np.minimum(open_, close) * 0.9997,
        'close': close,
        'volume': rng.integers(1_000, 50_000, len(idx)).astype(float),
    }, index=idx)


def test_minute_labels_never_span_overnight():
    df = make_minute_ohlcv(days=3)
    y = LabelCreator.create(df['close'], horizon=5, same_session=True)
    last5 = df.groupby(df.index.date).tail(5).index
    assert y.loc[last5].isna().all()
    assert y.drop(last5).notna().all()


def test_minute_time_features_use_1530_close():
    X = FeatureEngineer(make_minute_ohlcv(days=2), bars='minute').build()
    last = X.iloc[-1]                                     # 15:29 bar
    assert last['time_to_close'] == 1
    assert last['is_closing_30min'] == 1
    assert X.loc[X.index.time == pd.Timestamp('14:59').time(), 'is_closing_30min'].eq(0).all()


def test_minute_vwap_resets_each_session():
    df = make_minute_ohlcv(days=2)
    eng = FeatureEngineer(df, bars='minute')
    eng._add_vwap()
    first_bar = df.groupby(df.index.date).head(1)
    typical = (first_bar['high'] + first_bar['low'] + first_bar['close']) / 3
    expected = (first_bar['close'] - typical) / typical
    np.testing.assert_allclose(eng.features.loc[first_bar.index, 'vwap_dev'], expected)


def test_minute_fii_uses_previous_day_and_day_level_trend():
    df = make_minute_ohlcv(days=4, start='2026-10-05')     # Mon..Thu
    fii = make_fii(['2026-10-02', '2026-10-05', '2026-10-06', '2026-10-07'], [1.0, -2.0, 3.0, 4.0])
    merged = DataMerger.merge(df, fii)
    
    by_day = merged.groupby(merged.index.date)
    assert by_day['net_fii'].nunique().eq(1).all()        # one FII value per session
    assert by_day['net_fii'].first().tolist() == [1.0, -2.0, 3.0, 4.0]
    # trend compares FII days, not minutes: Tue sees Mon (-2) vs Fri (1) -> down
    assert by_day['fii_trend'].first().tolist()[1] == -1.0


def test_minute_backtest_trades_do_not_overlap():
    df = make_minute_ohlcv(days=12)
    X = FeatureEngineer(df, bars='minute').build()
    y = LabelCreator.create(df['close'], horizon=5, same_session=True).loc[X.index].dropna()
    X = X.loc[y.index]
    
    bt = WalkForwardBacktest(explain=False, cost=0.0003, bars_per_day=375, horizon=5)
    results = bt.run(X, y, n_windows=5)
    assert len(results) == 5
    assert bt.gap_rows == 375
    preds = bt.predictions_frame()
    for _, w in preds.groupby('window'):
        positions = X.index.get_indexer(w.index)
        assert (np.diff(positions) == 5).all()            # one trade every 5 bars


def test_minute_loader_keeps_history_beyond_yahoo_window(tmp_path, monkeypatch):
    from src.yahoo_minute_loader import YahooMinuteLoader
    old = make_minute_ohlcv(days=2, start='2026-09-01')
    new = make_minute_ohlcv(days=2, start='2026-10-27', seed=2)
    loader = YahooMinuteLoader(cache_dir=tmp_path)
    
    monkeypatch.setattr(loader, '_download', lambda symbol: old)
    loader.load('TEST.NS')
    monkeypatch.setattr(loader, '_download', lambda symbol: new)
    df = loader.load('TEST.NS')
    
    assert len(df) == len(old) + len(new)
    assert df.index.min() == old.index.min() and df.index.max() == new.index.max()
    assert str(df.index.tz) == 'Asia/Kolkata'
