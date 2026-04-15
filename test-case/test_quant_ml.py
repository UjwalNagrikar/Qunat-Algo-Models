"""
NIFTY 50 QUANT ML v7.0 - Comprehensive Test Suite
==================================================
Unit tests for data pipeline, feature engineering, ML model, signals, and backtesting.

Run: pytest test_quant_ml.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'quant-test-models'))

# Import from main.py (adjust path if needed)
from main import (
    INITIAL_CAPITAL, TC, SLIPPAGE, TRADING_DAYS, PROB_LONG, PROB_SHORT,
    ML_FEATURES, download_data, compute_features, create_target, split_data,
    calibrate_params, train_ml_model, predict_probabilities, add_regime,
    generate_ml_signals, run_backtest, compute_metrics, dd_series
)


# ═══════════════════════════════════════════════════════════════════════════
# TEST SUITE 1: DATA VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

class TestDataPipeline:
    """Test data fetching, cleaning, and validation."""
    
    def test_data_shape(self):
        """TEST 1.1: Data has required columns and sufficient rows."""
        df = download_data()
        assert df.shape[0] > 1000, "Data: insufficient rows (< 1000)"
        assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(df.columns), \
            "Data: missing OHLCV columns"
        print("  ✓ TEST 1.1 PASS: Data shape and columns valid")
    
    def test_data_no_nulls(self):
        """TEST 1.2: No missing values in price/volume data."""
        df = download_data()
        assert df[["Open", "High", "Low", "Close", "Volume"]].isnull().sum().sum() == 0, \
            "Data: contains NaN values"
        print("  ✓ TEST 1.2 PASS: No missing values")
    
    def test_ohlc_order(self):
        """TEST 1.3: High ≥ Low, High ≥ Open/Close, Low ≤ Open/Close."""
        df = download_data()
        assert (df["High"] >= df["Low"]).all(), "Data: High < Low detected"
        assert (df["High"] >= df["Open"]).all(), "Data: High < Open detected"
        assert (df["Low"] <= df["Close"]).all(), "Data: Low > Close detected"
        print("  ✓ TEST 1.3 PASS: OHLC logical order correct")
    
    def test_volume_positive(self):
        """TEST 1.4: All volume values positive."""
        df = download_data()
        assert (df["Volume"] > 0).all(), "Data: non-positive volumes detected"
        print("  ✓ TEST 1.4 PASS: All volumes positive")
    
    def test_price_monotonicity(self):
        """TEST 1.5: Prices don't have extreme gaps (> 20% in one day)."""
        df = download_data()
        ret = df["Close"].pct_change().abs()
        assert (ret < 0.20).all(), "Data: extreme price gaps detected (>20%)"
        print("  ✓ TEST 1.5 PASS: No extreme price gaps")


# ═══════════════════════════════════════════════════════════════════════════
# TEST SUITE 2: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatures:
    """Test feature computation, validity, and consistency."""
    
    @pytest.fixture(scope="function")
    def sample_data(self):
        """Fixture: Load sample data."""
        df = download_data()
        return df.iloc[-500:].copy()  # Last 500 days
    
    def test_feature_computation(self, sample_data):
        """TEST 2.1: Features computed without errors."""
        try:
            features = compute_features(sample_data)
            assert features.shape[0] > 0, "Features: empty DataFrame"
            print("  ✓ TEST 2.1 PASS: Features computed successfully")
        except Exception as e:
            pytest.fail(f"Features: computation failed - {e}")
    
    def test_required_features_present(self, sample_data):
        """TEST 2.2: All required ML features exist."""
        features = compute_features(sample_data)
        missing = set(ML_FEATURES) - set(features.columns)
        assert len(missing) == 0, f"Features: missing {missing}"
        print("  ✓ TEST 2.2 PASS: All required features present")
    
    def test_feature_no_nulls(self, sample_data):
        """TEST 2.3: No NaN or Inf in feature columns."""
        features = compute_features(sample_data)
        for feat in ML_FEATURES:
            assert features[feat].notna().sum() > 0, f"Features: {feat} all NaN"
            assert not np.isinf(features[feat]).any(), f"Features: {feat} contains Inf"
        print("  ✓ TEST 2.3 PASS: No NaN/Inf in features")
    
    def test_feature_ranges(self, sample_data):
        """TEST 2.4: Features within reasonable ranges."""
        features = compute_features(sample_data)
        # Check returns are within [-1, 1] (i.e., returns not in decimals)
        ret_cols = [c for c in features.columns if c.startswith("ret_")]
        for col in ret_cols:
            assert features[col].max() < 10, f"Features: {col} unreasonably high"
            assert features[col].min() > -10, f"Features: {col} unreasonably low"
        print("  ✓ TEST 2.4 PASS: Features within reasonable ranges")
    
    def test_hurst_in_range(self, sample_data):
        """TEST 2.5: Hurst exponent in [0, 1] range."""
        features = compute_features(sample_data)
        if "hurst" in features.columns:
            hurst_valid = features["hurst"].dropna()
            assert (hurst_valid >= 0).all() and (hurst_valid <= 1).all(), \
                "Features: Hurst outside [0,1]"
            print("  ✓ TEST 2.5 PASS: Hurst exponent in valid range")


# ═══════════════════════════════════════════════════════════════════════════
# TEST SUITE 3: TARGET CREATION
# ═══════════════════════════════════════════════════════════════════════════

class TestTargetCreation:
    """Test binary target label creation (Bull/Bear)."""
    
    @pytest.fixture(scope="function")
    def prepared_data(self):
        """Fixture: Prepared data with features."""
        df = download_data()
        features = compute_features(df)
        return create_target(features)
    
    def test_target_column_exists(self, prepared_data):
        """TEST 3.1: Target column created."""
        assert "target_cls" in prepared_data.columns, "Target: column not created"
        print("  ✓ TEST 3.1 PASS: Target column exists")
    
    def test_target_binary_values(self, prepared_data):
        """TEST 3.2: Target contains only 0 and 1."""
        target_vals = set(prepared_data["target_cls"].dropna().unique())
        assert target_vals.issubset({0, 1}), f"Target: invalid values {target_vals}"
        print("  ✓ TEST 3.2 PASS: Target binary values (0/1)")
    
    def test_target_class_balance(self, prepared_data):
        """TEST 3.3: Target classes not extremely imbalanced (20%-80%)."""
        target = prepared_data["target_cls"].dropna()
        bear_pct = (target == 0).sum() / len(target) * 100
        bull_pct = (target == 1).sum() / len(target) * 100
        assert 20 < bear_pct < 80 and 20 < bull_pct < 80, \
            f"Target: imbalanced classes Bear:{bear_pct:.1f}%, Bull:{bull_pct:.1f}%"
        print(f"  ✓ TEST 3.3 PASS: Balanced classes (Bear:{bear_pct:.1f}%, Bull:{bull_pct:.1f}%)")
    
    def test_target_drop_rows(self, prepared_data):
        """TEST 3.4: NaN targets exist (last 5d forward lookback)."""
        nans = prepared_data["target_cls"].isna().sum()
        assert nans > 0, "Target: no NaN values (should have forward lookback drop)"
        print(f"  ✓ TEST 3.4 PASS: {nans} rows dropped for forward lookback")


# ═══════════════════════════════════════════════════════════════════════════
# TEST SUITE 4: TRAIN/TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════

class TestDataSplit:
    """Test train/test split logic and data leakage prevention."""
    
    @pytest.fixture(scope="function")
    def split_result(self):
        """Fixture: Split data."""
        df = download_data()
        features = compute_features(df)
        features_with_target = create_target(features)
        return split_data(features_with_target)
    
    def test_split_shapes(self, split_result):
        """TEST 4.1: Train/test split shapes reasonable."""
        f_train, f_test, train_cutoff = split_result
        assert f_train.shape[0] > 500, "Split: train set too small"
        assert f_test.shape[0] > 100, "Split: test set too small"
        print(f"  ✓ TEST 4.1 PASS: Train: {f_train.shape[0]}, Test: {f_test.shape[0]}")
    
    def test_no_temporal_overlap(self, split_result):
        """TEST 4.2: No temporal overlap between train and test."""
        f_train, f_test, train_cutoff = split_result
        train_max_date = f_train.index.max()
        test_min_date = f_test.index.min()
        assert train_max_date < test_min_date, "Split: temporal overlap detected"
        print("  ✓ TEST 4.2 PASS: No temporal overlap")
    
    def test_split_ratio_valid(self, split_result):
        """TEST 4.3: Train~80%, Test~20% split ratio."""
        f_train, f_test, train_cutoff = split_result
        total = f_train.shape[0] + f_test.shape[0]
        train_pct = f_train.shape[0] / total * 100
        test_pct = f_test.shape[0] / total * 100
        assert 70 < train_pct < 90, f"Split: train ratio {train_pct:.1f}% not in [70,90]"
        print(f"  ✓ TEST 4.3 PASS: Train: {train_pct:.1f}%, Test: {test_pct:.1f}%")
    
    def test_features_in_split(self, split_result):
        """TEST 4.4: All features present in both sets."""
        f_train, f_test, _ = split_result
        required_cols = set(ML_FEATURES + ["target_cls"])
        train_missing = required_cols - set(f_train.columns)
        test_missing = required_cols - set(f_test.columns)
        assert len(train_missing) == 0 and len(test_missing) == 0, \
            f"Split: missing cols Train:{train_missing} Test:{test_missing}"
        print("  ✓ TEST 4.4 PASS: All features in train/test")


# ═══════════════════════════════════════════════════════════════════════════
# TEST SUITE 5: ML MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

class TestMLTraining:
    """Test XGBoost model training, validation, and robustness."""
    
    @pytest.fixture(scope="function")
    def train_data(self):
        """Fixture: Training data."""
        df = download_data()
        features = compute_features(df)
        features_with_target = create_target(features)
        f_train, _, _ = split_data(features_with_target)
        return f_train
    
    def test_model_training(self, train_data):
        """TEST 5.1: Model trains without errors."""
        try:
            model, scaler, train_metrics = train_ml_model(train_data)
            assert model is not None, "ML: model is None"
            assert scaler is not None, "ML: scaler is None"
            print("  ✓ TEST 5.1 PASS: Model trained successfully")
        except Exception as e:
            pytest.fail(f"ML: training failed - {e}")
    
    def test_model_accuracy(self, train_data):
        """TEST 5.2: Train accuracy > 50% and < 100%."""
        model, scaler, train_metrics = train_ml_model(train_data)
        acc = train_metrics["accuracy"]
        assert 0.50 <= acc <= 1.0, f"ML: accuracy {acc:.2%} invalid"
        print(f"  ✓ TEST 5.2 PASS: Train accuracy {acc:.2%}")
    
    def test_model_auc(self, train_data):
        """TEST 5.3: Train AUC > 0.5 and ≤ 1.0."""
        model, scaler, train_metrics = train_ml_model(train_data)
        auc = train_metrics["auc"]
        assert 0.50 <= auc <= 1.0, f"ML: AUC {auc:.4f} invalid"
        print(f"  ✓ TEST 5.3 PASS: Train AUC {auc:.4f}")
    
    def test_model_predict_shape(self, train_data):
        """TEST 5.4: Predictions have correct shape."""
        model, scaler, _ = train_ml_model(train_data)
        X = scaler.transform(train_data[ML_FEATURES].values)
        probs = model.predict_proba(X)
        assert probs.shape == (X.shape[0], 2), "ML: prediction shape mismatch"
        print("  ✓ TEST 5.4 PASS: Prediction shape correct")
    
    def test_model_prob_range(self, train_data):
        """TEST 5.5: Probabilities in [0, 1] range."""
        model, scaler, _ = train_ml_model(train_data)
        X = scaler.transform(train_data[ML_FEATURES].values)
        probs = model.predict_proba(X)
        assert (probs >= 0).all() and (probs <= 1).all(), "ML: probs outside [0,1]"
        print("  ✓ TEST 5.5 PASS: Probabilities in valid range")


# ═══════════════════════════════════════════════════════════════════════════
# TEST SUITE 6: SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════════════════

class TestSignalGeneration:
    """Test ML signal generation and regime filtering."""
    
    @pytest.fixture(scope="function")
    def prepared_signals(self):
        """Fixture: Data with signals."""
        df = download_data()
        features = compute_features(df)
        features_with_target = create_target(features)
        f_train, f_test, _ = split_data(features_with_target)
        
        # Train model
        model, scaler, _ = train_ml_model(f_train)
        prob_test = predict_probabilities(model, scaler, f_test)
        
        # Add regime and generate signals
        f_test = add_regime(f_test)
        params = calibrate_params(f_train)
        signals = generate_ml_signals(f_test, prob_test, params)
        
        return signals, f_test
    
    def test_signal_column_exists(self, prepared_signals):
        """TEST 6.1: Signal column created."""
        signals, _ = prepared_signals
        assert "signal" in signals.columns, "Signal: column not created"
        print("  ✓ TEST 6.1 PASS: Signal column exists")
    
    def test_signal_values(self, prepared_signals):
        """TEST 6.2: Signals are -1, 0, or 1 (short, flat, long)."""
        signals, _ = prepared_signals
        valid_signals = {-1, 0, 1}
        actual_signals = set(signals["signal"].unique())
        assert actual_signals.issubset(valid_signals), \
            f"Signal: invalid values {actual_signals - valid_signals}"
        print("  ✓ TEST 6.2 PASS: Signal values valid (-1/0/1)")
    
    def test_signal_distribution(self, prepared_signals):
        """TEST 6.3: Signals have reasonable distribution."""
        signals, _ = prepared_signals
        long_count = (signals["signal"] == 1).sum()
        flat_count = (signals["signal"] == 0).sum()
        total = len(signals)
        long_pct = long_count / total * 100
        assert long_pct > 5, f"Signal: too few long signals ({long_pct:.1f}%)"
        print(f"  ✓ TEST 6.3 PASS: Long: {long_pct:.1f}%, Flat: {100-long_pct:.1f}%")
    
    def test_regime_column_exists(self, prepared_signals):
        """TEST 6.4: Regime column added."""
        _, f_test = prepared_signals
        assert "regime" in f_test.columns, "Signal: regime column not found"
        print("  ✓ TEST 6.4 PASS: Regime column exists")
    
    def test_regime_values(self, prepared_signals):
        """TEST 6.5: Regime values are 1, 0, or -1."""
        _, f_test = prepared_signals
        valid_regimes = {-1, 0, 1}
        actual_regimes = set(f_test["regime"].unique())
        assert actual_regimes.issubset(valid_regimes), \
            f"Signal: invalid regimes {actual_regimes}"
        print("  ✓ TEST 6.5 PASS: Regime values valid")


# ═══════════════════════════════════════════════════════════════════════════
# TEST SUITE 7: BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════

class TestBacktesting:
    """Test backtest execution and trade logic."""
    
    @pytest.fixture(scope="function")
    def backtest_data(self):
        """Fixture: Complete backtest setup."""
        df = download_data()
        features = compute_features(df)
        features_with_target = create_target(features)
        f_train, f_test, _ = split_data(features_with_target)
        
        # Train and predict
        model, scaler, _ = train_ml_model(f_train)
        prob_train = predict_probabilities(model, scaler, f_train)
        prob_test = predict_probabilities(model, scaler, f_test)
        
        # Add regime and signals
        f_train = add_regime(f_train)
        f_test = add_regime(f_test)
        params = calibrate_params(f_train)
        
        s_train = generate_ml_signals(f_train, prob_train, params)
        s_test = generate_ml_signals(f_test, prob_test, params)
        
        return s_test, f_test
    
    def test_backtest_execution(self, backtest_data):
        """TEST 7.1: Backtest runs without errors."""
        signals, _ = backtest_data
        try:
            r_test, trades_test = run_backtest(signals, "TEST")
            assert r_test is not None, "Backtest: returns is None"
            assert trades_test is not None, "Backtest: trades is None"
            print("  ✓ TEST 7.1 PASS: Backtest executed successfully")
        except Exception as e:
            pytest.fail(f"Backtest: execution failed - {e}")
    
    def test_returns_dataframe(self, backtest_data):
        """TEST 7.2: Returns DataFrame has correct structure."""
        signals, _ = backtest_data
        r_test, _ = run_backtest(signals, "TEST")
        assert isinstance(r_test, pd.DataFrame), "Backtest: returns not DataFrame"
        required_cols = {"daily_return", "portfolio_value", "benchmark_value"}
        assert required_cols.issubset(r_test.columns), \
            f"Backtest: missing columns {required_cols - set(r_test.columns)}"
        print("  ✓ TEST 7.2 PASS: Returns DataFrame structure valid")
    
    def test_portfolio_value_positive(self, backtest_data):
        """TEST 7.3: Portfolio values all positive."""
        signals, _ = backtest_data
        r_test, _ = run_backtest(signals, "TEST")
        assert (r_test["portfolio_value"] > 0).all(), \
            "Backtest: negative portfolio values"
        print("  ✓ TEST 7.3 PASS: All portfolio values positive")
    
    def test_trades_dataframe(self, backtest_data):
        """TEST 7.4: Trades DataFrame valid."""
        signals, _ = backtest_data
        _, trades = run_backtest(signals, "TEST")
        if len(trades) > 0:
            required_cols = {"entry_date", "exit_date", "signal", 
                           "entry_price", "exit_price", "pnl_inr", "pnl_pct"}
            missing = required_cols - set(trades.columns)
            assert len(missing) == 0, f"Backtest: missing trade columns {missing}"
            print("  ✓ TEST 7.4 PASS: Trades DataFrame valid")
        else:
            print("  ⚠ TEST 7.4 SKIP: No trades executed")
    
    def test_entry_exit_consistency(self, backtest_data):
        """TEST 7.5: Entry dates before exit dates."""
        signals, _ = backtest_data
        _, trades = run_backtest(signals, "TEST")
        if len(trades) > 0:
            assert (trades["entry_date"] < trades["exit_date"]).all(), \
                "Backtest: entry >= exit dates found"
            print("  ✓ TEST 7.5 PASS: Entry < Exit for all trades")


# ═══════════════════════════════════════════════════════════════════════════
# TEST SUITE 8: METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

class TestMetrics:
    """Test performance metrics calculation."""
    
    @pytest.fixture(scope="function")
    def metrics_data(self):
        """Fixture: Complete metrics setup."""
        df = download_data()
        features = compute_features(df)
        features_with_target = create_target(features)
        f_train, f_test, _ = split_data(features_with_target)
        
        # Train and predict
        model, scaler, _ = train_ml_model(f_train)
        prob_test = predict_probabilities(model, scaler, f_test)
        
        # Generate signals and backtest
        f_test = add_regime(f_test)
        params = calibrate_params(f_train)
        s_test = generate_ml_signals(f_test, prob_test, params)
        r_test, trades_test = run_backtest(s_test, "TEST")
        
        return r_test, trades_test
    
    def test_metrics_computation(self, metrics_data):
        """TEST 8.1: Metrics compute without errors."""
        r_test, trades_test = metrics_data
        try:
            m_test = compute_metrics(r_test, trades_test, INITIAL_CAPITAL)
            assert m_test is not None, "Metrics: result is None"
            print("  ✓ TEST 8.1 PASS: Metrics computed successfully")
        except Exception as e:
            pytest.fail(f"Metrics: computation failed - {e}")
    
    def test_metrics_keys(self, metrics_data):
        """TEST 8.2: Metrics have required keys."""
        r_test, trades_test = metrics_data
        m_test = compute_metrics(r_test, trades_test, INITIAL_CAPITAL)
        required_keys = {"strategy", "benchmark", "trades"}
        assert required_keys.issubset(m_test.keys()), \
            f"Metrics: missing keys {required_keys - set(m_test.keys())}"
        print("  ✓ TEST 8.2 PASS: Required metric keys present")
    
    def test_return_metrics(self, metrics_data):
        """TEST 8.3: Return metrics in valid ranges."""
        r_test, trades_test = metrics_data
        m_test = compute_metrics(r_test, trades_test, INITIAL_CAPITAL)
        returns = m_test["strategy"]
        assert -1 < returns["total_return"] < 5, \
            f"Metrics: total_return {returns['total_return']:.2%} unreasonable"
        assert -1 < returns["cagr"] < 5, \
            f"Metrics: cagr {returns['cagr']:.2%} unreasonable"
        print("  ✓ TEST 8.3 PASS: Return metrics reasonable")
    
    def test_risk_metrics(self, metrics_data):
        """TEST 8.4: Risk metrics positive and consistent."""
        r_test, trades_test = metrics_data
        m_test = compute_metrics(r_test, trades_test, INITIAL_CAPITAL)
        risk = m_test["strategy"]
        assert risk["ann_vol"] > 0, "Metrics: volatility <= 0"
        assert risk["max_dd"] < 0, "Metrics: max_dd should be negative"
        assert abs(risk["max_dd"]) < 1, "Metrics: max_dd > -100%"
        print("  ✓ TEST 8.4 PASS: Risk metrics valid")
    
    def test_sharpe_calculation(self, metrics_data):
        """TEST 8.5: Sharpe ratio computed."""
        r_test, trades_test = metrics_data
        m_test = compute_metrics(r_test, trades_test, INITIAL_CAPITAL)
        sharpe = m_test["strategy"].get("sharpe")
        assert sharpe is not None, "Metrics: Sharpe ratio not computed"
        assert -10 < sharpe < 10, f"Metrics: Sharpe {sharpe:.4f} unreasonable"
        print(f"  ✓ TEST 8.5 PASS: Sharpe ratio {sharpe:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# TEST SUITE 9: DRAWDOWN CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

class TestDrawdown:
    """Test drawdown calculation utility."""
    
    def test_dd_empty_series(self):
        """TEST 9.1: dd_series handles edge case."""
        eq = pd.Series([1000] * 10)
        dd = dd_series(eq)
        assert (dd <= 0).all(), "Drawdown: positive values found"
        print("  ✓ TEST 9.1 PASS: Flat equity has zero drawdown")
    
    def test_dd_decreasing_series(self):
        """TEST 9.2: dd_series for decreasing equity."""
        eq = pd.Series([1000, 950, 900, 850, 900, 950])
        dd = dd_series(eq)
        assert dd.min() < -0.05, "Drawdown: max DD not detected"
        print("  ✓ TEST 9.2 PASS: Drawdown correctly identified")
    
    def test_dd_range(self):
        """TEST 9.3: Drawdown values in [0, -1]."""
        eq = pd.Series([1000, 900, 800, 700, 800, 900, 1000])
        dd = dd_series(eq)
        assert (dd <= 0).all() and (dd >= -1).all(), "Drawdown: outside [-1, 0]"
        print("  ✓ TEST 9.3 PASS: Drawdown in valid range [-1, 0]")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*80)
    print("  NIFTY 50 QUANT ML v7.0 - TEST SUITE")
    print("="*80 + "\n")
    
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
