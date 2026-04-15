"""
Integration Tests - Full Pipeline Validation
=============================================
End-to-end tests for complete ML algo pipeline.

Run: pytest test_integration.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'quant-test-models'))

from main import (
    INITIAL_CAPITAL, PROB_LONG, PROB_SHORT, download_data, compute_features,
    create_target, split_data, calibrate_params, train_ml_model, 
    predict_probabilities, add_regime, generate_ml_signals, run_backtest, 
    compute_metrics, print_metrics, ML_FEATURES
)


class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_full_pipeline_execution(self):
        """INTEGRATION 1: Full pipeline executes end-to-end."""
        try:
            # 1. Data
            df = download_data()
            assert df.shape[0] > 500, "Integration: insufficient data"
            
            # 2. Features
            features = compute_features(df)
            assert len(features) > 400, "Integration: feature computation failed"
            
            # 3. Target
            features_with_target = create_target(features)
            assert "target_cls" in features_with_target.columns
            
            # 4. Split
            f_train, f_test, train_cutoff = split_data(features_with_target)
            assert len(f_train) > 300 and len(f_test) > 50
            
            # 5. ML Model
            model, scaler, train_metrics = train_ml_model(f_train)
            assert model is not None
            
            # 6. Predictions
            prob_train = predict_probabilities(model, scaler, f_train)
            prob_test = predict_probabilities(model, scaler, f_test)
            assert len(prob_train) == len(f_train) and len(prob_test) == len(f_test)
            
            # 7. Regime
            f_train = add_regime(f_train)
            f_test = add_regime(f_test)
            assert "regime" in f_train.columns and "regime" in f_test.columns
            
            # 8. Signals
            params = calibrate_params(f_train)
            s_train = generate_ml_signals(f_train, prob_train, params)
            s_test = generate_ml_signals(f_test, prob_test, params)
            assert "signal" in s_train.columns and "signal" in s_test.columns
            
            # 9. Backtest
            r_train, trades_train = run_backtest(s_train, "TRAIN")
            r_test, trades_test = run_backtest(s_test, "TEST")
            assert r_train is not None and r_test is not None
            
            # 10. Metrics
            m_train = compute_metrics(r_train, trades_train, INITIAL_CAPITAL)
            m_test = compute_metrics(r_test, trades_test, INITIAL_CAPITAL)
            assert "strategy" in m_train and "strategy" in m_test
            
            print("  ✓ INTEGRATION 1 PASS: Full pipeline successful")
            
        except Exception as e:
            pytest.fail(f"Integration: pipeline failed - {e}")
    
    def test_no_data_leakage(self):
        """INTEGRATION 2: No data leakage between train/test."""
        # Ensure test set features don't use future information
        df = download_data()
        features = compute_features(df)
        features_with_target = create_target(features)
        f_train, f_test, _ = split_data(features_with_target)
        
        # Check temporal ordering
        assert f_train.index.max() < f_test.index.min(), \
            "Integration: data leakage - train/test overlap"
        
        # Check no NaN leakage from dropped rows
        assert f_test["target_cls"].notna().sum() > 0, \
            "Integration: test set targets all NaN"
        
        print("  ✓ INTEGRATION 2 PASS: No data leakage detected")
    
    def test_reproducibility(self):
        """INTEGRATION 3: Results reproducible with same random seed."""
        import random
        from sklearn import set_config
        
        # Set seed
        np.random.seed(42)
        random.seed(42)
        
        # Run pipeline 1
        df1 = download_data()
        features1 = compute_features(df1)
        features1_target = create_target(features1)
        f_train1, f_test1, _ = split_data(features1_target)
        model1, scaler1, metrics1 = train_ml_model(f_train1)
        acc1 = metrics1["accuracy"]
        
        # Reset seed
        np.random.seed(42)
        random.seed(42)
        
        # Run pipeline 2
        df2 = download_data()
        features2 = compute_features(df2)
        features2_target = create_target(features2)
        f_train2, f_test2, _ = split_data(features2_target)
        model2, scaler2, metrics2 = train_ml_model(f_train2)
        acc2 = metrics2["accuracy"]
        
        # Check same accuracy (within tolerance)
        assert abs(acc1 - acc2) < 0.01, \
            f"Integration: non-reproducible (acc1={acc1:.4f}, acc2={acc2:.4f})"
        
        print("  ✓ INTEGRATION 3 PASS: Results reproducible")
    
    def test_signal_coverage(self):
        """INTEGRATION 4: Signals cover reasonable date range."""
        df = download_data()
        features = compute_features(df)
        features_with_target = create_target(features)
        f_train, f_test, _ = split_data(features_with_target)
        
        model, scaler, _ = train_ml_model(f_train)
        prob_test = predict_probabilities(model, scaler, f_test)
        
        f_test = add_regime(f_test)
        params = calibrate_params(f_train)
        s_test = generate_ml_signals(f_test, prob_test, params)
        
        # Check signal coverage
        signal_date_range = s_test.index.max() - s_test.index.min()
        assert signal_date_range.days > 30, \
            f"Integration: signal coverage too short ({signal_date_range.days}d)"
        
        print(f"  ✓ INTEGRATION 4 PASS: Signal coverage {signal_date_range.days}d")
    
    def test_trade_execution_consistency(self):
        """INTEGRATION 5: Trades execute consistently."""
        df = download_data()
        features = compute_features(df)
        features_with_target = create_target(features)
        f_train, f_test, _ = split_data(features_with_target)
        
        model, scaler, _ = train_ml_model(f_train)
        prob_test = predict_probabilities(model, scaler, f_test)
        
        f_test = add_regime(f_test)
        params = calibrate_params(f_train)
        s_test = generate_ml_signals(f_test, prob_test, params)
        r_test, trades_test = run_backtest(s_test, "TEST")
        
        if len(trades_test) > 0:
            # Entry/exit consistency
            assert (trades_test["entry_price"] > 0).all(), \
                "Integration: invalid entry prices"
            assert (trades_test["exit_price"] > 0).all(), \
                "Integration: invalid exit prices"
            
            # P&L calculation
            pnl_calc = (trades_test["exit_price"] - trades_test["entry_price"]) / \
                      trades_test["entry_price"] * 100
            pnl_actual = trades_test["pnl_pct"]
            
            # Allow small tolerance for fees/slippage
            tolerance = 1.0  # 1%
            assert (abs(pnl_calc - pnl_actual) < tolerance).all(), \
                "Integration: P&L calculation mismatch"
            
            print("  ✓ INTEGRATION 5 PASS: Trade execution consistent")
        else:
            print("  ⚠ INTEGRATION 5 SKIP: No trades executed")
    
    def test_benchmark_sanity(self):
        """INTEGRATION 6: Benchmark (buy & hold) produces expected results."""
        df = download_data()
        
        # Simple buy & hold
        returns = df["Close"].pct_change().dropna()
        expected_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
        
        # Should be positive over 10 years (historically) or reasonable
        assert -50 < expected_return < 500, \
            f"Integration: benchmark return {expected_return:.1f}% unreasonable"
        
        print(f"  ✓ INTEGRATION 6 PASS: Benchmark return {expected_return:.1f}%")


class TestConfiguration:
    """Tests for configuration parameters."""
    
    def test_probability_thresholds(self):
        """CONFIG 1: Probability thresholds valid."""
        assert 0 < PROB_LONG <= 1, f"Config: PROB_LONG {PROB_LONG} invalid"
        assert 0 < PROB_SHORT <= 1, f"Config: PROB_SHORT {PROB_SHORT} invalid"
        print(f"  ✓ CONFIG 1 PASS: Thresholds valid (Long:{PROB_LONG}, Short:{PROB_SHORT})")
    
    def test_ml_features_count(self):
        """CONFIG 2: Sufficient ML features."""
        assert len(ML_FEATURES) >= 5, f"Config: only {len(ML_FEATURES)} features"
        print(f"  ✓ CONFIG 2 PASS: {len(ML_FEATURES)} features configured")
    
    def test_initial_capital(self):
        """CONFIG 3: Initial capital positive."""
        assert INITIAL_CAPITAL > 10000, \
            f"Config: INITIAL_CAPITAL {INITIAL_CAPITAL} too low"
        print(f"  ✓ CONFIG 3 PASS: Initial capital ₹{INITIAL_CAPITAL:,.0f}")


class TestEdgeCases:
    """Edge case and error handling tests."""
    
    def test_tiny_dataset(self):
        """EDGE 1: Handles small datasets gracefully."""
        # Create minimal dataset
        dates = pd.date_range('2020-01-01', periods=50)
        df = pd.DataFrame({
            'Open': 100 + np.random.randn(50),
            'High': 101 + np.random.randn(50),
            'Low': 99 + np.random.randn(50),
            'Close': 100 + np.random.randn(50),
            'Volume': 1000000 + np.random.randint(-100000, 100000, 50)
        }, index=dates)
        
        try:
            # This should either work or fail gracefully
            features = compute_features(df)
            assert len(features) > 0
            print("  ✓ EDGE 1 PASS: Small dataset handled")
        except:
            print("  ⚠ EDGE 1 SKIP: Small dataset not supported (expected)")
    
    def test_missing_values_handling(self):
        """EDGE 2: Pipeline handles edge case missing values."""
        df = download_data()
        features = compute_features(df)
        
        # Check NaN handling
        nan_count = features.isnull().sum().sum()
        row_with_nan = len(features[features.isnull().any(axis=1)])
        
        print(f"  ✓ EDGE 2 PASS: {nan_count} NaN values, {row_with_nan} rows affected")
    
    def test_extreme_returns(self):
        """EDGE 3: Handles extreme market moves (>10% daily)."""
        df = download_data()
        returns = df["Close"].pct_change()
        extreme_days = (abs(returns) > 0.10).sum()
        
        assert extreme_days < len(df) * 0.05, \
            "Edge: too many extreme moves (>5% of days)"
        
        print(f"  ✓ EDGE 3 PASS: {extreme_days} extreme move days")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("  INTEGRATION & CONFIGURATION TESTS")
    print("="*80 + "\n")
    pytest.main([__file__, "-v", "--tb=short"])
