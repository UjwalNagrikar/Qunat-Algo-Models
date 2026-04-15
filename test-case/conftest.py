"""
pytest configuration and shared fixtures for NIFTY 50 QUANT ML tests
=====================================================================

This file provides:
- pytest configuration
- Shared fixtures for test suites
- Utility functions for common test operations
- Session-level setup/teardown
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'quant-test-models'))


# ═══════════════════════════════════════════════════════════════════════════
# PYTEST CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """Configure pytest markers and setup."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration-level"
    )
    config.addinivalue_line(
        "markers", "data: marks tests that require data fetching"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark slow tests."""
    slow_keywords = [
        "test_full_pipeline",
        "test_ml_training",
        "test_backtest",
        "integration"
    ]
    for item in items:
        if any(slow_keyword in item.nodeid for slow_keyword in slow_keywords):
            item.add_marker(pytest.mark.slow)


# ═══════════════════════════════════════════════════════════════════════════
# SESSION-LEVEL FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def test_config():
    """Session-level test configuration."""
    config = {
        "use_synthetic": False,  # Set True to always use synthetic data
        "data_cache_dir": "/tmp/test_data_cache",
        "timeout": 300,  # 5 minutes
        "random_seed": 42,
    }
    os.makedirs(config["data_cache_dir"], exist_ok=True)
    return config


@pytest.fixture(scope="session")
def random_seed(test_config):
    """Set random seed for reproducibility."""
    seed = test_config["random_seed"]
    np.random.seed(seed)
    return seed


# ═══════════════════════════════════════════════════════════════════════════
# DATA FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="function")
def sample_data_small():
    """Small sample data (100 days) - fast execution."""
    from main import download_data
    df = download_data()
    return df.iloc[-100:].copy()


@pytest.fixture(scope="function")
def sample_data_medium():
    """Medium sample data (500 days) - standard tests."""
    from main import download_data
    df = download_data()
    return df.iloc[-500:].copy()


@pytest.fixture(scope="function")
def sample_data_large():
    """Large sample data (full history) - integration tests."""
    from main import download_data
    df = download_data()
    return df.copy()


@pytest.fixture(scope="function")
def synthetic_data_small():
    """Generate small synthetic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100)
    
    # GBM simulation
    close_prices = 30000 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, 100)))
    
    df = pd.DataFrame({
        'Open': close_prices * (1 + np.random.uniform(-0.002, 0.002, 100)),
        'High': close_prices * (1 + np.abs(np.random.normal(0.005, 0.01, 100))),
        'Low': close_prices * (1 - np.abs(np.random.normal(0.005, 0.01, 100))),
        'Close': close_prices,
        'Volume': np.random.randint(10000000, 500000000, 100)
    }, index=dates)
    
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    return df


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE & PREPARATION FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="function")
def data_with_features(sample_data_medium):
    """Data with computed features."""
    from main import compute_features
    return compute_features(sample_data_medium)


@pytest.fixture(scope="function")
def data_with_target(data_with_features):
    """Data with features and binary target."""
    from main import create_target
    return create_target(data_with_features)


@pytest.fixture(scope="function")
def split_data_fixture(data_with_target):
    """Train/test split."""
    from main import split_data
    f_train, f_test, train_cutoff = split_data(data_with_target)
    return {
        'train': f_train,
        'test': f_test,
        'cutoff': train_cutoff,
        'train_size': len(f_train),
        'test_size': len(f_test),
    }


@pytest.fixture(scope="function")
def trained_model(split_data_fixture):
    """Trained ML model with scaler."""
    from main import train_ml_model
    f_train = split_data_fixture['train']
    model, scaler, train_metrics = train_ml_model(f_train)
    return {
        'model': model,
        'scaler': scaler,
        'metrics': train_metrics,
    }


@pytest.fixture(scope="function")
def signals_generated(split_data_fixture, trained_model):
    """Generated ML signals for train and test."""
    from main import (
        predict_probabilities, add_regime, generate_ml_signals, calibrate_params
    )
    
    model = trained_model['model']
    scaler = trained_model['scaler']
    f_train = split_data_fixture['train']
    f_test = split_data_fixture['test']
    
    prob_train = predict_probabilities(model, scaler, f_train)
    prob_test = predict_probabilities(model, scaler, f_test)
    
    f_train = add_regime(f_train)
    f_test = add_regime(f_test)
    
    params = calibrate_params(f_train)
    
    s_train = generate_ml_signals(f_train, prob_train, params)
    s_test = generate_ml_signals(f_test, prob_test, params)
    
    return {
        'train': s_train,
        'test': s_test,
        'params': params,
    }


@pytest.fixture(scope="function")
def backtest_results(signals_generated):
    """Backtest results for train and test."""
    from main import run_backtest
    
    s_train = signals_generated['train']
    s_test = signals_generated['test']
    
    r_train, trades_train = run_backtest(s_train, "TRAIN")
    r_test, trades_test = run_backtest(s_test, "TEST")
    
    return {
        'returns_train': r_train,
        'returns_test': r_test,
        'trades_train': trades_train,
        'trades_test': trades_test,
    }


@pytest.fixture(scope="function")
def metrics_computed(backtest_results):
    """Computed performance metrics."""
    from main import compute_metrics, INITIAL_CAPITAL
    
    r_train = backtest_results['returns_train']
    r_test = backtest_results['returns_test']
    trades_train = backtest_results['trades_train']
    trades_test = backtest_results['trades_test']
    
    m_train = compute_metrics(r_train, trades_train, INITIAL_CAPITAL)
    m_test = compute_metrics(r_test, trades_test, INITIAL_CAPITAL)
    
    return {
        'train': m_train,
        'test': m_test,
    }


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def timer():
    """Simple timer utility for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, *args):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


@pytest.fixture
def assert_metrics():
    """Assertion helper for metrics validation."""
    
    class MetricsAssert:
        @staticmethod
        def is_probability(value):
            assert 0 <= value <= 1, f"Probability {value} not in [0,1]"
        
        @staticmethod
        def is_positive(value):
            assert value > 0, f"Value {value} not positive"
        
        @staticmethod
        def is_negative(value):
            assert value < 0, f"Value {value} not negative"
        
        @staticmethod
        def is_valid_return(value):
            assert -1 < value < 10, f"Return {value:.2%} unreasonable"
        
        @staticmethod
        def is_valid_sharpe(value):
            assert -10 < value < 10, f"Sharpe {value:.4f} unreasonable"
        
        @staticmethod
        def dataframe_has_columns(df, required_cols):
            missing = set(required_cols) - set(df.columns)
            assert len(missing) == 0, f"Missing columns: {missing}"
        
        @staticmethod
        def no_nulls(series):
            assert series.notna().sum() == len(series), \
                f"{series.name} has {series.isna().sum()} NaN values"
        
        @staticmethod
        def no_inf(series):
            assert not np.isinf(series).any(), f"{series.name} contains Inf"
    
    return MetricsAssert()


# ═══════════════════════════════════════════════════════════════════════════
# HOOKS FOR TEST LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test."""
    np.random.seed(42)
    yield
    # Cleanup if needed
    pass


def pytest_runtest_logreport(report):
    """Hook for test result reporting."""
    if report.when == "call":
        if report.passed:
            duration = f"{report.duration:.2f}s" if report.duration else "?"
            print(f"  ✅ PASS ({duration})")
        elif report.failed:
            print(f"  ❌ FAIL")
        elif report.skipped:
            print(f"  ⚠️  SKIP")


# ═══════════════════════════════════════════════════════════════════════════
# PARAMETRIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

probability_thresholds = [
    (0.50, 0.45),  # Current
    (0.55, 0.40),  # More conservative
    (0.45, 0.50),  # More aggressive
]

kelly_fractions = [1.0, 1.5, 2.0]

holding_periods = [4, 7, 14]


@pytest.fixture(params=probability_thresholds)
def prob_threshold_param(request):
    """Parametrized probability thresholds."""
    return request.param


@pytest.fixture(params=kelly_fractions)
def kelly_param(request):
    """Parametrized Kelly fraction."""
    return request.param


@pytest.fixture(params=holding_periods)
def holding_param(request):
    """Parametrized holding period."""
    return request.param


# ═══════════════════════════════════════════════════════════════════════════
# ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def expect_error():
    """Context manager for expected errors."""
    from contextlib import contextmanager
    
    @contextmanager
    def _expect_error(error_type):
        try:
            yield
            pytest.fail(f"Expected {error_type.__name__} but no error raised")
        except error_type:
            pass  # Expected
    
    return _expect_error


# ═══════════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session", autouse=True)
def cleanup_session():
    """Clean up after test session."""
    yield
    # Cleanup code here if needed


"""
USAGE EXAMPLES
==============

# Using sample data
def test_something(sample_data_medium):
    df = sample_data_medium
    assert len(df) == 500

# Using complete pipeline
def test_pipeline(metrics_computed):
    metrics = metrics_computed['test']
    assert metrics['strategy']['sharpe'] is not None

# Using timer
def test_performance(timer):
    with timer as t:
        # Code to time
        pass
    print(f"Took {t.elapsed:.2f}s")

# Using assertions
def test_metrics(assert_metrics):
    assert_metrics.is_probability(0.5)
    assert_metrics.is_valid_sharpe(0.5)

"""
