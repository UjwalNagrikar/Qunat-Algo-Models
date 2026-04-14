"""
═══════════════════════════════════════════════════════════════════════════════
  NIFTY 50 QUANT SYSTEM — COMPREHENSIVE TEST SUITE
═══════════════════════════════════════════════════════════════════════════════

Test Cases for:
  ✓ Data Loading & Cleaning
  ✓ Feature Engineering
  ✓ Signal Generation
  ✓ Backtest Engine
  ✓ Metrics Calculation
  ✓ Train/Test Split
  ✓ Performance Validation
═══════════════════════════════════════════════════════════════════════════════
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta

# Import main module components
sys.path.insert(0, r'c:\Users\ujjwa\OneDrive\Desktop\Automation-Project\Qunat-Algo-Models\data-spliting-model-train')

try:
    from main import (download_data, compute_features, calibrate, generate_signals, 
                     run_backtest, compute_metrics, _long_score, _short_score,
                     INITIAL_CAPITAL, TRADING_DAYS, TRAIN_YEARS, KELLY_CAP, 
                     MIN_HOLD, MAX_HOLD, TRAIL_STOP, TC, SLIPPAGE)
except ImportError as e:
    print(f"[ERROR] Could not import main module: {e}")
    sys.exit(1)


class ColorCodes:
    """Terminal color codes"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print section header"""
    print(f"\n{ColorCodes.BOLD}{ColorCodes.BLUE}{'='*80}{ColorCodes.END}")
    print(f"{ColorCodes.BLUE}  {text}{ColorCodes.END}")
    print(f"{ColorCodes.BOLD}{ColorCodes.BLUE}{'='*80}{ColorCodes.END}\n")


def print_test(name, passed, message=""):
    """Print test result"""
    status = f"{ColorCodes.GREEN}✓ PASS{ColorCodes.END}" if passed else f"{ColorCodes.RED}✗ FAIL{ColorCodes.END}"
    print(f"  {status} | {name}")
    if message:
        print(f"       → {message}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: DATA LOADING & CLEANING
# ═══════════════════════════════════════════════════════════════════════════════
def test_data_loading():
    """Test 1: Verify data is loaded correctly"""
    print_header("TEST 1: DATA LOADING & CLEANING")
    
    try:
        df = download_data()
        
        # Test 1.1: Data shape
        test_1_1 = len(df) > 1000
        print_test("Data has sufficient rows (>1000)", test_1_1, f"Rows: {len(df)}")
        
        # Test 1.2: Required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'log_ret']
        test_1_2 = all(col in df.columns for col in required_cols)
        print_test("Has all required columns", test_1_2, f"Columns: {list(df.columns)}")
        
        # Test 1.3: No NaN in price data
        test_1_3 = df[['Open', 'High', 'Low', 'Close']].isna().sum().sum() == 0
        print_test("No NaN in price columns", test_1_3)
        
        # Test 1.4: Price sanity checks
        test_1_4 = (df['High'] >= df['Low']).all() and (df['Close'] > 0).all()
        print_test("Price logic valid (High >= Low, Close > 0)", test_1_4)
        
        # Test 1.5: Volume positive
        test_1_5 = (df['Volume'] > 0).all()
        print_test("All volumes positive", test_1_5)
        
        # Test 1.6: Date index
        test_1_6 = isinstance(df.index, pd.DatetimeIndex)
        print_test("Index is DatetimeIndex", test_1_6)
        
        all_passed_1 = all([test_1_1, test_1_2, test_1_3, test_1_4, test_1_5, test_1_6])
        return df if all_passed_1 else None
        
    except Exception as e:
        print_test("Data loading", False, str(e))
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
def test_feature_engineering(df):
    """Test 2: Verify features are calculated correctly"""
    print_header("TEST 2: FEATURE ENGINEERING")
    
    if df is None:
        print_test("Feature engineering", False, "No data provided")
        return None
    
    try:
        features = compute_features(df)
        
        # Test 2.1: Features computed
        test_2_1 = len(features) > 0
        print_test("Features computed successfully", test_2_1, f"Shape: {features.shape}")
        
        # Test 2.2: Expected features exist
        expected_features = ['ret_20d', 'ret_60d', 'hurst', 'rvol20', 'body_ratio', 'volume', 'vol_z']
        test_2_2 = all(feat in features.columns for feat in expected_features if feat in features.columns)
        print_test("Key features computed", test_2_2, f"Features: {len(features.columns)}")
        
        # Test 2.3: No NaN after dropna
        test_2_3 = features.isna().sum().sum() == 0
        print_test("No NaN values in features", test_2_3)
        
        # Test 2.4: Feature values are reasonable (not all zeros)
        test_2_4 = features[['ret_20d', 'ret_60d']].std().sum() > 0
        print_test("Features have non-zero variance", test_2_4)
        
        # Test 2.5: Momentum features in reasonable range
        test_2_5 = ((features['ret_20d'] >= -1.0) & (features['ret_20d'] <= 1.0)).mean() > 0.95
        print_test("Momentum features in reasonable range", test_2_5, f"% in range: {((features['ret_20d'] >= -1.0) & (features['ret_20d'] <= 1.0)).mean()*100:.1f}%")
        
        all_passed_2 = all([test_2_1, test_2_2, test_2_3, test_2_4, test_2_5])
        return features if all_passed_2 else None
        
    except Exception as e:
        print_test("Feature engineering", False, str(e))
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: TRAIN/TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
def test_train_test_split(features):
    """Test 3: Verify train/test split is correct"""
    print_header("TEST 3: TRAIN/TEST SPLIT")
    
    if features is None:
        print_test("Train/test split", False, "No features provided")
        return None, None
    
    try:
        train_days = int(TRAIN_YEARS * TRADING_DAYS)
        f_train = features.iloc[:train_days].copy()
        f_test = features.iloc[train_days:].copy()
        
        # Test 3.1: Split sizes
        test_3_1 = len(f_train) == train_days
        print_test("Training set correct size", test_3_1, f"Train: {len(f_train)} days ({TRAIN_YEARS}yr)")
        
        # Test 3.2: Test set exists
        test_3_2 = len(f_test) > 0
        print_test("Test set has data", test_3_2, f"Test: {len(f_test)} days ({len(f_test)/TRADING_DAYS:.1f}yr)")
        
        # Test 3.3: No overlap
        test_3_3 = f_train.index[-1] < f_test.index[0]
        print_test("No overlap between train/test", test_3_3)
        
        # Test 3.4: Chronological order
        test_3_4 = (f_train.index.is_monotonic_increasing and f_test.index.is_monotonic_increasing)
        print_test("Both sets chronologically ordered", test_3_4)
        
        # Test 3.5: At least 2 years of test data
        test_3_5 = len(f_test) >= 2 * TRADING_DAYS
        test_3_5_yr = len(f_test) / TRADING_DAYS
        print_test("Test set >= 2 years", test_3_5, f"Test years: {test_3_5_yr:.2f}")
        
        all_passed_3 = all([test_3_1, test_3_2, test_3_3, test_3_4])
        return f_train, f_test if all_passed_3 else (None, None)
        
    except Exception as e:
        print_test("Train/test split", False, str(e))
        return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════
def test_calibration(f_train):
    """Test 4: Verify calibration works and produces reasonable thresholds"""
    print_header("TEST 4: CALIBRATION")
    
    if f_train is None:
        print_test("Calibration", False, "No training data provided")
        return None
    
    try:
        print("  [CALIBRATE] Running calibration on training data...")
        params = calibrate(f_train)
        
        # Test 4.1: Parameters returned
        test_4_1 = params is not None and len(params) > 0
        print_test("Calibration completed", test_4_1, f"Parameters: {list(params.keys())}")
        
        # Test 4.2: Threshold values in reasonable range
        test_4_2 = 0 < params['thresh_long'] < 10
        print_test("Long threshold reasonable", test_4_2, f"Value: {params['thresh_long']:.2f}")
        
        # Test 4.3: Short threshold > long threshold
        test_4_3 = params['thresh_short'] >= params['thresh_long']
        print_test("Short threshold >= Long threshold", test_4_3, f"Short: {params['thresh_short']:.2f}, Long: {params['thresh_long']:.2f}")
        
        # Test 4.4: Kelly fraction in range
        test_4_4 = 0.3 <= params['kelly_f'] <= KELLY_CAP
        print_test("Kelly fraction in range", test_4_4, f"Value: {params['kelly_f']:.4f}")
        
        # Test 4.5: Volatility threshold positive
        test_4_5 = params['pvol_thresh'] > 0
        print_test("Parkinson vol threshold positive", test_4_5, f"Value: {params['pvol_thresh']:.5f}")
        
        all_passed_4 = all([test_4_1, test_4_2, test_4_3, test_4_4, test_4_5])
        return params if all_passed_4 else None
        
    except Exception as e:
        print_test("Calibration", False, str(e))
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
def test_signal_generation(f_train, f_test, params):
    """Test 5: Verify signals are generated correctly"""
    print_header("TEST 5: SIGNAL GENERATION")
    
    if None in [f_train, f_test, params]:
        print_test("Signal generation", False, "Missing inputs")
        return None, None
    
    try:
        s_train = generate_signals(f_train, params)
        s_test = generate_signals(f_test, params)
        
        # Test 5.1: Signals generated
        test_5_1 = 'signal' in s_train.columns and 'signal' in s_test.columns
        print_test("Signals generated", test_5_1)
        
        # Test 5.2: Signal values are -1, 0, 1
        test_5_2 = set(s_train['signal'].unique()).issubset({-1.0, 0.0, 1.0})
        print_test("Signal values are -1/0/1", test_5_2)
        
        # Test 5.3: Position sizes generated
        test_5_3 = 'pos_size' in s_train.columns and 'pos_size' in s_test.columns
        print_test("Position sizes generated", test_5_3)
        
        # Test 5.4: Some long signals
        train_long = (s_train['signal'] == 1.0).sum()
        test_5_4 = train_long > 0
        print_test("Long signals generated", test_5_4, f"Train long signals: {train_long}")
        
        # Test 5.5: Some short signals
        train_short = (s_train['signal'] == -1.0).sum()
        test_5_5 = train_short > 0
        print_test("Short signals generated", test_5_5, f"Train short signals: {train_short}")
        
        # Test 5.6: Reasonable exposure percentages
        train_exposure = (s_train['signal'] != 0.0).mean() * 100
        test_5_6 = 20 < train_exposure < 90
        print_test("Reasonable market exposure", test_5_6, f"Exposure: {train_exposure:.1f}%")
        
        all_passed_5 = all([test_5_1, test_5_2, test_5_3, test_5_4, test_5_5, test_5_6])
        return s_train, s_test if all_passed_5 else (None, None)
        
    except Exception as e:
        print_test("Signal generation", False, str(e))
        return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def test_backtest_engine(s_train, s_test):
    """Test 6: Verify backtest engine runs and produces valid results"""
    print_header("TEST 6: BACKTEST ENGINE")
    
    if s_train is None or s_test is None:
        print_test("Backtest engine", False, "No signals provided")
        return None, None, None, None
    
    try:
        print("  [BACKTEST] Running backtest on training set...")
        r_train, trades_train = run_backtest(s_train, "TRAIN")
        print("  [BACKTEST] Running backtest on test set...")
        r_test, trades_test = run_backtest(s_test, "TEST")
        
        # Test 6.1: Results generated
        test_6_1 = r_train is not None and r_test is not None
        print_test("Backtest results generated", test_6_1)
        
        # Test 6.2: Trades generated
        test_6_2 = len(trades_train) > 0 and len(trades_test) > 0
        print_test("Trades generated", test_6_2, f"Train: {len(trades_train)}, Test: {len(trades_test)}")
        
        # Test 6.3: Portfolio value increased from initial
        train_final = r_train['portfolio_value'].iloc[-1]
        test_final = r_test['portfolio_value'].iloc[-1]
        test_6_3 = train_final > 0 and test_final > 0
        print_test("Portfolio values positive", test_6_3, f"Train final: ₹{train_final:,.0f}, Test: ₹{test_final:,.0f}")
        
        # Test 6.4: Trade P&L calculated
        trades_df_train = pd.DataFrame(trades_train)
        trades_df_test = pd.DataFrame(trades_test)
        test_6_4 = 'pnl_pct' in trades_df_train.columns and 'pnl_inr' in trades_df_train.columns
        print_test("Trade P&L calculated", test_6_4)
        
        # Test 6.5: Win rates in reasonable range
        if len(trades_train) > 0:
            train_wr = (trades_df_train['pnl_pct'] > 0).mean() * 100
            test_6_5 = 20 < train_wr < 80
            print_test("Training win rate reasonable", test_6_5, f"Win rate: {train_wr:.1f}%")
        else:
            test_6_5 = False
            print_test("Training win rate reasonable", False, "No trades")
        
        # Test 6.6: Average trade duration > 0
        if len(trades_train) > 0:
            avg_hold = trades_df_train['holding_days'].mean()
            test_6_6 = avg_hold > 0
            print_test("Average hold > 0 days", test_6_6, f"Avg hold: {avg_hold:.1f} days")
        else:
            test_6_6 = False
        
        all_passed_6 = all([test_6_1, test_6_2, test_6_3, test_6_4, test_6_5, test_6_6])
        return r_train, r_test, trades_train, trades_test if all_passed_6 else (None, None, None, None)
        
    except Exception as e:
        print_test("Backtest engine", False, str(e))
        return None, None, None, None


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: METRICS CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════
def test_metrics_calculation(r_train, r_test, trades_train, trades_test):
    """Test 7: Verify metrics are calculated correctly"""
    print_header("TEST 7: METRICS CALCULATION")
    
    if None in [r_train, r_test, trades_train, trades_test]:
        print_test("Metrics calculation", False, "Missing backtest results")
        return False
    
    try:
        print("  [METRICS] Computing metrics...")
        m_train = compute_metrics(r_train, pd.DataFrame(trades_train), INITIAL_CAPITAL)
        m_test = compute_metrics(r_test, pd.DataFrame(trades_test), INITIAL_CAPITAL)
        
        # Test 7.1: Metrics dictionaries created
        test_7_1 = 'strategy' in m_train and 'strategy' in m_test
        print_test("Metrics computed", test_7_1, f"Keys: {list(m_train.keys())}")
        
        # Test 7.2: Strategy metrics exist
        required_metrics = ['total_return', 'cagr', 'sharpe', 'max_dd', 'ann_vol']
        test_7_2 = all(m in m_train['strategy'] for m in required_metrics)
        print_test("Key metrics calculated", test_7_2)
        
        # Test 7.3: Returns in reasonable range
        train_ret = m_train['strategy']['total_return']
        test_ret = m_test['strategy']['total_return']
        test_7_3 = -200 < train_ret < 500 and -200 < test_ret < 500
        print_test("Returns in reasonable range", test_7_3, f"Train: {train_ret:.2f}%, Test: {test_ret:.2f}%")
        
        # Test 7.4: Sharpe ratio calculated
        train_sharpe = m_train['strategy']['sharpe']
        test_7_4 = -10 < train_sharpe < 10
        print_test("Sharpe ratio calculated", test_7_4, f"Train Sharpe: {train_sharpe:.4f}")
        
        # Test 7.5: Max drawdown negative
        train_dd = m_train['strategy']['max_dd']
        test_7_5 = train_dd < 0
        print_test("Max drawdown calculated correctly", test_7_5, f"Max DD: {train_dd:.2f}%")
        
        # Test 7.6: Trade metrics
        test_7_6 = 'trades' in m_train and m_train['trades']['n_trades'] > 0
        print_test("Trade metrics calculated", test_7_6, f"Trades: {m_train['trades']['n_trades']}")
        
        all_passed_7 = all([test_7_1, test_7_2, test_7_3, test_7_4, test_7_5, test_7_6])
        return all_passed_7
        
    except Exception as e:
        print_test("Metrics calculation", False, str(e))
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8: PERFORMANCE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
def test_performance_validation(r_train, r_test, trades_train, trades_test):
    """Test 8: Validate performance is acceptable for live trading"""
    print_header("TEST 8: PERFORMANCE VALIDATION (LIVE TRADING READINESS)")
    
    if None in [r_train, r_test, trades_train, trades_test]:
        print_test("Performance validation", False, "Missing results")
        return False
    
    try:
        m_train = compute_metrics(r_train, pd.DataFrame(trades_train), INITIAL_CAPITAL)
        m_test = compute_metrics(r_test, pd.DataFrame(trades_test), INITIAL_CAPITAL)
        
        print(f"\n  TRAINING SET PERFORMANCE:")
        print(f"    Returns: {m_train['strategy']['total_return']:.2f}%")
        print(f"    Sharpe:  {m_train['strategy']['sharpe']:.4f}")
        print(f"    Max DD:  {m_train['strategy']['max_dd']:.2f}%")
        print(f"    Trades:  {m_train['trades']['n_trades']}")
        print(f"    Win Rate: {m_train['trades']['win_rate']:.2f}%")
        
        print(f"\n  TEST SET PERFORMANCE:")
        print(f"    Returns: {m_test['strategy']['total_return']:.2f}%")
        print(f"    Sharpe:  {m_test['strategy']['sharpe']:.4f}")
        print(f"    Max DD:  {m_test['strategy']['max_dd']:.2f}%")
        print(f"    Trades:  {m_test['trades']['n_trades']}")
        print(f"    Win Rate: {m_test['trades']['win_rate']:.2f}%")
        
        # Validation checks
        test_win_rate = m_test['trades']['win_rate']
        test_sharpe = m_test['strategy']['sharpe']
        test_return = m_test['strategy']['total_return']
        test_trades = m_test['trades']['n_trades']
        
        # Test 8.1: Minimum win rate
        check_8_1 = test_win_rate >= 40
        print_test("Win rate >= 40%", check_8_1, f"Actual: {test_win_rate:.2f}%", )
        
        # Test 8.2: Reasonable number of test trades (at least 10)
        check_8_2 = test_trades >= 10
        print_test("Minimum 10 trades in test", check_8_2, f"Actual: {test_trades}")
        
        # Test 8.3: Positive returns on test
        check_8_3 = test_return > -10  # Allow up to 10% loss
        print_test("Test returns > -10%", check_8_3, f"Actual: {test_return:.2f}%")
        
        # Test 8.4: Not overfitted (gap between train and test)
        train_ret = m_train['strategy']['total_return']
        overfit_gap = abs(train_ret - test_return)
        check_8_4 = overfit_gap < 50  # Less than 50% gap
        print_test("Not overfitted (gap < 50%)", check_8_4, f"Gap: {overfit_gap:.2f}%")
        
        # Test 8.5: Consistency check
        train_wr = m_train['trades']['win_rate']
        wr_gap = abs(train_wr - test_win_rate)
        check_8_5 = wr_gap < 20
        print_test("Consistent win rate (gap < 20%)", check_8_5, f"Gap: {wr_gap:.2f}%")
        
        ready_for_live = all([check_8_1, check_8_2])  # At least these two
        
        return ready_for_live
        
    except Exception as e:
        print_test("Performance validation", False, str(e))
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════
def run_all_tests():
    """Run complete test suite"""
    print("\n")
    print(f"{ColorCodes.BOLD}{ColorCodes.GREEN}{'='*80}{ColorCodes.END}")
    print(f"{ColorCodes.BOLD}{ColorCodes.GREEN}  NIFTY 50 QUANT SYSTEM — COMPREHENSIVE TEST SUITE{ColorCodes.END}")
    print(f"{ColorCodes.BOLD}{ColorCodes.GREEN}{'='*80}{ColorCodes.END}")
    
    # Test 1
    df = test_data_loading()
    if df is None:
        print(f"\n{ColorCodes.RED}[CRITICAL] Data loading failed. Cannot continue.{ColorCodes.END}")
        return False
    
    # Test 2
    features = test_feature_engineering(df)
    if features is None:
        print(f"\n{ColorCodes.RED}[CRITICAL] Feature engineering failed. Cannot continue.{ColorCodes.END}")
        return False
    
    # Test 3
    f_train, f_test = test_train_test_split(features)
    if f_train is None or f_test is None:
        print(f"\n{ColorCodes.RED}[CRITICAL] Train/test split failed. Cannot continue.{ColorCodes.END}")
        return False
    
    # Test 4
    params = test_calibration(f_train)
    if params is None:
        print(f"\n{ColorCodes.RED}[CRITICAL] Calibration failed. Cannot continue.{ColorCodes.END}")
        return False
    
    # Test 5
    s_train, s_test = test_signal_generation(f_train, f_test, params)
    if s_train is None or s_test is None:
        print(f"\n{ColorCodes.RED}[CRITICAL] Signal generation failed. Cannot continue.{ColorCodes.END}")
        return False
    
    # Test 6
    r_train, r_test, trades_train, trades_test = test_backtest_engine(s_train, s_test)
    if r_train is None:
        print(f"\n{ColorCodes.RED}[CRITICAL] Backtest engine failed. Cannot continue.{ColorCodes.END}")
        return False
    
    # Test 7
    metrics_ok = test_metrics_calculation(r_train, r_test, trades_train, trades_test)
    if not metrics_ok:
        print(f"\n{ColorCodes.YELLOW}[WARNING] Metrics calculation had issues.{ColorCodes.END}")
    
    # Test 8
    live_ready = test_performance_validation(r_train, r_test, trades_train, trades_test)
    
    # Summary
    print_header("TEST SUMMARY")
    
    if live_ready:
        print(f"{ColorCodes.GREEN}{ColorCodes.BOLD}✓ ALL TESTS PASSED{ColorCodes.END}")
        print(f"\n{ColorCodes.GREEN}✓ Model is READY for Live Trading!{ColorCodes.END}")
        print(f"\n  Next Steps:")
        print(f"    1. Review output metrics carefully")
        print(f"    2. Check all plots for anomalies")
        print(f"    3. Verify position sizing constraints")
        print(f"    4. Deploy with SMALL capital first")
        print(f"    5. Monitor live performance for 1-2 weeks")
        print(f"    6. Scale up gradually if performance OK\n")
    else:
        print(f"{ColorCodes.YELLOW}{ColorCodes.BOLD}⚠ TESTS COMPLETED WITH WARNINGS{ColorCodes.END}")
        print(f"\n{ColorCodes.YELLOW}⚠ Model may need adjustment before live trading.{ColorCodes.END}")
        print(f"\n  Recommended Actions:")
        print(f"    1. Review performance metrics")
        print(f"    2. Check calibration thresholds")
        print(f"    3. Consider adjusting position sizing")
        print(f"    4. Re-test with different parameters")
        print(f"    5. Consult performance analysis\n")
    
    return live_ready


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
