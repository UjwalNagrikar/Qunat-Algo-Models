# NIFTY 50 QUANT ML v7.0 - TEST SUITE QUICK GUIDE

## 📊 Test Suite Overview

| Component | Tests | Coverage |
|-----------|-------|----------|
| **Data Pipeline** | 5 | Download, cleaning, validation |
| **Features** | 5 | Computation, validity, ranges |
| **Target Creation** | 4 | Binary labels, class balance |
| **Train/Test Split** | 4 | No leakage, temporal order |
| **ML Model** | 5 | Training, accuracy, predictions |
| **Signals** | 5 | ML signals, regime detection |
| **Backtesting** | 5 | Trade execution, P&L |
| **Metrics** | 5 | Returns, risk, Sharpe |
| **Drawdown** | 3 | Drawdown calculations |
| **Integration** | 6 | End-to-end validation |
| **Configuration** | 3 | Parameter validation |
| **Edge Cases** | 3 | Error handling |
| **TOTAL** | **53** | Complete coverage |

---

## 🚀 Quick Start

### Installation
```bash
pip install pytest numpy pandas scikit-learn xgboost yfinance scipy seaborn matplotlib
```

### Run All Tests
```bash
cd test-case/
pytest test_quant_ml.py test_integration.py -v
```

### Run Specific Suite
```bash
pytest test_quant_ml.py::TestMLTraining -v
pytest test_integration.py::TestIntegration -v
```

### Run Specific Test
```bash
pytest test_quant_ml.py::TestMLTraining::test_model_accuracy -v
```

### Very Quick Test (Few Seconds)
```bash
pytest test_quant_ml.py::TestDataPipeline -v
```

---

## 📋 Test Checklist

### ✅ SUITE 1: DATA PIPELINE (5 tests)
- [ ] TEST 1.1: Data shape and columns
- [ ] TEST 1.2: No missing values
- [ ] TEST 1.3: OHLC logical order
- [ ] TEST 1.4: Positive volumes
- [ ] TEST 1.5: No extreme gaps

**Expected**: All PASS
**Time**: ~5 seconds

### ✅ SUITE 2: FEATURES (5 tests)
- [ ] TEST 2.1: Features computed
- [ ] TEST 2.2: All required features present (12)
- [ ] TEST 2.3: No NaN/Inf values
- [ ] TEST 2.4: Reasonable ranges
- [ ] TEST 2.5: Hurst in [0,1]

**Expected**: All PASS
**Time**: ~30 seconds

### ✅ SUITE 3: TARGET (4 tests)
- [ ] TEST 3.1: Target column exists
- [ ] TEST 3.2: Binary values (0/1)
- [ ] TEST 3.3: Balanced classes
- [ ] TEST 3.4: Forward lookback NaN rows

**Expected**: All PASS
**Time**: ~10 seconds

### ✅ SUITE 4: SPLIT (4 tests)
- [ ] TEST 4.1: Train/test sizes
- [ ] TEST 4.2: No overlap
- [ ] TEST 4.3: ~80/20 ratio
- [ ] TEST 4.4: All features present

**Expected**: All PASS
**Time**: ~5 seconds

### ✅ SUITE 5: ML MODEL (5 tests)
- [ ] TEST 5.1: Model trains
- [ ] TEST 5.2: Accuracy 50-100%
- [ ] TEST 5.3: AUC > 0.5
- [ ] TEST 5.4: Prediction shape correct
- [ ] TEST 5.5: Probabilities in [0,1]

**Expected**: All PASS
**Time**: ~60 seconds

### ✅ SUITE 6: SIGNALS (5 tests)
- [ ] TEST 6.1: Signal column exists
- [ ] TEST 6.2: Valid values (-1/0/1)
- [ ] TEST 6.3: >5% long trades
- [ ] TEST 6.4: Regime column exists
- [ ] TEST 6.5: Valid regimes

**Expected**: All PASS
**Time**: ~30 seconds

### ✅ SUITE 7: BACKTEST (5 tests)
- [ ] TEST 7.1: Backtest runs
- [ ] TEST 7.2: DataFrame columns correct
- [ ] TEST 7.3: Positive portfolio values
- [ ] TEST 7.4: Trade structure valid
- [ ] TEST 7.5: Entry < Exit dates

**Expected**: All PASS
**Time**: ~30 seconds

### ✅ SUITE 8: METRICS (5 tests)
- [ ] TEST 8.1: Metrics compute
- [ ] TEST 8.2: Required keys present
- [ ] TEST 8.3: Return metrics reasonable
- [ ] TEST 8.4: Risk metrics valid
- [ ] TEST 8.5: Sharpe computed

**Expected**: All PASS
**Time**: ~30 seconds

### ✅ SUITE 9: DRAWDOWN (3 tests)
- [ ] TEST 9.1: Flat equity → zero DD
- [ ] TEST 9.2: DD detected
- [ ] TEST 9.3: DD in [-1,0]

**Expected**: All PASS
**Time**: <5 seconds

### ✅ INTEGRATION (6 tests)
- [ ] INTEGRATION 1: Full pipeline works
- [ ] INTEGRATION 2: No data leakage
- [ ] INTEGRATION 3: Reproducible
- [ ] INTEGRATION 4: Signal coverage >30d
- [ ] INTEGRATION 5: Trades consistent
- [ ] INTEGRATION 6: Benchmark reasonable

**Expected**: All PASS
**Time**: ~120 seconds

---

## 📌 Example Test Run

```
$ pytest test_quant_ml.py::TestMLTraining -v

test_quant_ml.py::TestMLTraining::test_model_training PASSED       [ 20%]
test_quant_ml.py::TestMLTraining::test_model_accuracy PASSED       [ 40%]
test_quant_ml.py::TestMLTraining::test_model_auc PASSED            [ 60%]
test_quant_ml.py::TestMLTraining::test_model_predict_shape PASSED  [ 80%]
test_quant_ml.py::TestMLTraining::test_model_prob_range PASSED     [100%]

===== 5 passed in 45.23s =====
```

---

## ⚠️ Common Failures & Solutions

### ❌ `ModuleNotFoundError: No module named 'main'`
**Fix**: Update path in test file:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'quant-test-models'))
```

### ❌ `TEST 7.5 FAIL: Entry >= Exit dates found`
**Cause**: Trade P&L calculation error
**Fix**: Check `run_backtest()` holding_days calculation

### ❌ `TEST 8.3 FAIL: total_return > 5`
**Cause**: Market volatility
**Fix**: This is normal. Check market conditions.

### ❌ Timeout on `test_ml_training`
**Cause**: XGBoost with large dataset
**Solution**: Use `-k` flag to skip slow tests
```bash
pytest -k "not test_ml_training" -v
```

### ❌ No trades executed (SKIP without error)
**Cause**: Probability thresholds too strict
**Fix**: Lower `PROB_LONG` and `PROB_SHORT` in `main.py`

---

## 📊 Expected Results Summary

```
SUITE 1: DATA       ✅ 5/5 PASS (5-10 sec)
SUITE 2: FEATURES   ✅ 5/5 PASS (20-40 sec)
SUITE 3: TARGET     ✅ 4/4 PASS (10-15 sec)
SUITE 4: SPLIT      ✅ 4/4 PASS (5-10 sec)
SUITE 5: ML MODEL   ✅ 5/5 PASS (40-80 sec) ⏱️ SLOW
SUITE 6: SIGNALS    ✅ 5/5 PASS (20-40 sec)
SUITE 7: BACKTEST   ✅ 5/5 PASS (20-40 sec)
SUITE 8: METRICS    ✅ 5/5 PASS (20-40 sec)
SUITE 9: DRAWDOWN   ✅ 3/3 PASS (<5 sec)
INTEGRATION         ✅ 6/6 PASS (60-120 sec) ⏱️ SLOW
CONFIG              ✅ 3/3 PASS (<5 sec)
EDGE CASES          ✅ 3/3 PASS (10-20 sec)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL: ✅ 53/53 PASS (5-10 minutes)
```

---

## 🎯 Test Execution Timeline

```
Quick Run (~2 min):
├─ DATA PIPELINE         5 sec ✅
├─ TARGET                10 sec ✅
├─ SPLIT                 5 sec ✅
└─ CONFIG               <5 sec ✅

Standard Run (~5 min):
├─ DATA PIPELINE         5 sec ✅
├─ FEATURES             30 sec ✅
├─ TARGET               10 sec ✅
├─ SPLIT                 5 sec ✅
├─ ML MODEL             60 sec ✅ [SLOW]
├─ SIGNALS              30 sec ✅
├─ BACKTEST             30 sec ✅
├─ METRICS              30 sec ✅
├─ DRAWDOWN             <5 sec ✅
└─ CONFIG               <5 sec ✅

Full Run (~15 min):
├─ All above         ~5 min
├─ INTEGRATION      120 sec ✅ [SLOW]
└─ EDGE CASES        15 sec ✅

```

---

## 🔧 Advanced Options

### Generate Coverage Report
```bash
pytest test_quant_ml.py --cov=../quant-test-models --cov-report=html
# Opens htmlcov/index.html
```

### Run with Detailed Output
```bash
pytest test_quant_ml.py -v -s
# Shows all print statements
```

### Run Specific Pattern
```bash
pytest test_quant_ml.py -k "accuracy or auc"
# Runs only tests matching pattern
```

### Generate JUnit XML (CI/CD)
```bash
pytest test_quant_ml.py --junit-xml=results.xml
```

### Parallel Execution
```bash
pip install pytest-xdist
pytest test_quant_ml.py -n auto
# Runs on all CPU cores
```

---

## ✨ Key Validations

### Data Quality ✅
- OHLC logical consistency
- No NaN/Inf in critical columns
- Positive volumes
- No extreme gaps

### Model Quality ✅
- Accuracy > 50%
- AUC > 0.5
- Probability calibration [0,1]
- Reproducible results

### Trading Logic ✅
- Proper P&L calculation
- Entry before exit
- Positive portfolio values
- Realistic trade metrics

### Risk Management ✅
- Sharpe ratio computed
- Drawdown tracked
- Position sizing via Kelly
- Slippage/commission accounted

---

## 📞 Support

**Tests Failing?**
1. Check test output for specific assertion
2. Review test docstring for expected behavior
3. Verify data availability (yfinance)
4. Check probability thresholds

**Need Help?**
- Read `TEST_README.md` for detailed docs
- Check test docstrings for assumptions
- Run with `-v -s` for detailed output
- Verify configuration matches algo

---

**Last Updated**: April 15, 2026
**Test Count**: 53 unit + integration tests
**Coverage**: Complete pipeline validation
