# NIFTY 50 QUANT SYSTEM — TEST SUITE & VALIDATION

## Quick Start

### 1. Run Comprehensive Tests
```bash
cd c:\Users\ujjwa\OneDrive\Desktop\Automation-Project\Qunat-Algo-Models\data-spliting-model-train
python test_model.py
```

**What it does:**
- ✅ Validates data integrity
- ✅ Tests all 28 features
- ✅ Verifies train/test split
- ✅ Checks calibration
- ✅ Validates signal generation
- ✅ Tests backtest engine
- ✅ Verifies metrics calculation
- ✅ **Determines if model is ready for live trading**

**Expected Time:** 5-10 minutes

**Success Indicator:**
```
✓ ALL TESTS PASSED
✓ Model is READY for Live Trading!
```

---

### 2. Run Full Backtest with Visualizations
```bash
python main.py
```

**Output includes:**
- 📊 9 interactive plots
- 📈 Detailed metrics table
- 📋 All trades in test period
- 🎯 Monte Carlo simulation
- 📉 Accuracy curves

**Key Metrics to Review:**
- Test Win Rate (should be > 35%)
- Test Sharpe Ratio (should be > -1.0)
- Total Trades in Test (should be > 20)
- Max Drawdown in Test (should be < -30%)

---

## Test Files

### `test_model.py`
Complete test suite with 8 test categories:

| Test | Purpose | Pass Criteria |
|------|---------|---------------|
| TEST 1 | Data Loading | Rows > 1000, no NaN |
| TEST 2 | Features | 28 features calculated correctly |
| TEST 3 | Train/Test Split | 8yr train, 1.3yr test, no overlap |
| TEST 4 | Calibration | Thresholds in valid range |
| TEST 5 | Signals | Short/long signals generated |
| TEST 6 | Backtest | Trades generated, P&L calculated |
| TEST 7 | Metrics | All metrics computed correctly |
| TEST 8 | Validation | Ready for live trading |

### `VALIDATION_CHECKLIST.txt`
Pre-live deployment checklist with:
- ✓ Target metrics for each phase
- ✓ Plot analysis guide
- ✓ Risk assessment
- ✓ Parameter tuning guide
- ✓ Deployment phases
- ✓ Monitoring dashboard
- ✓ Emergency stop conditions

---

## Validation Workflow

### Phase 1: TEST SUITE (5-10 min)
```bash
python test_model.py
# Check all 8 tests pass
```

### Phase 2: BACKTEST ANALYSIS (10-15 min)
```bash
python main.py
# Review:
# - All 9 plots
# - Detailed trade table
# - Metrics summary
# - Accuracy curves
```

### Phase 3: CHECKLIST VERIFICATION (20-30 min)
```
Review VALIDATION_CHECKLIST.txt:
☐ All backtest targets met
☐ Plots show healthy patterns
☐ Risk assessment complete
☐ Deployment strategy defined
```

### Phase 4: PARAMETER TUNING (if needed)
Edit `main.py` config:
```python
KELLY_CAP       = 3.5          # Adjust position sizing
MIN_HOLD        = 2            # Adjust holding period
MAX_HOLD        = 15           # Adjust max hold
TRAIL_STOP      = 0.035        # Adjust stop loss
```
Then re-run test_model.py and main.py

---

## Key Performance Targets

### TRAINING PERIOD (8 Years)
| Metric | Target | Current |
|--------|--------|---------|
| Sharpe Ratio | > 0.3 | ? |
| Total Return | > +50% | ? |
| Max Drawdown | < -40% | ? |
| Win Rate | > 40% | ? |
| Total Trades | > 100 | ? |

### TEST PERIOD (1.3 Years - Out-of-Sample)
| Metric | Target | Current |
|--------|--------|---------|
| Sharpe Ratio | > -1.0 | ? |
| Total Return | > -20% | ? |
| Max Drawdown | < -30% | ? |
| Win Rate | > 35% | ? |
| Total Trades | > 20 | ? |

---

## Common Issues & Solutions

### ❌ Too Many Losses (Win Rate < 35%)
**Solution:**
- Reduce KELLY_CAP to 2.5
- Raise long threshold (fewer trades)
- Reduce breadth multipliers

### ❌ Too Few Trades (< 15 in test)
**Solution:**
- Lower signal thresholds
- Reduce MIN_HOLD to 1
- Increase exposure targets

### ❌ High Drawdown (> 40%)
**Solution:**
- Reduce TRAIL_STOP to 2.5%
- Lower MAX_HOLD to 10
- Cap position sizing

### ❌ Overfitting (Train >> Test performance)
**Solution:**
- Remove shortest-term features
- Increase MIN_HOLD
- Use simpler signal logic

---

## Deployment Phases

### Phase 1: Paper Trading (Weeks 1-2)
- Run without real money
- Verify signals
- Check slippage assumptions

### Phase 2: Micro Trading (Weeks 3-4)
- 5% of capital
- 0.5x position sizing
- Daily monitoring

### Phase 3: Ramp Up (Weeks 5-8)
- 25% of capital
- 1.0-1.5x position sizing
- Weekly review

### Phase 4: Production (Week 9+)
- Full capital
- Full position sizing
- Monthly reviews

---

## Emergency Stop Conditions

🚨 **HALT if:**
- Daily loss > 2% of account
- Drawdown > 25% from peak
- Win rate < 30% (5-trade window)
- Losing > 3 consecutive trades
- Sharpe ratio negative for 3+ days
- Position violations (2x+ expected)
- Liquidity issues

---

## Files Structure

```
Qunat-Algo-Models/
├── data-spliting-model-train/
│   ├── main.py                    # Main backtest model
│   ├── test_model.py              # ⭐ Test suite (RUN THIS FIRST)
│   ├── qunat_nodel_test.ipynb     # Jupyter notebook
│   └── README.md                  # Model documentation
├── VALIDATION_CHECKLIST.txt       # ⭐ Pre-live checklist
└── [other models]
```

---

## Quick Reference Commands

```bash
# Navigate to folder
cd c:\Users\ujjwa\OneDrive\Desktop\Automation-Project\Qunat-Algo-Models\data-spliting-model-train

# Run tests
python test_model.py

# Run backtest
python main.py

# Check syntax only
python -m py_compile main.py
```

---

## Success Criteria for Go-Live

✅ Pass ALL test_model.py tests
✅ Training Sharpe > 0.3
✅ Test Sharpe > -1.0
✅ Test Win Rate > 35%
✅ Test Trades > 20
✅ Plots show healthy patterns
✅ Risk assessment complete
✅ Deployment plan ready

---

## Support

**Issues with tests?**
1. Check test output messages
2. Review VALIDATION_CHECKLIST.txt
3. Adjust parameters in main.py
4. Re-run test_model.py

**Ready to go live?**
1. Complete all phases
2. Monitor Phase 1 closely
3. Scale gradually
4. Review monthly

---

## Contact & Updates

For questions or updates, refer to:
- Model documentation: README.md
- Test documentation: test_model.py docstrings
- Validation guide: VALIDATION_CHECKLIST.txt

**Good luck with your live trading! 💰**
