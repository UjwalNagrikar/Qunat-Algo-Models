
# ── Data ─────────────────────────────────────────────────────────────────────
TICKER          = "^NSEI"          # Yahoo Finance proxy (replace with real futures data)
START_DATE      = "2010-01-01"
END_DATE        = "2024-12-31"

# Walk-Forward Splits
TRAIN_END       = "2018-12-31"
VAL_END         = "2021-12-31"
# Test = 2022-01-01 to END_DATE

# ── Futures Contract ──────────────────────────────────────────────────────────
LOT_SIZE        = 75               # Nifty 50 lot size (verify current)
SPAN_MARGIN_PCT = 0.065            # ~6.5% of contract value (approx)

# ── Feature Engineering ───────────────────────────────────────────────────────
HURST_WINDOW        = 60           # Days for Hurst exponent calculation
BREAKOUT_WINDOW     = 20           # Rolling high/low lookback
ZSCORE_WINDOW       = 20           # Rolling z-score window
VOL_WINDOW          = 20           # Realized volatility window
VOLUME_ZSCORE_WIN   = 20           # Volume z-score window
RETURN_WINDOW_SHORT = 3            # Short-term return window (days)
RETURN_WINDOW_MED   = 5            # Medium-term return window (days)

# ── Regime Filter ─────────────────────────────────────────────────────────────
HURST_TREND_THRESH      = 0.55     # H > 0.55 → Trending regime
HURST_MR_THRESH         = 0.45     # H < 0.45 → Mean reversion regime
# Between 0.45–0.55 = No trade zone

HIGH_VOL_MULTIPLIER     = 2.0      # Skip if realized vol > 2x historical avg

# ── Entry Signal — Trending Regime ────────────────────────────────────────────
TREND_BREAKOUT_WINDOW   = 20       # Breakout of N-day high/low
TREND_VOL_ZSCORE_MIN    = 0.8      # Minimum volume z-score for breakout confirm
TREND_RETURN_ZSCORE_MIN = 0.5      # Minimum 5-day return z-score for momentum

# ── Entry Signal — Mean Reversion Regime ─────────────────────────────────────
MR_PRICE_ZSCORE_ENTRY   = 2.0      # Enter when price z-score crosses this
MR_VOL_ZSCORE_MAX       = 0.8      # Low volume confirms reversion (not panic)
MR_RETURN_3D_THRESH     = 0.03     # 3-day return must be > 3% for short entry

# ── Exit Rules ────────────────────────────────────────────────────────────────
STOP_VOL_MULTIPLIER     = 1.5      # Stop = entry ± (1.5 × realized_vol × price)
TARGET_VOL_MULTIPLIER   = 3.0      # Target = entry ± (3.0 × realized_vol × price)
TIME_STOP_DAYS          = 7        # Exit if trade not working after 7 days

# ── Risk Management ───────────────────────────────────────────────────────────
CAPITAL             = 1000000    # ₹10 Lakhs starting capital
RISK_PER_TRADE_PCT  = 0.01         # Risk 1% of capital per trade
MAX_OPEN_TRADES     = 1            # Only 1 position at a time (futures)

# ── Cost Model (NSE) ─────────────────────────────────────────────────────────
BROKERAGE_PER_LOT   = 40.0         # ₹20 each side (flat fee broker like Zerodha)
STT_PCT             = 0.0001       # 0.01% on sell side (futures)
EXCHANGE_PCT        = 0.00002      # NSE transaction charge
SEBI_PCT            = 0.000001     # SEBI turnover fee
STAMP_DUTY_PCT      = 0.00002      # Stamp duty on buy
SLIPPAGE_POINTS     = 5.0          # Assumed slippage in Nifty points per trade
