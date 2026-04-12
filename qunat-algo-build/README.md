=== NIFTY 50 QUANT MODEL — MASTER PROMPT ===

[SECTION 1: ROLE & OBJECTIVE]
You are an expert quantitative researcher and Python developer. Build a complete, institutional-grade quantitative trading system for Nifty 50 index futures trading using a non-lagging, signal-based approach grounded in statistical and mathematical principles — no lagging indicators (no RSI, MACD, Bollinger Bands, moving averages, or any indicator derived from past price smoothing).

Objective: Design, backtest, and visualize a fully self-contained quant model for the Nifty 50 index using 10 years of daily OHLCV data fetched live from Yahoo Finance.

[SECTION 2: DATA ACQUISITION]
Use yfinance to download Nifty 50 daily OHLCV data:
  Ticker  : "^NSEI"
  Period  : Last 10 years from today's date (use datetime.today() dynamically)
  Interval: "1d" (daily)
  Columns : Open, High, Low, Close, Volume

After downloading:
  - Drop any rows with NaN values
  - Ensure the index is a proper DatetimeIndex
  - Print shape, date range, and first 5 rows as a sanity check
  - Compute daily log returns: log_ret = np.log(Close / Close.shift(1))

Do NOT use any pre-computed or hardcoded data. All data must be fetched live.

[SECTION 3: STRATEGY — NON-LAGGING QUANTITATIVE SIGNALS]
Build a multi-factor signal model using ONLY non-lagging features. Implement all of the following:

A. PRICE ACTION FEATURES (no smoothing)
   - Daily return: (Close - Open) / Open
   - Intraday range: (High - Low) / Open
   - Gap: (Open - prev_Close) / prev_Close
   - Body ratio: |Close - Open| / (High - Low + 1e-9)
   - Upper wick: (High - max(Open,Close)) / (High - Low + 1e-9)
   - Lower wick: (min(Open,Close) - Low) / (High - Low + 1e-9)

B. STATISTICAL / VOLATILITY FEATURES
   - Realized volatility (rolling std of log returns, window=5)
   - Volume Z-score: (Volume - rolling_mean_vol) / rolling_std_vol (window=5)
   - Overnight gap direction: sign of gap
   - Parkinson volatility: sqrt((1/(4*ln(2))) * (ln(H/L))^2)

C. REGIME DETECTION (no MA — use statistical tests)
   - Hurst Exponent (rolling 60-day window) — H > 0.55 = trending, H < 0.45 = mean-reverting
   - Variance ratio test (Lo-MacKinlay) on 5 vs 1 day returns
   - Z-score of close within rolling 20-day range: (Close - rolling_min) / (rolling_max - rolling_min)

D. SIGNAL GENERATION
   - Combine features using a simple logistic scoring rule or threshold-based composite score
   - Signal = +1 (long), -1 (short), 0 (flat/no trade)
   - Apply regime filter: only take trending signals when Hurst > 0.55, mean-reversion signals when Hurst < 0.45
   - Apply volatility filter: skip trades when Parkinson vol is in top 10%
   - Entry on next day's Open, Exit on next day's Close (avoid look-ahead bias)

E. POSITION SIZING
   - Use fixed fractional sizing with Kelly Criterion fraction: f* = (edge / odds)
   - Cap position size at 2x (never more than 2 units)
   - Apply volatility scaling: size = base_size / realized_vol (normalized to 1)

[SECTION 4: BACKTESTING ENGINE]
Build a vectorized backtesting engine from scratch (no Backtrader or Zipline):
  - Initial capital     : INR 10,00,000 (10 lakhs)
  - Transaction costs   : 0.05% per trade (both sides)
  - Slippage            : 0.02% per trade
  - No leverage         : Position value cannot exceed available capital
  - No shorting margin  : Treat short as negative position on index returns

Execution logic:
  - Signal generated on day T using day T's OHLCV data only
  - Trade entered at day T+1 Open
  - Trade exited at day T+1 Close
  - Portfolio value updated daily

Output daily series: portfolio_value, daily_pnl, strategy_returns, benchmark_returns

[SECTION 5: PERFORMANCE METRICS — PRINT ALL]
Compute and print a full performance report (strategy vs benchmark side by side):

RETURN METRICS: Total Return (%), CAGR (%), Best Year (%), Worst Year (%), Average Annual Return (%)

RISK METRICS: Annualized Volatility (%), Maximum Drawdown (%) and Duration (days), Average Drawdown (%), VaR 95% (daily %), CVaR 95%, Downside Deviation

RISK-ADJUSTED METRICS: Sharpe Ratio (risk-free = 6.5% Indian Gsec), Sortino Ratio, Calmar Ratio, Omega Ratio, Information Ratio

TRADE METRICS: Total Trades, Win Rate (%), Avg Win (%), Avg Loss (%), Profit Factor, Expectancy, Avg Holding Period (days), Max Consecutive Wins/Losses

STATISTICAL METRICS: Skewness, Kurtosis, Hurst Exponent of strategy returns, Autocorrelation lag-1

Print all values using tabulate. 4 decimal places for ratios, 2 for percentages, INR formatting for P&L.

[SECTION 6: VISUALIZATIONS — ALL 8 REQUIRED]
Use matplotlib and seaborn. Dark professional theme. Output: "nifty50_quant_report.pdf"

PLOT 1 — EQUITY CURVE: Strategy vs Benchmark normalized to 100, drawdown periods shaded red, rolling Sharpe subplot
PLOT 2 — DRAWDOWN CURVE: Underwater equity as filled area, top-5 drawdowns annotated
PLOT 3 — RETURNS DISTRIBUTION: Histogram + Normal/t-distribution fit + KDE + VaR/CVaR lines
PLOT 4 — TRADE-BY-TRADE P&L: Scatter by date vs P&L %, green/red dots, size = holding duration, cumulative P&L on secondary axis
PLOT 5 — ROLLING METRICS: 4-panel — 63d rolling Sharpe, Sortino, Volatility, Win Rate
PLOT 6 — HEATMAPS: Monthly returns heatmap (years x months) + day-of-week average returns bar chart
PLOT 7 — RISK-RETURN SCATTER: Rolling 252d windows (vol vs return), color by time, iso-Sharpe lines
PLOT 8 — MONTE CARLO: 1000 bootstrap simulations, percentile paths (P5/P25/P50/P75/P95), actual path highlighted, final value histogram

[SECTION 7: CODE REQUIREMENTS]
Libraries: numpy, pandas, scipy, matplotlib, seaborn, yfinance, tabulate
Structure: 8 clearly labeled sections with comments
No look-ahead bias: all features computed only from data available at time T
Reproducible: np.random.seed(42)
Single file: model.py
Robust: handle NaN, zero-volume days, holiday gaps gracefully
Type hints + docstrings on all functions

[SECTION 8: FINAL OUTPUT CHECKLIST]
[1] Console output   : Full metrics table (strategy vs benchmark)
[2] PDF file         : nifty50_quant_report.pdf (all 8 plots)
[3] CSV file         : trades_log.csv (entry, exit, signal, P&L per trade)
[4] CSV file         : equity_curve.csv (daily portfolio, drawdown, rolling Sharpe)

Run end-to-end with: python model.py