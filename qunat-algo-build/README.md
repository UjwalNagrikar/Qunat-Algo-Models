You are an expert quantitative researcher and Python developer. Build a complete, institutional-grade quantitative trading system for Nifty 50 index futures trading using a non-lagging, signal-based approach grounded in statistical and mathematical principles — no lagging indicators (no RSI, MACD, Bollinger Bands, moving averages, or any indicator derived from past price smoothing).

Objective: Design, backtest, and visualize a fully self-contained quant model for the Nifty 50 index using 10 years of daily OHLCV data fetched live from Yahoo Finance.

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

Build a multi-factor signal model using ONLY non-lagging features. Implement all of the following:

  A. PRICE ACTION FEATURES (no smoothing)
     ─ Daily return: (Close - Open) / Open
     ─ Intraday range: (High - Low) / Open
     ─ Gap: (Open - prev_Close) / prev_Close
     ─ Body ratio: |Close - Open| / (High - Low + 1e-9)
     ─ Upper wick: (High - max(Open,Close)) / (High - Low + 1e-9)
     ─ Lower wick: (min(Open,Close) - Low) / (High - Low + 1e-9)

  B. STATISTICAL / VOLATILITY FEATURES
     ─ Realized volatility (rolling std of log returns, window=5 — this is NOT a lagging indicator, it is a statistical measure)
     ─ Volume Z-score: (Volume - rolling_mean_vol) / rolling_std_vol (window=5)
     ─ Overnight gap direction: sign of gap
     ─ Parkinson volatility: sqrt((1/(4*ln(2))) * (ln(H/L))^2)

  C. REGIME DETECTION (no MA — use statistical tests)
     ─ Hurst Exponent (rolling 60-day window) — H > 0.55 = trending, H < 0.45 = mean-reverting
     ─ Variance ratio test (Lo-MacKinlay) on 5 vs 1 day returns
     ─ Z-score of close within rolling 20-day range: (Close - rolling_min) / (rolling_max - rolling_min)

  D. SIGNAL GENERATION
     ─ Combine features using a simple logistic scoring rule or threshold-based composite score
     ─ Signal = +1 (long), -1 (short), 0 (flat/no trade)
     ─ Apply regime filter: only take trending signals when Hurst > 0.55, mean-reversion signals when Hurst < 0.45
     ─ Apply volatility filter: skip trades when Parkinson vol is in top 10% (extreme volatility)
     ─ Entry on next day's Open, Exit on next day's Close (avoid look-ahead bias)

  E. POSITION SIZING
     ─ Use fixed fractional sizing with Kelly Criterion fraction: f* = (edge / odds)
     ─ Cap position size at 2x (never more than 2 units)
     ─ Apply volatility scaling: size = base_size / realized_vol (normalized to 1)

     Build a vectorized backtesting engine from scratch (no Backtrader or Zipline):

  - Initial capital     : INR 10,00,000 (10 lakhs)
  - Transaction costs   : 0.05% per trade (both sides), applied to each entry and exit
  - Slippage            : 0.02% per trade (realistic for Nifty futures)
  - No leverage         : Position value cannot exceed available capital
  - No shorting margin  : Treat short as negative position on index returns

  Execution logic:
    - Signal generated on day T using day T's OHLCV data only
    - Trade entered at day T+1 Open
    - Trade exited at day T+1 Close
    - Portfolio value updated daily

  Output daily series:
    - portfolio_value   : running equity curve
    - daily_pnl         : daily profit/loss in INR
    - strategy_returns  : daily percentage returns of the strategy
    - benchmark_returns : buy-and-hold Nifty 50 (same period)

Compute and print a full performance report with the following metrics (strategy vs benchmark side by side):

  RETURN METRICS
  ────────────────────────────────────────────────
  Total Return (%)
  CAGR — Compound Annual Growth Rate (%)
  Best Year (%), Worst Year (%)
  Average Annual Return (%)

  RISK METRICS
  ────────────────────────────────────────────────
  Annualized Volatility (%)
  Maximum Drawdown (%) and Duration (days)
  Average Drawdown (%)
  Value at Risk — VaR 95% (daily %)
  Conditional VaR — CVaR / Expected Shortfall (95%)
  Downside Deviation (semi-deviation)

  RISK-ADJUSTED METRICS
  ────────────────────────────────────────────────
  Sharpe Ratio (annualized, risk-free = 6.5% — Indian 10Y Gsec)
  Sortino Ratio
  Calmar Ratio (CAGR / Max Drawdown)
  Omega Ratio (threshold = 0)
  Information Ratio (vs benchmark)

  TRADE METRICS
  ────────────────────────────────────────────────
  Total Number of Trades
  Win Rate (%)
  Average Win (%), Average Loss (%)
  Profit Factor (gross profit / gross loss)
  Expectancy (avg P&L per trade)
  Average Holding Period (days)
  Max Consecutive Wins / Max Consecutive Losses

  STATISTICAL METRICS
  ────────────────────────────────────────────────
  Skewness of returns
  Kurtosis of returns
  Hurst Exponent of strategy returns
  Autocorrelation (lag 1) of returns

  Print all values formatted neatly using tabulate or a print table. Use 4 decimal places for ratios, 2 for percentages, and INR formatting for P&L values.


  Use matplotlib and seaborn. Use a dark professional theme (plt.style.use('seaborn-v0_8-darkgrid') or similar). All plots in a single PDF output file "nifty50_quant_report.pdf" AND displayed inline.

PLOT 1 — EQUITY CURVE
  - Strategy portfolio value vs Benchmark buy-and-hold (both normalized to 100)
  - Annotate all drawdown periods as shaded red zones
  - Add 252-day rolling Sharpe in a subplot below

PLOT 2 — DRAWDOWN CURVE
  - Rolling underwater equity (drawdown from peak) as a filled area chart
  - Highlight top-5 worst drawdowns with annotations (date + depth)
  - Color: red fill with 40% alpha

PLOT 3 — RETURNS DISTRIBUTION
  - Histogram of daily strategy returns overlaid with Normal and t-distribution fit
  - Add vertical lines for VaR 95% and CVaR 95%
  - Add KDE curve
  - Annotate skewness and kurtosis on the chart

PLOT 4 — TRADE-BY-TRADE P&L (Execution Quality)
  - Scatter plot: each trade as a dot (x = trade entry date, y = trade P&L in %)
  - Color: green for wins, red for losses
  - Size of dot proportional to holding duration
  - Add cumulative P&L as a line overlaid on secondary axis

PLOT 5 — ROLLING METRICS (Stability Over Time)
  - 4-panel subplot:
      (a) 63-day rolling Sharpe Ratio
      (b) 63-day rolling Sortino Ratio
      (c) 63-day rolling Volatility (annualized)
      (d) 63-day rolling Win Rate
  - Add horizontal reference lines (e.g., Sharpe = 1.0)

PLOT 6 — HEATMAPS (When Strategy Works)
  - Heatmap 1: Monthly returns (rows = years, cols = months) — color: green/red diverging
  - Heatmap 2: Day-of-week average returns — bar chart showing which weekday the strategy performs best
  - Use seaborn heatmap with annotations showing exact return %

PLOT 7 — RISK-RETURN SCATTER
  - Scatter plot of rolling 252-day windows: x = annualized vol, y = annualized return
  - Color gradient by time (early = blue, recent = orange)
  - Add benchmark point and global strategy point as special markers
  - Add iso-Sharpe lines (Sharpe = 0.5, 1.0, 1.5) as reference

PLOT 8 — MONTE CARLO SIMULATION
  - Run 1000 simulations by randomly resampling daily strategy returns (bootstrap with replacement)
  - Plot all simulated equity curves with low alpha (0.05)
  - Highlight P5, P25, P50, P75, P95 percentile paths in bold
  - Highlight actual strategy path in white/yellow
  - Print: P5 final portfolio value, P50 final value, P95 final value
  - Add a histogram of final portfolio values across all simulations

  Code must follow these standards:

  LIBRARIES   : numpy, pandas, scipy, matplotlib, seaborn, yfinance, tabulate, sklearn (only for preprocessing, not strategy signals)
  STRUCTURE   : Organize into clearly labeled sections with comments:
                  1. Imports & Config
                  2. Data Download
                  3. Feature Engineering
                  4. Signal Generation
                  5. Backtesting Engine
                  6. Performance Metrics
                  7. Visualizations
                  8. Monte Carlo
  NO LOOK-AHEAD BIAS : All features must be computed using only data available at time T to generate signal for T+1
  REPRODUCIBLE       : Set random seed np.random.seed(42)
  SINGLE FILE        : Entire model in one Python script (model.py)
  ROBUST             : Handle missing data, zero-volume days, and index holiday gaps gracefully
  TYPE HINTS         : Use Python type hints on all functions
  DOCSTRINGS         : Add a one-line docstring to every function

At the end of execution, the script must produce:

  [1] Console output   : Full performance metrics table (strategy vs benchmark, side by side)
  [2] PDF file         : "nifty50_quant_report.pdf" containing all 8 plots
  [3] CSV file         : "trades_log.csv" — every trade with entry date, exit date, signal,
                          entry price, exit price, P&L %, P&L INR, holding days
  [4] CSV file         : "equity_curve.csv" — daily portfolio value, benchmark value,
                          drawdown, rolling Sharpe

  If any plot or metric fails to compute, print a WARNING and skip — do not crash the script.
  Run the full pipeline end-to-end with a single command: python model.py