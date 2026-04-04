# Build a professional multi-stock quantitative trading backtest system in Python using daily timeframe data.

# The system should implement a cross-sectional mean reversion alpha strategy across multiple NSE stocks.

# CORE REQUIREMENTS:

# 1. DATA:

# * Use yfinance to download daily OHLCV data
# * Universe: 15–30 NSE stocks (large-cap preferred)
# * Time period: at least 1–2 years
# * Combine all stocks into a single DataFrame with columns: date, symbol, open, high, low, close, volume

# 2. FEATURE ENGINEERING:

# * Compute daily returns (1-day, 3-day, 5-day)
# * Compute rolling volatility (std of returns)
# * Do NOT use technical indicators (no RSI, MACD, Bollinger)

# 3. ALPHA MODEL (CROSS-SECTIONAL MEAN REVERSION):

# * For each day:

#   * Rank all stocks based on 1-day return
#   * Identify:

#     * Bottom 20% (worst performers) → LONG
#     * Top 20% (best performers) → SHORT
# * This should be done using groupby(date)

# 4. ENTRY RULE:

# * Enter trades at next day open price
# * Allow multiple positions simultaneously across different stocks

# 5. EXIT RULE:

# * Exit after fixed holding period (2–5 days)
# * OR exit when return reverts toward mean

# 6. POSITION SIZING:

# * Equal capital allocation per trade
# * Optional: risk-based sizing (max 2% capital per trade)
# * Limit max concurrent positions (e.g., 5–10)

# 7. COST MODEL:

# * Include realistic trading costs:

#   * brokerage
#   * STT
#   * exchange charges
#   * slippage (approximate)

# 8. PORTFOLIO MANAGEMENT:

# * Maintain a single portfolio equity curve
# * Track all open trades across stocks
# * Update capital after each trade

# 9. METRICS (MUST MATCH PROFESSIONAL LEVEL):

# * Total return (% and ₹)
# * Max drawdown
# * Sharpe ratio
# * Win rate
# * Profit factor
# * Expectancy per trade
# * Number of trades (target: 400–500)
# * Average holding period
# * Consecutive wins/losses

# 10. OUTPUT:

# * Print all trades in tabular format
# * Generate equity curve
# * Generate drawdown chart
# * Generate per-trade PnL distribution
# * Show monthly returns

# 11. STRUCTURE:

# * Modular functions:

#   * load_data()
#   * build_features()
#   * generate_signals()
#   * run_backtest()
#   * compute_metrics()
#   * plot_results()

# 12. EXTRA (IMPORTANT):

# * Add volatility filter (skip trades when volatility is too high)
# * Add cooldown period per stock (avoid immediate re-entry)
# * Ensure no lookahead bias

# GOAL:

# * Generate 400–500 trades over the test period
# * Build a clean, scalable, hedge-fund-style alpha model
# * Code should be efficient, readable, and production-ready
