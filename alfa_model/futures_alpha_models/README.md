You are a professional quantitative trading engineer.

I have an existing NSE futures (FUTSTK) trading model that is:

* Profitable but weak
* Low Sharpe (~0.2–0.4)
* High transaction cost impact (up to 50% of gross PnL)
* Short side is losing money
* Too many trades with low edge

Your task is to **FIX and REDESIGN** this model into an **aggressive but profitable futures trading system**.

==============================
🎯 OBJECTIVE
============

Improve the strategy to achieve:

* Higher Sharpe ratio (> 0.8)
* Profit Factor > 1.3
* Lower cost-to-gross (< 30%)
* Fewer but higher-quality trades
* Stronger edge per trade

==============================
⚠️ EXISTING PROBLEMS (MUST FIX)
===============================

1. Weak signal (mean reversion only)
2. Too many trades → high costs
3. Short positions losing money
4. Uniform position sizing (no conviction scaling)
5. No cost-aware filtering
6. Strategy trades in low-quality setups

==============================
🔧 REQUIRED IMPROVEMENTS
========================

1. SIGNAL STRENGTH

* Increase z-score threshold (e.g. 1.5–2.5)
* Use composite signals (1d + 3d returns)
* Only trade strong deviations

2. HYBRID STRATEGY
   Combine:

* Mean reversion (short-term)
* Momentum filter (trend confirmation)

Example logic:

* LONG → oversold + uptrend
* SHORT → overbought + downtrend

3. SHORT SIDE FIX

* Disable shorts in uptrend regimes OR
* Only allow shorts in strong downtrend

4. DYNAMIC POSITION SIZING (IMPORTANT)

* Scale position size based on signal strength

Example:

* Strong signal → higher risk (3–5%)
* Medium signal → lower risk (1–2%)
* Weak signal → skip

5. TRADE FILTERING

* Skip trades where expected return < transaction cost
* Avoid high volatility regimes

6. REDUCE OVERTRADING

* Increase holding period (5 → 7–10 days)
* Increase cooldown
* Limit number of trades

7. FUTURES REALISM

* Keep:

  * lot size
  * margin system
  * transaction costs
* Do NOT revert to spot logic

==============================
📊 BACKTEST STRUCTURE
=====================

* Training: 2010–2018
* Validation: 2019–2021
* Execution: 2022–2024

NO data leakage.
Parameters must be locked after training.

==============================
📉 OUTPUT REQUIREMENT
=====================

ONLY print clean performance metrics:

---

## PERFORMANCE SUMMARY

Initial Capital
Final Capital
Total Return (%)
CAGR
Sharpe Ratio
Max Drawdown
Win Rate
Total Trades
Profit Factor
Expectancy
Total Costs
Cost-to-Gross (%)
-----------------

NO:

* charts
* trade logs
* debug prints

==============================
🎯 GOAL
=======

Transform the model from:
"high-frequency low-edge system"

into:
"low-frequency high-conviction futures trading system"

Focus on **quality over quantity** and **real profitability after costs**.
