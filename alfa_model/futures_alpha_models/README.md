#!/usr/bin/env python3
"""
================================================================================
  NSE FUTSTK BACKTESTING ENGINE  v3.0
  Strategy   : Cross-Sectional Mean Reversion  |  Futures (FUTSTK) Mechanics
  Universe   : 12 NSE Large-Cap Futures Contracts  |  2010 – 2024
  Walk-Forward: Training 2010-18  |  Validation 2019-21  |  Execution 2022-24

  FUTURES MECHANICS IMPLEMENTED
  ─────────────────────────────
  • Predefined lot sizes per symbol (integer lots only)
  • SPAN + Exposure margin (20%)
  • Risk-based sizing in number of lots (1–2% capital per trade)
  • PnL = (Exit – Entry) × Lot Size × Num Lots × Direction
  • Full NSE futures cost model (brokerage, STT, exchange, SEBI, GST, slippage)
  • No spot / equity share logic anywhere

  SIGNAL ARCHITECTURE
  ───────────────────
  • Composite signal = average of 1-day and 3-day cross-sectional z-scores
  • |z_composite| > z_threshold to trade (default 1.5σ)
  • 20-day MA trend filter (long only above MA, short only below MA)
  • 10-day rolling vol filter (skip high-vol stocks)

  OUTPUT: Clean performance table only — no plots, no trade logs
================================================================================
"""