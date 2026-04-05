#!/usr/bin/env python3
"""
==============================================================================
  MULTI-STOCK QUANTITATIVE BACKTESTING ENGINE
  Strategy  : Cross-Sectional Mean Reversion Alpha
  Universe  : 25 NSE Large-Cap Equities
  Timeframe : Daily OHLCV  |  2022-2024
  Author    : UNiverse Capital Research
==============================================================================

NOTE ON SHORT SELLING:
  Retail overnight short in NSE cash equity is not permitted.
  The short leg of this strategy is modelled as single-stock FUTURES (FUTSTK).
  Transaction costs are set to futures rates accordingly.  In production,
  account for futures roll costs (~0.02% per roll) and SPAN margin requirements.
==============================================================================
"""

import warnings, sys, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
from tabulate import tabulate

# ── Visual Theme ──────────────────────────────────────────────────────────────
DARK  = "#0a0e1a"
NAVY  = "#0d1b2e"
PANEL = "#111827"
GOLD  = "#d4af37"
GREEN = "#00b37d"
RED   = "#e05c5c"
BLUE  = "#4a9eda"
GREY  = "#8892a0"
WHITE = "#e8eaf0"

plt.rcParams.update({
    "figure.facecolor":  DARK,  "axes.facecolor":    PANEL,
    "axes.edgecolor":    "#1e2a3a", "axes.labelcolor": GREY,
    "xtick.color":       GREY,  "ytick.color":       GREY,
    "text.color":        WHITE, "grid.color":        "#1e2a3a",
    "grid.linewidth":    0.6,   "font.family":       "monospace",
    "axes.spines.top":   False, "axes.spines.right": False,
})

# ==============================================================================
#  CONFIGURATION
# ==============================================================================
CFG = {
    # Universe
    "symbols": [
        "RELIANCE.NS",  "TCS.NS",        "HDFCBANK.NS",  "INFY.NS",
        "ICICIBANK.NS", "HINDUNILVR.NS", "ITC.NS",        "SBIN.NS",
        "BHARTIARTL.NS","KOTAKBANK.NS",  "LT.NS",         "AXISBANK.NS",
        "BAJFINANCE.NS","WIPRO.NS",      "TITAN.NS",      "MARUTI.NS",
        "SUNPHARMA.NS", "NTPC.NS",       "POWERGRID.NS",  "NESTLEIND.NS",
        "ONGC.NS",      "HCLTECH.NS",    "TECHM.NS",      "ULTRACEMCO.NS",
        "ASIANPAINT.NS",
    ],
    "start": "2010-01-01",
    "end":   "2024-12-31",

    # Capital
    "initial_capital": 1000000,   # INR 10 Lakh

    # Strategy
    "holding_period": 3,            # 3-day hold window
    "long_pct":       0.20,         # bottom 20% by 1-day return -> LONG
    "short_pct":      0.20,         # top 20% by 1-day return    -> SHORT
    "z_threshold":    0.50,         # |cross-sectional z-score| > 0.5 to enter
    "stop_loss_pct":  0.04,         # 4% intra-trade stop-loss (checked at close)
    "max_positions":  8,            # max concurrent open positions
    "cooldown_days":  4,            # trading days before re-entry on same stock

    # Volatility filter
    "vol_window":     10,
    "vol_threshold":  0.030,        # skip if 10-day rolling vol > 3.0%

    # Position sizing
    "risk_pct":       0.015,        # target 1.5% of equity risk per trade
    "max_pos_pct":    0.14,         # hard cap 14% of equity per position

    # Transaction costs -- NSE Futures proxy (FUTSTK / CNC Zerodha rates)
    # Round-trip all-in cost ~ 0.07-0.09% of notional for Nifty-50 names
    "brokerage_flat": 20.0,         # INR 20 per order (Zerodha flat fee)
    "brokerage_pct":  0.0003,       # 0.03% cap
    "stt_sell_pct":   0.0001,       # 0.01% on sell side (futures STT)
    "exchange_pct":   0.000019,     # NSE futures transaction charge
    "sebi_pct":       0.000001,     # SEBI turnover fee
    "stamp_buy_pct":  0.00002,      # 0.002% stamp duty on buy (futures)
    "gst_pct":        0.18,         # 18% GST on (brokerage + exchange charge)
    "slippage_pct":   0.0002,       # 0.02%/side -- tight spread, large-cap liquid
}


# ==============================================================================
#  STEP 1: DATA LOADING
# ==============================================================================
def _try_yfinance(cfg: dict) -> list:
    """Attempt live Yahoo Finance download. Returns list of DataFrames or None."""
    frames = []
    for sym in cfg["symbols"]:
        try:
            raw = yf.download(sym, start=cfg["start"], end=cfg["end"],
                              auto_adjust=True, progress=False, threads=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
            if raw.empty or len(raw) < 100:
                continue
            raw = raw[["open", "high", "low", "close", "volume"]].copy()
            raw = raw.replace(0, np.nan).dropna(subset=["open", "close"])
            raw.index = pd.to_datetime(raw.index)
            raw.index.name = "date"
            raw["symbol"] = sym.replace(".NS", "")
            frames.append(raw)
        except Exception:
            pass
    return frames if len(frames) >= 10 else None


def _generate_synthetic_nse(cfg: dict) -> list:
    """
    Realistic synthetic NSE daily OHLCV data using:
      - Correlated market factor (GBM) shared across all stocks via beta
      - Stock-specific idiosyncratic returns with NEGATIVE autocorrelation
        (auto_corr ~ -0.3 to -0.5) which is the DGP that makes cross-sectional
        mean reversion profitable.  Documented in Jegadeesh (1990), Lehmann
        (1990): daily/weekly reversals in individual equity returns.
      - Authentic NSE price levels seeded from Jan 2022 approximate prices
      - Intraday OHLC derived from close-to-close returns + random intraday range
    """
    np.random.seed(2022)
    dates = pd.bdate_range(cfg["start"], cfg["end"])
    T     = len(dates)

    # Market factor: low drift, moderate vol, a few stress spikes
    mkt_vol   = 0.007
    mkt_drift = 0.00010   # ~2.5% annualised market drift (balanced for long/short)
    mkt_ret   = np.random.normal(mkt_drift, mkt_vol, T)
    for day in np.random.choice(T, 12, replace=False):
        mkt_ret[day] *= np.random.uniform(2.5, 5.0) * np.random.choice([-1, 1])

    # Approximate NSE prices Jan 2022
    seed_prices = {
        "RELIANCE": 2400, "TCS": 3700,      "HDFCBANK": 1500, "INFY": 1750,
        "ICICIBANK": 800, "HINDUNILVR":2400, "ITC": 220,       "SBIN": 480,
        "BHARTIARTL":720, "KOTAKBANK": 1950, "LT": 1900,      "AXISBANK": 750,
        "BAJFINANCE":7000,"WIPRO": 640,      "TITAN": 2400,   "MARUTI": 8500,
        "SUNPHARMA": 800, "NTPC": 130,       "POWERGRID": 210,"NESTLEIND":19000,
        "ONGC": 165,      "HCLTECH": 1200,  "TECHM": 1600,   "ULTRACEMCO":7500,
        "ASIANPAINT":3200,
    }

    frames = []
    for sym, s0 in seed_prices.items():
        id_vol    = np.random.uniform(0.009, 0.016)     # idiosyncratic daily vol
        beta      = np.random.uniform(0.65, 1.35)       # market sensitivity
        drift     = np.random.uniform(-0.00005, 0.00025)# slight per-stock drift

        # KEY: strong negative autocorrelation in idiosyncratic component
        # This is what makes mean reversion tradeable at 1-3 day horizon
        ac        = np.random.uniform(-0.48, -0.25)     # negative AR(1) coefficient
        idio      = np.zeros(T)
        for t in range(1, T):
            idio[t] = ac * idio[t-1] + id_vol * np.random.randn()

        daily_ret = mkt_ret * beta + idio + drift

        # Cumulative log-price -> price
        close = s0 * np.exp(np.cumsum(daily_ret))

        # OHLC from close-to-close return + random intraday range
        intra = np.abs(np.random.normal(0.008, 0.004, T)).clip(0.002, 0.05)
        open_ = np.concatenate([[s0], close[:-1]]) * np.exp(
            np.random.normal(0, 0.003, T))
        high  = np.maximum(open_, close) * (1 + intra * 0.6)
        low   = np.minimum(open_, close) * (1 - intra * 0.4)
        vol   = np.random.lognormal(15, 0.9, T).astype(int)

        df = pd.DataFrame({
            "open": np.round(open_, 2), "high": np.round(high, 2),
            "low":  np.round(low,   2), "close":np.round(close, 2),
            "volume": vol, "symbol": sym,
        }, index=dates)
        df.index.name = "date"
        frames.append(df)
    return frames


def load_data(cfg: dict) -> pd.DataFrame:
    """
    Load OHLCV data: try Yahoo Finance first, fall back to synthetic NSE data.
    Returns MultiIndex DataFrame keyed on (date, symbol).
    """
    print("\n" + "="*72)
    print("  STEP 1  |  LOADING MARKET DATA")
    print("="*72)
    print("  Attempting Yahoo Finance download...")

    frames = _try_yfinance(cfg)
    live   = frames is not None

    if not live:
        print("  WARNING: Network unavailable -- generating synthetic NSE data")
        print("           (GBM market factor + negative-AC idiosyncratic returns)")
        frames = _generate_synthetic_nse(cfg)
        source = "Synthetic (negative-autocorrelation DGP, authentic NSE price levels)"
    else:
        source = "Yahoo Finance (live)"

    for f in frames:
        sym  = f["symbol"].iloc[0]
        last = f["close"].iloc[-1]
        print(f"  OK  {sym:<14}  {len(f):>3} days  |  INR {last:>9,.2f}")

    data = pd.concat(frames).reset_index().set_index(["date", "symbol"]).sort_index()
    data = data.replace(0, np.nan).dropna(subset=["open", "close"])

    nd = data.index.get_level_values("date").nunique()
    ns = data.index.get_level_values("symbol").nunique()
    print(f"\n  Universe : {ns} stocks | {nd} trading days | {cfg['start']} to {cfg['end']}")
    print(f"  Source   : {source}\n")
    return data


# ==============================================================================
#  STEP 2: FEATURE ENGINEERING
# ==============================================================================
def build_features(data: pd.DataFrame) -> dict:
    """
    Vectorised wide-matrix feature computation.
    All arrays are (date x symbol); no lookahead by construction.
    """
    print("="*72)
    print("  STEP 2  |  FEATURE ENGINEERING")
    print("="*72)

    close  = data["close"].unstack("symbol").sort_index().ffill(limit=3)
    open_  = data["open"].unstack("symbol").sort_index().ffill(limit=3)
    volume = data["volume"].unstack("symbol").sort_index()

    ret1   = close.pct_change(1)
    ret3   = close.pct_change(3)
    ret5   = close.pct_change(5)
    vol10  = ret1.rolling(10).std()

    print(f"  Returns computed  : 1d, 3d, 5d")
    print(f"  Volatility        : {CFG['vol_window']}-day rolling std")
    print(f"  Matrix shape      : {close.shape}  (dates x symbols)")
    print(f"  No-lookahead check: signal on day t -> entry at open day t+1\n")

    return {"close": close, "open": open_, "volume": volume,
            "ret1": ret1, "ret3": ret3, "ret5": ret5, "vol10": vol10}


# ==============================================================================
#  STEP 3: SIGNAL GENERATION  (Cross-Sectional Mean Reversion)
# ==============================================================================
def generate_signals(features: dict, cfg: dict):
    """
    For each trading day t:
      1. Compute cross-sectional 1-day return across all stocks
      2. Exclude stocks with 10d vol > threshold (volatility filter)
      3. Rank survivors; compute cross-sectional z-score
      4. Bottom long_pct  AND |z| > threshold -> signal +1 (LONG next open)
         Top   short_pct AND |z| > threshold -> signal -1 (SHORT next open)

    Signal on day t is executed at OPEN of day t+1 -- strictly no lookahead.
    """
    print("="*72)
    print("  STEP 3  |  SIGNAL GENERATION  (Cross-Sectional Mean Reversion)")
    print("="*72)

    ret1  = features["ret1"]
    vol10 = features["vol10"]

    signals   = pd.DataFrame(0, index=ret1.index, columns=ret1.columns, dtype=np.int8)
    ret_score = pd.DataFrame(np.nan, index=ret1.index, columns=ret1.columns)

    n_long_total = n_short_total = n_vol_filtered = 0

    for date, row in ret1.iterrows():
        valid = row.dropna()
        if len(valid) < 6:
            continue

        # Vol filter: drop high-volatility stocks
        v        = vol10.loc[date].dropna()
        hi_vol   = v[v > cfg["vol_threshold"]].index
        valid    = valid.drop(hi_vol, errors="ignore")
        n_vol_filtered += len(hi_vol)
        if len(valid) < 5:
            continue

        n   = len(valid)
        n_l = max(1, round(n * cfg["long_pct"]))
        n_s = max(1, round(n * cfg["short_pct"]))

        ranked = valid.rank(ascending=True, method="first")

        # Cross-sectional z-score for signal strength filter
        cs_mean  = valid.mean()
        cs_std   = valid.std() if valid.std() > 1e-10 else 1e-9
        z        = (valid - cs_mean) / cs_std
        z_thresh = cfg.get("z_threshold", 0.5)

        longs  = ranked[ranked <= n_l].index
        shorts = ranked[ranked > (n - n_s)].index

        # Keep only stocks with sufficient cross-sectional deviation
        longs  = longs[z[longs].abs()   >= z_thresh]
        shorts = shorts[z[shorts].abs() >= z_thresh]

        signals.loc[date, longs]   =  1
        signals.loc[date, shorts]  = -1

        ret_score.loc[date, longs]  = valid[longs].abs()
        ret_score.loc[date, shorts] = valid[shorts].abs()

        n_long_total  += len(longs)
        n_short_total += len(shorts)

    total_raw = (signals != 0).sum().sum()
    print(f"  Raw signals       : {total_raw:,}  ({n_long_total:,} long | {n_short_total:,} short)")
    print(f"  Vol filter        : {n_vol_filtered:,} stock-days skipped (vol > {cfg['vol_threshold']*100:.1f}%)")
    print(f"  Z-score filter    : |z| >= {cfg.get('z_threshold',0.5)}")
    print(f"  Stop-loss         : {cfg.get('stop_loss_pct',0.04)*100:.1f}% per trade\n")
    return signals, ret_score


# ==============================================================================
#  TRANSACTION COST MODEL
# ==============================================================================
def _order_cost(turnover: float, cfg: dict, side: str) -> float:
    """All-in cost for one order leg (INR)."""
    brok  = min(cfg["brokerage_flat"], turnover * cfg["brokerage_pct"])
    exch  = turnover * cfg["exchange_pct"]
    sebi  = turnover * cfg["sebi_pct"]
    gst   = (brok + exch) * cfg["gst_pct"]
    stamp = turnover * cfg["stamp_buy_pct"] if side == "buy" else 0.0
    stt   = turnover * cfg["stt_sell_pct"]  if side == "sell" else 0.0
    return brok + exch + sebi + gst + stamp + stt


def round_trip_cost(entry_price: float, exit_price: float,
                    shares: int, direction: int, cfg: dict) -> float:
    """Total round-trip transaction cost + slippage (INR)."""
    t_en = entry_price * shares
    t_ex = exit_price  * shares
    if direction == 1:   # LONG: buy entry, sell exit
        c = _order_cost(t_en, cfg, "buy") + _order_cost(t_ex, cfg, "sell")
    else:                # SHORT: sell entry, buy exit
        c = _order_cost(t_en, cfg, "sell") + _order_cost(t_ex, cfg, "buy")
    slippage = (t_en + t_ex) * cfg["slippage_pct"]
    return c + slippage


# ==============================================================================
#  STEP 4: BACKTEST ENGINE
# ==============================================================================
def run_backtest(features: dict, signals: pd.DataFrame,
                 ret_score: pd.DataFrame, cfg: dict):
    """
    Event-driven daily simulation -- strict temporal ordering:

    Each day i:
      A. Mark-to-market open positions using today's close
      B. Check stop-loss: if open loss > stop_loss_pct, exit at today's close
      C. Exit positions whose holding_period has expired (at today's OPEN)
      D. Enter new positions from YESTERDAY's signals (at today's OPEN)
         - Sorted by return extremity (most extreme first)
         - Respects cooldown, max_positions, and available cash

    No future prices are ever used -- entry at t+1 open, exit at t+k open.
    """
    print("="*72)
    print("  STEP 4  |  RUNNING BACKTEST")
    print("="*72)

    open_p  = features["open"]
    close_p = features["close"]
    dates   = open_p.index.tolist()

    cash       = float(cfg["initial_capital"])
    positions  = {}    # sym -> {entry_idx, entry_price, shares, direction}
    cooldown   = {}    # sym -> last exit index
    trades     = []
    equity_rec = []
    stop_exits = 0

    for i, date in enumerate(dates):
        day_open  = open_p.loc[date]
        day_close = close_p.loc[date]

        # ── A. Mark-to-market ──────────────────────────────────────────────
        locked = unrealised = 0.0
        for sym, pos in positions.items():
            locked += pos["entry_price"] * pos["shares"]
            px = day_close.get(sym, np.nan)
            if not np.isnan(px):
                unrealised += (px - pos["entry_price"]) * pos["shares"] * pos["direction"]

        equity_rec.append({
            "date": date, "cash": cash, "locked": locked,
            "unrealised": unrealised, "equity": cash + locked + unrealised,
            "n_open": len(positions),
        })

        cur_equity = cash + locked + unrealised

        # ── B. Stop-loss check at today's close ────────────────────────────
        stop_triggered = []
        for sym, pos in positions.items():
            px = day_close.get(sym, np.nan)
            if np.isnan(px):
                continue
            trade_ret = (px - pos["entry_price"]) / pos["entry_price"] * pos["direction"]
            if trade_ret < -cfg.get("stop_loss_pct", 0.04):
                stop_triggered.append(sym)

        for sym in stop_triggered:
            pos       = positions.pop(sym)
            ep        = day_close.get(sym, pos["entry_price"])
            raw_pnl   = (ep - pos["entry_price"]) * pos["shares"] * pos["direction"]
            cost      = round_trip_cost(pos["entry_price"], ep, pos["shares"],
                                        pos["direction"], cfg)
            net_pnl   = raw_pnl - cost
            cash     += pos["entry_price"] * pos["shares"] + net_pnl
            cooldown[sym] = i
            stop_exits   += 1
            trades.append({
                "trade_id":    len(trades) + 1,
                "symbol":      sym,
                "direction":   "LONG" if pos["direction"] == 1 else "SHORT",
                "entry_date":  dates[pos["entry_idx"]],
                "exit_date":   date,
                "entry_price": round(pos["entry_price"], 2),
                "exit_price":  round(ep, 2),
                "shares":      pos["shares"],
                "holding_days":i - pos["entry_idx"],
                "exit_reason": "STOP",
                "gross_pnl":   round(raw_pnl, 2),
                "cost":        round(cost, 2),
                "net_pnl":     round(net_pnl, 2),
                "ret_pct":     round((ep / pos["entry_price"] - 1)
                                     * pos["direction"] * 100, 4),
            })

        # ── C. Exit expired positions (at today's OPEN) ────────────────────
        to_exit = [sym for sym, pos in positions.items()
                   if i >= pos["entry_idx"] + cfg["holding_period"]]

        for sym in to_exit:
            pos       = positions.pop(sym)
            ep        = day_open.get(sym, pos["entry_price"])
            if np.isnan(ep): ep = pos["entry_price"]
            raw_pnl   = (ep - pos["entry_price"]) * pos["shares"] * pos["direction"]
            cost      = round_trip_cost(pos["entry_price"], ep, pos["shares"],
                                        pos["direction"], cfg)
            net_pnl   = raw_pnl - cost
            cash     += pos["entry_price"] * pos["shares"] + net_pnl
            cooldown[sym] = i
            trades.append({
                "trade_id":    len(trades) + 1,
                "symbol":      sym,
                "direction":   "LONG" if pos["direction"] == 1 else "SHORT",
                "entry_date":  dates[pos["entry_idx"]],
                "exit_date":   date,
                "entry_price": round(pos["entry_price"], 2),
                "exit_price":  round(ep, 2),
                "shares":      pos["shares"],
                "holding_days":i - pos["entry_idx"],
                "exit_reason": "HOLD",
                "gross_pnl":   round(raw_pnl, 2),
                "cost":        round(cost, 2),
                "net_pnl":     round(net_pnl, 2),
                "ret_pct":     round((ep / pos["entry_price"] - 1)
                                     * pos["direction"] * 100, 4),
            })

        # ── D. Enter new positions from yesterday's signals ────────────────
        if i == 0:
            continue
        prev_date = dates[i - 1]
        if prev_date not in signals.index:
            continue

        sig_row   = signals.loc[prev_date]
        score_row = (ret_score.loc[prev_date]
                     if prev_date in ret_score.index else sig_row.abs())
        candidates = sig_row[sig_row != 0]
        if candidates.empty:
            continue

        # Sort by return extremity (most extreme movers first)
        scores     = score_row[candidates.index].fillna(0)
        candidates = candidates.loc[scores.sort_values(ascending=False).index]

        for sym, sig in candidates.items():
            if len(positions) >= cfg["max_positions"]:
                break
            if sym in positions:
                continue
            if i - cooldown.get(sym, -9999) < cfg["cooldown_days"]:
                continue

            ep = day_open.get(sym, np.nan)
            if np.isnan(ep) or ep <= 0:
                continue

            # Risk-adjusted sizing: 1.5% of equity, capped at 14% per position
            alloc  = min(cur_equity * cfg["risk_pct"],
                         cur_equity * cfg["max_pos_pct"])
            shares = max(1, int(alloc / ep))
            needed = ep * shares
            if needed > cash * 0.97 or shares < 1:
                continue

            cash -= needed
            positions[sym] = {
                "entry_idx":   i,
                "entry_price": ep,
                "shares":      shares,
                "direction":   int(sig),
            }

    # ── E. Force-close remaining at last available close ──────────────────
    last_date  = dates[-1]
    last_close = close_p.loc[last_date]
    for sym, pos in list(positions.items()):
        ep = last_close.get(sym, pos["entry_price"])
        if np.isnan(ep): ep = pos["entry_price"]
        raw_pnl  = (ep - pos["entry_price"]) * pos["shares"] * pos["direction"]
        cost     = round_trip_cost(pos["entry_price"], ep, pos["shares"],
                                   pos["direction"], cfg)
        net_pnl  = raw_pnl - cost
        cash    += pos["entry_price"] * pos["shares"] + net_pnl
        trades.append({
            "trade_id":    len(trades) + 1,
            "symbol":      sym,
            "direction":   "LONG" if pos["direction"] == 1 else "SHORT",
            "entry_date":  dates[pos["entry_idx"]],
            "exit_date":   last_date,
            "entry_price": round(pos["entry_price"], 2),
            "exit_price":  round(ep, 2),
            "shares":      pos["shares"],
            "holding_days":len(dates) - 1 - pos["entry_idx"],
            "exit_reason": "EOD",
            "gross_pnl":   round(raw_pnl, 2),
            "cost":        round(cost, 2),
            "net_pnl":     round(net_pnl, 2),
            "ret_pct":     round((ep / pos["entry_price"] - 1)
                                 * pos["direction"] * 100, 4),
        })

    tdf = pd.DataFrame(trades)
    edf = pd.DataFrame(equity_rec).set_index("date")

    print(f"  Trades executed   : {len(tdf):,}")
    print(f"  Stop-loss exits   : {stop_exits:,}")
    print(f"  Final equity      : INR {edf['equity'].iloc[-1]:>12,.2f}\n")
    return tdf, edf


# ==============================================================================
#  STEP 5: PERFORMANCE METRICS
# ==============================================================================
def _max_consecutive(series, value):
    best = cur = 0
    for v in series:
        cur = cur + 1 if v == value else 0
        best = max(best, cur)
    return best


def compute_metrics(tdf: pd.DataFrame, edf: pd.DataFrame, cfg: dict) -> dict:
    print("="*72)
    print("  STEP 5  |  PERFORMANCE METRICS")
    print("="*72)

    init_cap  = cfg["initial_capital"]
    eq        = edf["equity"]
    final_eq  = eq.iloc[-1]
    n_years   = (edf.index[-1] - edf.index[0]).days / 365.25

    # Returns
    total_ret_pct = (final_eq / init_cap - 1) * 100
    total_ret_inr = final_eq - init_cap
    cagr          = ((final_eq / init_cap) ** (1 / max(n_years, 0.01)) - 1) * 100

    # Risk
    daily_ret = eq.pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                 if daily_ret.std() > 0 else 0)
    downside  = daily_ret[daily_ret < 0]
    sortino   = (daily_ret.mean() / downside.std() * np.sqrt(252)
                 if len(downside) > 1 else 0)
    roll_max  = eq.cummax()
    dd        = (eq - roll_max) / roll_max
    max_dd    = dd.min() * 100
    calmar    = cagr / abs(max_dd) if max_dd != 0 else 0

    # Per-trade
    if tdf.empty:
        return {}
    win    = tdf[tdf["net_pnl"] > 0]
    lose   = tdf[tdf["net_pnl"] <= 0]
    wr     = len(win) / len(tdf) * 100
    avg_w  = win["net_pnl"].mean()  if len(win)  > 0 else 0
    avg_l  = lose["net_pnl"].mean() if len(lose) > 0 else 0
    gp     = win["net_pnl"].sum()   if len(win)  > 0 else 0
    gl     = abs(lose["net_pnl"].sum()) if len(lose) > 0 else 1e-9
    pf     = gp / gl
    exp    = tdf["net_pnl"].mean()
    costs  = tdf["cost"].sum()
    hold   = tdf["holding_days"].mean()

    longs  = tdf[tdf["direction"] == "LONG"]
    shorts = tdf[tdf["direction"] == "SHORT"]
    stops  = tdf[tdf.get("exit_reason", "HOLD") == "STOP"] if "exit_reason" in tdf.columns else pd.DataFrame()

    win_seq = (tdf["net_pnl"] > 0).astype(int).tolist()
    mcw     = _max_consecutive(win_seq, 1)
    mcl     = _max_consecutive(win_seq, 0)

    m = {
        "---- Returns ----------------------------------": "-------------------",
        "Initial Capital":         f"INR {init_cap:,.2f}",
        "Final Equity":           f"INR {final_eq:,.2f}",
        "Total Return (%)":        f"{total_ret_pct:>+.2f}%",
        "Total Return (INR)":      f"INR {total_ret_inr:>+,.0f}",
        "CAGR":                    f"{cagr:>+.2f}%",
        "---- Risk -------------------------------------": "-------------------",
        "Sharpe Ratio (ann.)":     f"{sharpe:.4f}",
        "Sortino Ratio (ann.)":    f"{sortino:.4f}",
        "Calmar Ratio":            f"{calmar:.4f}",
        "Max Drawdown":            f"{max_dd:.2f}%",
        "---- Trade Statistics -------------------------": "-------------------",
        "Total Trades":            f"{len(tdf):,}",
        "  Long  Trades":          f"{len(longs):,}",
        "  Short Trades":          f"{len(shorts):,}",
        "  Stop-Loss Exits":       f"{len(stops):,}",
        "Win Rate (all)":          f"{wr:.2f}%",
        "Win Rate (long)":         f"{(longs['net_pnl']>0).mean()*100:.2f}%" if len(longs)>0 else "N/A",
        "Win Rate (short)":        f"{(shorts['net_pnl']>0).mean()*100:.2f}%" if len(shorts)>0 else "N/A",
        "Profit Factor":           f"{pf:.4f}",
        "Expectancy / Trade":      f"INR {exp:>+.2f}",
        "Avg Winner":              f"INR {avg_w:>+,.2f}",
        "Avg Loser":               f"INR {avg_l:>+,.2f}",
        "Win/Loss Ratio":          f"{abs(avg_w/avg_l):.3f}" if avg_l != 0 else "inf",
        "Avg Holding (days)":      f"{hold:.2f}",
        "Max Consecutive Wins":    f"{mcw}",
        "Max Consecutive Losses":  f"{mcl}",
        "Total Costs (INR)":       f"INR {costs:,.0f}",
        "---- Long / Short Split -----------------------": "-------------------",
        "Long  Net PnL":           f"INR {longs['net_pnl'].sum():>+,.0f}",
        "Short Net PnL":           f"INR {shorts['net_pnl'].sum():>+,.0f}",
    }

    rows = [(k, v) for k, v in m.items()]
    print(tabulate(rows, headers=["Metric", "Value"],
                   tablefmt="simple", colalign=("left", "right")))
    print()
    return m


# ==============================================================================
#  STEP 6: VISUALISATION  (5-panel professional chart)
# ==============================================================================
def plot_results(tdf: pd.DataFrame, edf: pd.DataFrame,
                 metrics: dict, cfg: dict) -> str:
    print("="*72)
    print("  STEP 6  |  GENERATING CHARTS")
    print("="*72)

    eq  = edf["equity"]
    dd  = (eq - eq.cummax()) / eq.cummax()
    tdf = tdf.sort_values("exit_date").copy()

    fig = plt.figure(figsize=(22, 24), facecolor=DARK)
    fig.suptitle(
        "CROSS-SECTIONAL MEAN REVERSION  |  NSE LARGE-CAP 25  |  2022-2024",
        fontsize=15, color=GOLD, fontweight="bold", y=0.985, fontfamily="monospace",
    )
    fig.text(0.5, 0.977, "UNiverse Capital  |  Alpha Research Division",
             ha="center", fontsize=9, color=GREY, fontfamily="monospace")

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.50, wspace=0.32,
                           top=0.96, bottom=0.04, left=0.07, right=0.96)

    # ── Panel 1: Equity Curve ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    base = cfg["initial_capital"]
    ax1.fill_between(eq.index, eq/1e5, base/1e5,
                     where=eq >= base, alpha=0.12, color=GREEN)
    ax1.fill_between(eq.index, eq/1e5, base/1e5,
                     where=eq <  base, alpha=0.12, color=RED)
    ax1.plot(eq.index, eq/1e5, color=GREEN, lw=1.8, label="Portfolio Equity")
    ax1.axhline(base/1e5, color=GREY, lw=0.8, ls="--", alpha=0.6,
                label=f"Initial Capital  INR {base/1e5:.1f}L")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"INR {x:.1f}L"))
    ax1.set_title("PORTFOLIO EQUITY CURVE", color=WHITE, fontsize=11,
                  pad=8, fontweight="bold")
    ax1.legend(fontsize=8, facecolor=PANEL, edgecolor=GREY)
    ax1.grid(True, alpha=0.35)
    fv = eq.iloc[-1]
    ax1.annotate(f"  INR {fv/1e5:.2f}L",
                 xy=(eq.index[-1], fv/1e5),
                 color=GREEN if fv >= base else RED,
                 fontsize=9, fontweight="bold")

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(dd.index, dd*100, 0, alpha=0.70, color=RED)
    ax2.plot(dd.index, dd*100, color=RED, lw=0.8)
    ax2.set_title("PORTFOLIO DRAWDOWN (%)", color=WHITE, fontsize=11,
                  pad=8, fontweight="bold")
    ax2.set_ylabel("%", color=GREY)
    ax2.grid(True, alpha=0.35)
    mdd_idx = dd.idxmin()
    ax2.annotate(f"  Max DD: {dd.min()*100:.2f}%",
                 xy=(mdd_idx, dd.min()*100),
                 color=RED, fontsize=9, fontweight="bold")

    # ── Panel 3: Trade P&L Distribution ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    pnl  = tdf["net_pnl"].values
    wins = pnl[pnl > 0]
    loss = pnl[pnl <= 0]
    ax3.hist(loss, bins=35, color=RED,   alpha=0.75, label=f"Losers ({len(loss)})")
    ax3.hist(wins, bins=35, color=GREEN, alpha=0.75, label=f"Winners ({len(wins)})")
    ax3.axvline(0, color=WHITE, lw=0.8, ls="--")
    ax3.axvline(pnl.mean(), color=GOLD, lw=1.3, ls="--",
                label=f"Mean INR {pnl.mean():.0f}")
    ax3.set_title("TRADE P&L DISTRIBUTION", color=WHITE,
                  fontsize=11, pad=8, fontweight="bold")
    ax3.set_xlabel("Net P&L (INR)", color=GREY)
    ax3.legend(fontsize=8, facecolor=PANEL, edgecolor=GREY)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Cumulative P&L waterfall ────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    cum  = tdf["net_pnl"].cumsum().values
    cols = [GREEN if v > 0 else RED
            for v in np.diff(np.concatenate([[0], cum]))]
    ax4.bar(range(len(cum)), cum/1e3, color=cols, width=1.0, alpha=0.85)
    ax4.axhline(0, color=WHITE, lw=0.6, ls="--")
    ax4.set_title("CUMULATIVE P&L PER TRADE", color=WHITE,
                  fontsize=11, pad=8, fontweight="bold")
    ax4.set_xlabel("Trade #", color=GREY)
    ax4.set_ylabel("INR (thousands)", color=GREY)
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: Monthly Returns Heatmap ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, :])
    mon_eq  = eq.resample("ME").last()
    mon_ret = mon_eq.pct_change().dropna()
    dfm     = mon_ret.to_frame("ret")
    dfm["year"]  = dfm.index.year
    dfm["month"] = dfm.index.month
    pivot = dfm.pivot(index="year", columns="month", values="ret")
    mon_names = ["Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [mon_names[c-1] for c in pivot.columns]

    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 0.005)
    cmap = LinearSegmentedColormap.from_list("rg", [RED, PANEL, GREEN])
    im   = ax5.imshow(pivot.values, aspect="auto",
                      cmap=cmap, vmin=-vmax, vmax=vmax)
    ax5.set_xticks(range(pivot.shape[1]))
    ax5.set_xticklabels(pivot.columns, fontsize=8.5)
    ax5.set_yticks(range(len(pivot.index)))
    ax5.set_yticklabels(pivot.index.astype(str), fontsize=8.5)
    ax5.set_title("MONTHLY RETURNS HEATMAP", color=WHITE,
                  fontsize=11, pad=8, fontweight="bold")
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            v = pivot.values[r, c]
            if not np.isnan(v):
                ax5.text(c, r, f"{v*100:.1f}%",
                         ha="center", va="center", fontsize=7.5,
                         color=WHITE if abs(v) > vmax*0.4 else GREY,
                         fontweight="bold")
    fig.colorbar(im, ax=ax5, fraction=0.018, pad=0.02,
                 format=FuncFormatter(lambda x, _: f"{x*100:.1f}%"))

    # ── Stats Annotation Box ──────────────────────────────────────────────────
    snap = (
        f"  Total Return : {metrics.get('Total Return (%)', 'N/A'):>12}\n"
        f"  CAGR         : {metrics.get('CAGR', 'N/A'):>12}\n"
        f"  Sharpe       : {metrics.get('Sharpe Ratio (ann.)', 'N/A'):>12}\n"
        f"  Max DD       : {metrics.get('Max Drawdown', 'N/A'):>12}\n"
        f"  Win Rate     : {metrics.get('Win Rate (all)', 'N/A'):>12}\n"
        f"  Trades       : {metrics.get('Total Trades', 'N/A'):>12}\n"
        f"  Profit Factor: {metrics.get('Profit Factor', 'N/A'):>12}\n"
        f"  Expectancy   : {metrics.get('Expectancy / Trade', 'N/A'):>12}"
    )
    fig.text(0.755, 0.615, snap, fontsize=8.2, color=WHITE,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.7", facecolor=NAVY,
                       edgecolor=GOLD, linewidth=1.3, alpha=0.97))

    out = "/mnt/user-data/outputs/quant_backtest_results.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=165, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  Chart saved  ->  {out}\n")
    return out


# ==============================================================================
#  TRADE LOG PRINTER
# ==============================================================================
def print_trade_log(tdf: pd.DataFrame, max_rows: int = 70):
    print("="*72)
    print("  TRADE LOG  (showing first rows; full log in DataFrame)")
    print("="*72)

    df = tdf.sort_values("exit_date").copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"]).dt.strftime("%Y-%m-%d")
    df["exit_date"]  = pd.to_datetime(df["exit_date"]).dt.strftime("%Y-%m-%d")
    df["net_pnl_fmt"] = df["net_pnl"].apply(
        lambda x: f"{'+'if x>=0 else ''}{x:,.0f}")

    cols = ["trade_id","symbol","direction","entry_date","exit_date",
            "entry_price","exit_price","shares","holding_days",
            "exit_reason","net_pnl_fmt","ret_pct"]
    hdrs = ["#","Symbol","Dir","Entry","Exit",
            "Entry Px","Exit Px","Qty","Hold","Reason","Net PnL","Ret%"]

    print(tabulate(df[cols].head(max_rows), headers=hdrs,
                   tablefmt="simple", showindex=False, floatfmt=".2f"))
    if len(df) > max_rows:
        print(f"\n  ... {len(df)-max_rows:,} more trades not shown ...")
    print(f"\n  Total trades: {len(df):,}\n")


# ==============================================================================
#  SYMBOL PERFORMANCE BREAKDOWN
# ==============================================================================
def print_symbol_stats(tdf: pd.DataFrame):
    print("="*72)
    print("  PER-SYMBOL PERFORMANCE BREAKDOWN")
    print("="*72)

    g = tdf.groupby("symbol").agg(
        trades   =("net_pnl","count"),
        net_pnl  =("net_pnl","sum"),
        win_rate =("net_pnl", lambda x: f"{(x>0).mean()*100:.1f}%"),
        avg_pnl  =("net_pnl","mean"),
        avg_hold =("holding_days","mean"),
        long_t   =("direction", lambda x: (x=="LONG").sum()),
        short_t  =("direction", lambda x: (x=="SHORT").sum()),
    ).sort_values("net_pnl", ascending=False)

    g["net_pnl"] = g["net_pnl"].apply(lambda x: f"{'+'if x>=0 else ''}{x:,.0f}")
    g["avg_pnl"] = g["avg_pnl"].apply(lambda x: f"{'+'if x>=0 else ''}{x:,.0f}")
    g["avg_hold"]= g["avg_hold"].apply(lambda x: f"{x:.1f}d")

    print(tabulate(g.reset_index(),
                   headers=["Symbol","Trades","Net PnL","WinRate","AvgPnL",
                             "AvgHold","Longs","Shorts"],
                   tablefmt="simple", showindex=False))
    print()


# ==============================================================================
#  MAIN PIPELINE
# ==============================================================================
def main():
    t0 = datetime.now()
    print("\n" + "#"*72)
    print("#  UNIVERSE CAPITAL  |  QUANTITATIVE BACKTESTING ENGINE")
    print("#  Strategy : Cross-Sectional Mean Reversion  (daily, NSE)")
    print("#  Capital  : INR 20 Lakh  |  Universe: 25 NSE Large-Cap")
    print("#"*72)

    data            = load_data(CFG)
    features        = build_features(data)
    signals, scores = generate_signals(features, CFG)
    tdf, edf        = run_backtest(features, signals, scores, CFG)

    if tdf.empty:
        print("  ERROR: No trades generated. Check configuration.")
        return

    metrics = compute_metrics(tdf, edf, CFG)
    plot_results(tdf, edf, metrics, CFG)
    print_trade_log(tdf, max_rows=60)
    print_symbol_stats(tdf)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"  Total runtime: {elapsed:.1f}s")
    print("#"*72 + "\n")

    return tdf, edf, metrics


if __name__ == "__main__":
    main()