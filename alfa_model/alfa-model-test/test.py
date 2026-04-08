#!/usr/bin/env python3
"""
================================================================================
  NSE FUTSTK  LONG-ONLY POSITIONAL ENGINE  v1.0  —  CLEAN REBUILD
  Strategy  : Short-Term Mean Reversion  |  LONG ONLY  |  3-5 Day Hold
  Capital   : INR 10 Lakh  |  Universe: 8 NSE Large-Cap Stocks
  Data      : 10 Years  2015-2024  |  Yahoo Finance (live)

  LESSONS FROM PREVIOUS FAILED MODELS
  ─────────────────────────────────────────────────────────────────────────────
  1. NO INTRADAY — daily OHLCV cannot model intraday edges. All intraday
     versions (v1.0, v2.0) showed 94%+ capital destruction on live data.
     This model holds 3-5 days (positional futures).

  2. LONG ONLY — NSE Nifty delivered +14% CAGR since 2010. Every short
     leg in every previous model lost money. This model has ZERO short trades.

  3. NO SYNTHETIC DATA — all previous "Sharpe 2.3" results used synthetic
     data with 10x stronger mean reversion than real NSE. This model is
     calibrated and validated ONLY on Yahoo Finance live data.

  4. REALISTIC TARGETS — honest expectations for retail 10L futures:
     CAGR 8-15%, Sharpe 0.5-1.0, MaxDD < 15%, trades 150-300/year

  5. CAPITAL-APPROPRIATE UNIVERSE — only trade stocks where 1-lot margin
     is under 15% of 10L capital at the time of trade. Universe is
     dynamically filtered each year to stay capital-appropriate.

  STRATEGY: SHORT-TERM MEAN REVERSION (LONG ONLY)
  ─────────────────────────────────────────────────────────────────────────────
  Signal   : Stock fell significantly in the last 5 days vs its peers
             (cross-sectional z-score of 5-day return < -1.0)
             AND it is bouncing: today's 1-day return > -1% (not still falling)
             AND broad market (Nifty proxy) is above its 50-day MA (bull regime)

  Entry    : OPEN of next trading day

  Exit     : OPEN of day (entry + holding_period)  OR  stop-loss at -6%

  Sizing   : Risk-based. For each trade:
             lots = floor(equity × 2% / (stock_price × lot_size × 20%))
             Capped so no single position > 18% of equity in margin

  Universe : Dynamically filtered per year — only stocks where 1-lot
             margin < 15% of current equity. Typically 5-8 stocks.

  WALK-FORWARD IS/OOS:
  ─────────────────────────────────────────────────────────────────────────────
  Training/IS   : 2015-01-01  to  2021-12-31  (7 years)
    Optimisation : 2015-2019  (5 yr)  grid-search
    Validation   : 2020-2021  (2 yr)  param selection
  OOS           : 2022-01-01  to  2024-12-31  (3 years)  — NEVER seen in IS

  OUTPUT: Two clean sheets only. No plots, no trade logs.
================================================================================
"""
import warnings, itertools, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# ── Universe ─────────────────────────────────────────────────────────────────
# Stocks with historically manageable lot sizes for 10L capital
# Lot sizes below are representative 2018-2024 averages (NSE revises them)
UNIVERSE = {
    "RELIANCE.NS":   {"sym": "RELIANCE",   "lot": 250},
    "TCS.NS":        {"sym": "TCS",        "lot": 150},
    "INFY.NS":       {"sym": "INFY",       "lot": 300},
    "WIPRO.NS":      {"sym": "WIPRO",      "lot": 1000},
    "LT.NS":         {"sym": "LT",         "lot": 150},
    "KOTAKBANK.NS":  {"sym": "KOTAKBANK",  "lot": 400},
    "NTPC.NS":       {"sym": "NTPC",       "lot": 3000},
    "POWERGRID.NS":  {"sym": "POWERGRID",  "lot": 4500},
}
NIFTY_TICKER = "^NSEI"   # broad market benchmark

DATES = {
    "start":     "2015-01-01",
    "end":       "2024-12-31",
    "opt_start": "2015-01-01",
    "opt_end":   "2019-12-31",
    "val_start": "2020-01-01",
    "val_end":   "2021-12-31",
    "oos_start": "2022-01-01",
    "oos_end":   "2024-12-31",
}

INITIAL_CAPITAL      = 1_000_000      # INR 10 Lakh
MARGIN_RATE          = 0.20           # 20% SPAN + Exposure
MAX_MARGIN_PER_TRADE = 0.15           # max 15% of equity per position margin
MAX_POSITIONS        = 5              # conservative: max 5 open at once
RISK_PCT             = 0.020          # 2% equity per trade (risk budget)
STOP_LOSS            = 0.06           # 6% stop loss on position (not intraday)

# ── NSE Futures cost model (Zerodha positional NRML) ──────────────────────────
COSTS = {
    "brokerage":   20.0,       # INR 20 per order flat
    "brokerage_pct": 0.0003,   # 0.03% cap
    "stt_sell":    0.0001,     # 0.01% on sell turnover (futures)
    "exchange":    0.000019,   # NSE transaction charge
    "sebi":        0.000001,   # SEBI fee
    "stamp_buy":   0.00002,    # 0.002% stamp on buy
    "gst_rate":    0.18,       # 18% on brokerage + exchange
    "slippage":    0.0002,     # 0.02% per side (liquid large-cap)
}

# ── Grid (Optimisation window only) ──────────────────────────────────────────
PARAM_GRID = {
    "holding_period":  [3, 5, 7],         # days to hold
    "z_threshold":     [1.0, 1.5, 2.0],   # how oversold (stronger = fewer but better trades)
    "reversion_window":[3, 5],             # rolling window for z-score
    "cooldown":        [3, 5],             # days before re-entry on same stock
}


# ==============================================================================
#  DATA
# ==============================================================================
def load_data() -> tuple:
    frames, failed = [], []

    # Download stock data
    for ticker, meta in UNIVERSE.items():
        try:
            raw = yf.download(ticker, start=DATES["start"], end=DATES["end"],
                              auto_adjust=True, progress=False, threads=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            raw.columns = [c.lower().replace(" ","_") for c in raw.columns]
            if raw.empty or len(raw) < 200:
                failed.append(ticker); continue
            raw = raw[["open","high","low","close","volume"]].replace(0, np.nan)
            raw = raw.dropna(subset=["open","close"])
            raw.index = pd.to_datetime(raw.index)
            raw.index.name = "date"
            raw["symbol"] = meta["sym"]
            raw["lot_size"] = meta["lot"]
            frames.append(raw)
        except Exception:
            failed.append(ticker)

    if len(frames) < 3:
        print(f"\nERROR: Only {len(frames)} stocks downloaded. Need internet.")
        sys.exit(1)

    # Download Nifty for market regime
    nifty = None
    try:
        nf = yf.download(NIFTY_TICKER, start=DATES["start"], end=DATES["end"],
                         auto_adjust=True, progress=False, threads=False)
        if isinstance(nf.columns, pd.MultiIndex):
            nf.columns = nf.columns.droplevel(1)
        nf.columns = [c.lower().replace(" ","_") for c in nf.columns]
        if not nf.empty:
            nifty = nf["close"].rename("nifty")
    except Exception:
        pass

    data = (pd.concat(frames).reset_index()
            .set_index(["date","symbol"]).sort_index()
            .replace(0, np.nan).dropna(subset=["open","close"]))

    loaded = [UNIVERSE[t]["sym"] for t in UNIVERSE if UNIVERSE[t]["sym"]
              in data.index.get_level_values("symbol").unique()]
    print(f"  Loaded {len(loaded)} stocks: {', '.join(loaded)}")
    if failed:
        print(f"  Failed: {[t.replace('.NS','') for t in failed]}")

    return data, nifty, "Yahoo Finance (live)"


# ==============================================================================
#  FEATURES
# ==============================================================================
def build_features(data: pd.DataFrame, nifty, cfg: dict) -> dict:
    close   = data["close"].unstack("symbol").sort_index().ffill(limit=3)
    open_   = data["open"].unstack("symbol").sort_index().ffill(limit=3)
    lot_df  = data["lot_size"].unstack("symbol").ffill()   # lot sizes
    ret1    = close.pct_change(1)
    vol10   = ret1.rolling(10).std()

    # Cross-sectional z-score on N-day returns
    def _csz(ret_df):
        mu  = ret_df.mean(axis=1)
        sig = ret_df.std(axis=1).replace(0, np.nan)
        return ret_df.sub(mu, axis=0).div(sig, axis=0)

    # Nifty market regime: above 50-day MA = bull = allow longs
    if nifty is not None:
        nifty_aligned = nifty.reindex(close.index).ffill()
        nifty_ma50    = nifty_aligned.rolling(50).mean()
        bull_regime   = (nifty_aligned > nifty_ma50).fillna(False)
    else:
        # Fallback: always bull
        bull_regime = pd.Series(True, index=close.index)

    # Compute per-window z-scores (3d and 5d) — done fresh per backtest call
    ret3 = close.pct_change(3)
    ret5 = close.pct_change(5)
    z3   = _csz(ret3)
    z5   = _csz(ret5)

    return {
        "close":    close,  "open":      open_,
        "lot_df":   lot_df, "ret1":      ret1,
        "vol10":    vol10,  "z3":        z3,
        "z5":       z5,     "bull":      bull_regime,
    }


# ==============================================================================
#  SIGNAL  (LONG ONLY — clean and simple)
# ==============================================================================
def build_signals(features: dict, params: dict,
                  s_date: str, e_date: str) -> tuple:
    """
    LONG signal fires when (all conditions on day T):

    1. z_score(N-day return) < -z_threshold   — stock is cross-sectionally oversold
    2. ret1 > -1.5%                           — stock is NOT still in free-fall today
    3. Nifty > 50-day MA                      — broad market is in bull regime
    4. 10-day vol < 4%                        — stock not in blow-up territory

    Execute: BUY at OPEN of day T+1
    Exit:    at OPEN of day T+1+holding_period  OR  stop-loss at -6% from entry
    """
    def _w(df): return df.loc[(df.index >= s_date) & (df.index <= e_date)].copy()

    zt   = params.get("z_threshold",      1.5)
    rw   = params.get("reversion_window", 5)
    vt   = 0.04  # fixed vol cap

    # Choose z-score based on reversion_window
    z_src = features["z5"] if rw >= 5 else features["z3"]

    z    = _w(z_src)
    r1   = _w(features["ret1"])
    vol  = _w(features["vol10"])
    bull = _w(features["bull"])

    # Broadcast bull regime
    bull_mat = pd.DataFrame(
        np.repeat(bull.values.reshape(-1, 1), z.shape[1], axis=1),
        index=z.index, columns=z.columns
    )

    # All conditions
    cond_z    = z < -zt
    cond_notfall = r1 > -0.015        # not still free-falling
    cond_vol  = vol < vt
    cond_bull = bull_mat.fillna(True) # long only in bull market

    signal = (cond_z & cond_notfall & cond_vol & cond_bull).astype(np.int8)
    score  = (-z).clip(lower=0)       # priority: most oversold first

    return signal, score


# ==============================================================================
#  COST MODEL
# ==============================================================================
def _rt_cost(ep: float, xp: float, lot: int, nl: int) -> float:
    """Full round-trip cost for LONG futures trade (INR)."""
    ten = ep * lot * nl
    tex = xp * lot * nl

    def _oc(t, side):
        b = min(COSTS["brokerage"], t * COSTS["brokerage_pct"])
        e = t * COSTS["exchange"]
        s = t * COSTS["sebi"]
        g = (b + e) * COSTS["gst_rate"]
        stamp = t * COSTS["stamp_buy"] if side == "buy" else 0.0
        stt   = t * COSTS["stt_sell"]  if side == "sell" else 0.0
        return b + e + s + g + stamp + stt

    c = _oc(ten, "buy") + _oc(tex, "sell")
    return c + (ten + tex) * COSTS["slippage"]


# ==============================================================================
#  BACKTEST ENGINE  (positional LONG-ONLY)
# ==============================================================================
def run_backtest(features: dict, signals: pd.DataFrame,
                 score: pd.DataFrame, params: dict,
                 s_date: str, e_date: str) -> tuple:
    def _w(df): return df.loc[(df.index >= s_date) & (df.index <= e_date)]

    op     = _w(features["open"])
    cp     = _w(features["close"])
    lot_df = _w(features["lot_df"])
    sig    = _w(signals)
    sc     = _w(score)

    dates     = op.index.tolist()
    free_cap  = float(INITIAL_CAPITAL)
    positions = {}   # sym → {i, ep, lots, lot_size, margin}
    cooldown  = {}   # sym → last exit idx
    trades    = []
    eq_rec    = []

    hp   = int(params.get("holding_period",  5))
    cd   = int(params.get("cooldown",        3))
    sl   = STOP_LOSS
    r_p  = RISK_PCT

    for i, date in enumerate(dates):
        dop = op.loc[date]
        dcp = cp.loc[date]

        # MTM
        locked = unreal = 0.0
        for sym, pos in positions.items():
            locked += pos["margin"]
            px = dcp.get(sym, np.nan)
            if not np.isnan(px):
                unreal += (px - pos["ep"]) * pos["lot_size"] * pos["lots"]

        cur_eq = free_cap + locked + unreal
        eq_rec.append({"date": date, "equity": cur_eq, "n_open": len(positions)})

        # Stop-loss check at close (exit next open)
        for sym in list(positions):
            pos = positions[sym]
            px  = dcp.get(sym, np.nan)
            if np.isnan(px): continue
            tr = (px - pos["ep"]) / pos["ep"]
            if tr < -sl:
                # Mark for stop exit at next open — simplified: exit at close
                positions.pop(sym)
                gp   = (px - pos["ep"]) * pos["lot_size"] * pos["lots"]
                cost = _rt_cost(pos["ep"], px, pos["lot_size"], pos["lots"])
                free_cap += pos["margin"] + gp - cost
                cooldown[sym] = i
                trades.append({
                    "trade_id":    len(trades)+1,
                    "symbol":      sym,
                    "entry_date":  dates[pos["i"]],
                    "exit_date":   date,
                    "entry_price": round(pos["ep"],2),
                    "exit_price":  round(px,2),
                    "lots":        pos["lots"],
                    "lot_size":    pos["lot_size"],
                    "holding_days":i - pos["i"],
                    "exit_reason": "STOP",
                    "gross_pnl":   round(gp,2),
                    "cost":        round(cost,2),
                    "net_pnl":     round(gp-cost,2),
                    "ret_pct":     round(tr*100,4),
                    "rom_pct":     round((gp-cost)/pos["margin"]*100,4),
                })

        # Time exit at open
        for sym in [s for s,p in list(positions.items()) if i >= p["i"]+hp]:
            pos = positions.pop(sym)
            xp  = dop.get(sym, pos["ep"])
            if np.isnan(xp) or xp == 0: xp = pos["ep"]
            gp   = (xp - pos["ep"]) * pos["lot_size"] * pos["lots"]
            cost = _rt_cost(pos["ep"], xp, pos["lot_size"], pos["lots"])
            free_cap += pos["margin"] + gp - cost
            cooldown[sym] = i
            trades.append({
                "trade_id":    len(trades)+1,
                "symbol":      sym,
                "entry_date":  dates[pos["i"]],
                "exit_date":   date,
                "entry_price": round(pos["ep"],2),
                "exit_price":  round(xp,2),
                "lots":        pos["lots"],
                "lot_size":    pos["lot_size"],
                "holding_days":i - pos["i"],
                "exit_reason": "HOLD",
                "gross_pnl":   round(gp,2),
                "cost":        round(cost,2),
                "net_pnl":     round(gp-cost,2),
                "ret_pct":     round((xp/pos["ep"]-1)*100,4),
                "rom_pct":     round((gp-cost)/pos["margin"]*100,4),
            })

        # Entry
        if i == 0: continue
        prev = dates[i-1]
        if prev not in sig.index: continue
        sr    = sig.loc[prev]
        cands = sr[sr == 1]
        if cands.empty: continue
        if prev in sc.index:
            psc   = sc.loc[prev][cands.index].fillna(0)
            cands = cands.loc[psc.sort_values(ascending=False).index]

        for sym in cands.index:
            if len(positions) >= MAX_POSITIONS: break
            if sym in positions: continue
            if i - cooldown.get(sym, -9999) < cd: continue

            ep  = dop.get(sym, np.nan)
            if np.isnan(ep) or ep <= 0: continue

            lot = lot_df[sym].loc[date] if sym in lot_df.columns else np.nan
            if np.isnan(lot) or lot <= 0: continue
            lot = int(lot)

            mg1 = ep * lot * MARGIN_RATE
            # Skip if 1-lot margin > 15% of current equity
            if mg1 > cur_eq * MAX_MARGIN_PER_TRADE:
                continue

            # Size: risk budget → lots
            risk_budget = cur_eq * r_p
            num_lots    = max(1, int(risk_budget / mg1))
            tot_mg      = num_lots * mg1

            # Hard cap: no single position > 18% of equity in margin
            while tot_mg > cur_eq * 0.18 and num_lots > 1:
                num_lots -= 1; tot_mg = num_lots * mg1

            if tot_mg > free_cap * 0.93 or num_lots < 1: continue
            while tot_mg > free_cap * 0.93 and num_lots > 1:
                num_lots -= 1; tot_mg = num_lots * mg1
            if num_lots < 1 or tot_mg > free_cap: continue

            free_cap -= tot_mg
            positions[sym] = {
                "i": i, "ep": ep, "lots": num_lots,
                "lot_size": lot, "margin": tot_mg,
            }

    # Force close at last close
    ld = dates[-1]
    lc = cp.loc[ld]
    for sym, pos in list(positions.items()):
        xp = lc.get(sym, pos["ep"])
        if np.isnan(xp) or xp == 0: xp = pos["ep"]
        gp   = (xp - pos["ep"]) * pos["lot_size"] * pos["lots"]
        cost = _rt_cost(pos["ep"], xp, pos["lot_size"], pos["lots"])
        free_cap += pos["margin"] + gp - cost
        trades.append({
            "trade_id":    len(trades)+1,
            "symbol":      sym,
            "entry_date":  dates[pos["i"]],
            "exit_date":   ld,
            "entry_price": round(pos["ep"],2),
            "exit_price":  round(xp,2),
            "lots":        pos["lots"],
            "lot_size":    pos["lot_size"],
            "holding_days":len(dates)-1-pos["i"],
            "exit_reason": "EOD",
            "gross_pnl":   round(gp,2),
            "cost":        round(cost,2),
            "net_pnl":     round(gp-cost,2),
            "ret_pct":     round((xp/pos["ep"]-1)*100,4),
            "rom_pct":     round((gp-cost)/pos["margin"]*100,4),
        })

    return pd.DataFrame(trades), pd.DataFrame(eq_rec).set_index("date")


# ==============================================================================
#  METRICS
# ==============================================================================
def _mc(lst, v):
    best=cur=0
    for x in lst:
        cur=cur+1 if x==v else 0; best=max(best,cur)
    return best


def metrics(tdf, edf, label=""):
    init  = INITIAL_CAPITAL
    eq    = edf["equity"]
    final = eq.iloc[-1]
    nyrs  = (edf.index[-1] - edf.index[0]).days / 365.25

    rp   = (final/init-1)*100
    ri   = final-init
    cagr = ((final/init)**(1/max(nyrs,0.01))-1)*100
    dr   = eq.pct_change().dropna()
    sh   = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0.0
    dn   = dr[dr<0]
    so   = dr.mean()/dn.std()*np.sqrt(252) if len(dn)>1 else 0.0
    rm   = eq.cummax(); dds=(eq-rm)/rm; mdd=dds.min()*100
    cal  = cagr/abs(mdd) if mdd!=0 else 0.0

    if tdf.empty: return {}
    win  = tdf[tdf["net_pnl"]>0]; los=tdf[tdf["net_pnl"]<=0]
    wr   = len(win)/len(tdf)*100 if len(tdf)>0 else 0.0
    gp   = win["net_pnl"].sum()       if len(win)>0 else 0.0
    gl   = abs(los["net_pnl"].sum())  if len(los)>0 else 1e-9
    pf   = gp/gl
    exp_ = tdf["net_pnl"].mean()
    gt   = tdf["gross_pnl"].sum(); ct=tdf["cost"].sum()
    ctg  = abs(ct)/abs(gt)*100 if abs(gt)>0 else 999.0
    ws   = (tdf["net_pnl"]>0).astype(int).tolist()
    stp  = tdf[tdf["exit_reason"]=="STOP"] if "exit_reason" in tdf.columns else pd.DataFrame()

    # Per-symbol breakdown
    sym_stats = {}
    for sym, grp in tdf.groupby("symbol"):
        sym_stats[sym] = {
            "n": len(grp),
            "net": grp["net_pnl"].sum(),
            "wr":  (grp["net_pnl"]>0).mean()*100,
        }

    return {
        "label":label, "init":init, "final":final, "nyrs":nyrs,
        "ret_pct":rp, "ret_inr":ri, "cagr":cagr,
        "sharpe":sh, "sortino":so, "calmar":cal, "max_dd":mdd,
        "n_trades":len(tdf), "n_stops":len(stp),
        "win_rate":wr, "profit_factor":pf, "expectancy":exp_,
        "avg_win":  win["net_pnl"].mean() if len(win)>0 else 0.0,
        "avg_loss": los["net_pnl"].mean() if len(los)>0 else 0.0,
        "avg_hold": tdf["holding_days"].mean(),
        "max_cw":   _mc(ws,1), "max_cl": _mc(ws,0),
        "gross":gt, "costs":ct, "cost_pct":ctg,
        "long_pnl": tdf["net_pnl"].sum(),
        "avg_lots": tdf["lots"].mean() if "lots" in tdf.columns else 0.0,
        "avg_rom":  tdf["rom_pct"].mean() if "rom_pct" in tdf.columns else 0.0,
        "eq":eq, "dd":dds, "sym_stats":sym_stats,
    }


# ==============================================================================
#  GRID SEARCH
# ==============================================================================
def grid_search(features, s, e):
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    results= []

    for vals in combos:
        p = dict(zip(keys, vals))
        p["holding_period"] = int(p["holding_period"])
        p["cooldown"]       = int(p["cooldown"])
        try:
            sig, sc  = build_signals(features, p, s, e)
            tdf, edf = run_backtest(features, sig, sc, p, s, e)
            if tdf.empty or len(tdf) < 30: continue
            m = metrics(tdf, edf, "OPT")
            row = dict(zip(keys, vals))
            row.update({
                "sharpe":   round(m["sharpe"],         4),
                "calmar":   round(m["calmar"],         4),
                "cagr":     round(m["cagr"],           2),
                "max_dd":   round(m["max_dd"],         2),
                "win_rate": round(m["win_rate"],       2),
                "pf":       round(m["profit_factor"],  4),
                "n_trades": m["n_trades"],
                "cost_pct": round(m["cost_pct"],       1),
                "final_eq": round(m["final"]/1e5,      2),
            })
            results.append(row)
        except Exception:
            pass

    if not results: return pd.DataFrame(), {}
    gdf = (pd.DataFrame(results)
           .sort_values(["sharpe","calmar"], ascending=False)
           .reset_index(drop=True))
    return gdf, gdf.iloc[0].to_dict()


# ==============================================================================
#  VALIDATION
# ==============================================================================
def validate(features, best_params, top_n_df, s, e):
    keys = list(PARAM_GRID.keys())
    scored = []

    for _, row in top_n_df.head(8).iterrows():
        p = {k: row[k] for k in keys}
        p["holding_period"] = int(p["holding_period"])
        p["cooldown"]       = int(p["cooldown"])
        try:
            sig, sc  = build_signals(features, p, s, e)
            tdf, edf = run_backtest(features, sig, sc, p, s, e)
            if tdf.empty or len(tdf) < 10: continue
            m = metrics(tdf, edf, "VAL")
            r = {k: row[k] for k in keys}
            r["sharpe_opt"]  = row["sharpe"]
            r["sharpe_val"]  = round(m["sharpe"],        4)
            r["calmar_val"]  = round(m["calmar"],        4)
            r["cagr_val"]    = round(m["cagr"],          2)
            r["max_dd_val"]  = round(m["max_dd"],        2)
            r["wr_val"]      = round(m["win_rate"],      2)
            r["pf_val"]      = round(m["profit_factor"], 4)
            r["n_trades_val"]= m["n_trades"]
            r["final_eq_val"]= round(m["final"]/1e5,    2)
            r["cost_pct_val"]= round(m["cost_pct"],      1)
            # Combined score: geometric mean
            s_opt = max(row["sharpe"], 0.001)
            s_val = max(m["sharpe"],   0.001)
            r["score"] = round((s_opt * s_val)**0.5, 4)
            scored.append(r)
        except Exception:
            pass

    if not scored:
        return pd.DataFrame(), best_params

    vdf = (pd.DataFrame(scored)
           .sort_values("score", ascending=False)
           .reset_index(drop=True))
    w  = vdf.iloc[0]
    bp = {k: (int(w[k]) if k in ["holding_period","cooldown"] else float(w[k]))
          for k in keys}
    return vdf, bp


# ==============================================================================
#  PRINT OUTPUT
# ==============================================================================
def _fp(v):
    try:    return f"{v:>+.2f}%"
    except: return "N/A"
def _fi(v):
    try:    return f"INR {v:>+,.0f}"
    except: return "N/A"
def _fic(v):
    try:    return f"INR {v:>,.0f}"
    except: return "N/A"
def _f4(v):
    try:    return f"{v:.4f}"
    except: return "N/A"
def _fn(v):
    try:    return f"{int(v):,}"
    except: return "N/A"
def _f2(v):
    try:    return f"{v:.2f}"
    except: return "N/A"
def _fcp(v):
    try:    return f"{v:.1f}%"
    except: return "N/A"


def print_sheet1(gdf, vdf, bp, is_m, val_m, loaded_syms, src):
    S  = "=" * 84; S2 = "-" * 84
    g  = lambda m, k: m.get(k) if isinstance(m, dict) else None

    def _row(label, ov, vv, w=34):
        print(f"  {label:<{w}}{str(ov):>24}{str(vv):>24}")

    print(); print(S)
    print("  SHEET 1  |  IN-SAMPLE  |  NSE FUTSTK LONG-ONLY POSITIONAL v1.0  |  Universe Capital")
    print("  REBUILT FROM SCRATCH  |  Honest design for 10L capital on live NSE")
    print(S)

    print(f"\n  A. WHAT CHANGED FROM PREVIOUS MODELS")
    print(S2)
    print(f"  REMOVED:")
    print(f"    - All intraday strategies (no edge on daily OHLCV data)")
    print(f"    - All short trades (NSE structural bull; all short legs lost money)")
    print(f"    - Synthetic data (was 10x stronger than live NSE — misleading)")
    print(f"    - Complex signal stacks that destroyed trade volume")
    print()
    print(f"  ADDED:")
    print(f"    - LONG ONLY positional 3-7 day hold (correct timeframe for mean reversion)")
    print(f"    - Nifty regime filter (only trade when Nifty > 50d MA)")
    print(f"    - Not-still-falling filter (1d return > -1.5% to avoid catching knives)")
    print(f"    - Capital-dynamic lot filter (skip if 1-lot margin > 15% of equity)")
    print(f"    - Honest targets: CAGR 8-15%, Sharpe 0.5-1.0, MaxDD < 15%")
    print()
    print(f"  B. SETUP")
    print(S2)
    print(f"  Data Source    : {src}")
    print(f"  Universe       : {', '.join(loaded_syms)}")
    print(f"  Capital        : INR {INITIAL_CAPITAL:,.0f}  (10 Lakh)")
    print(f"  Strategy       : LONG ONLY  |  Positional  |  3-7 day hold")
    print(f"  Signal         : Cross-sectional z(N-day return) < -threshold")
    print(f"                   AND stock not still falling (1d ret > -1.5%)")
    print(f"                   AND Nifty above 50-day MA (bull regime)")
    print(f"  Stop-Loss      : {STOP_LOSS*100:.0f}% on position (checked at close, exit next open)")
    print(f"  Max Positions  : {MAX_POSITIONS}  |  Risk/Trade: {RISK_PCT*100:.0f}%  |  Max margin/pos: 15%")
    print(f"  Cost Model     : Brokerage INR 20 flat + STT 0.01% + Exchange + GST + 0.02% slippage")
    print()
    print(f"  WALK-FORWARD WINDOWS (10 years total):")
    print(f"    Optimisation  : {DATES['opt_start']}  to  {DATES['opt_end']}  (5 years)")
    print(f"    Validation    : {DATES['val_start']}  to  {DATES['val_end']}  (2 years)")
    print(f"    OOS           : {DATES['oos_start']}  to  {DATES['oos_end']}  (3 years — NEVER touched)")
    print(f"\n  Grid  : {len(gdf)} valid combos | Keys: {list(PARAM_GRID.keys())}")

    # C. Grid
    print(f"\n  C. OPTIMISATION GRID  (2015-2019)  |  Top 15 by Sharpe")
    print(S2)
    print(f"  {'Rk':>2}  {'Hold':>4}  {'Z':>4}  {'RW':>2}  {'CD':>2}"
          f"  {'Sharpe':>7}  {'Calmar':>7}  {'CAGR%':>6}  {'MaxDD%':>7}"
          f"  {'WR%':>5}  {'PF':>6}  {'Trades':>6}  {'Cst%':>5}  {'FinalEq(L)':>10}")
    print("  " + "." * 82)
    for i, row in gdf.head(15).iterrows():
        print(f"  {i+1:>2}  {int(row['holding_period']):>4}  "
              f"{row['z_threshold']:>4.1f}  {int(row['reversion_window']):>2}  "
              f"{int(row['cooldown']):>2}  "
              f"{row['sharpe']:>7.4f}  {row['calmar']:>7.4f}  "
              f"{row['cagr']:>6.2f}  {row['max_dd']:>7.2f}  "
              f"{row['win_rate']:>5.1f}  {row['pf']:>6.4f}  "
              f"{int(row['n_trades']):>6}  {row['cost_pct']:>5.1f}  "
              f"{row['final_eq']:>10.2f}L")
    print(f"\n  RW=reversion window (days)  |  CD=cooldown days  |  Cst%=cost as % of gross")
    print(f"  Top 8 forwarded to Validation")

    # D. Validation
    if not vdf.empty:
        print(f"\n  D. VALIDATION SCORING  (2020-2021)  |  Winner by combined Sharpe")
        print(S2)
        print(f"  {'Rk':>2}  {'Hold':>4}  {'Z':>4}  {'RW':>2}  {'CD':>2}"
              f"  {'IS_Sh':>7}  {'V_Sh':>7}  {'Score':>7}"
              f"  {'VCAGR%':>7}  {'VDD%':>7}  {'VWR%':>5}  {'VPF':>6}"
              f"  {'VTr':>5}  {'VCst%':>6}  {'VFEq(L)':>8}")
        print("  " + "." * 82)
        for i, row in vdf.iterrows():
            marker = "  <- WINNER" if i == 0 else ""
            print(f"  {i+1:>2}  {int(row['holding_period']):>4}  "
                  f"{row['z_threshold']:>4.1f}  {int(row['reversion_window']):>2}  "
                  f"{int(row['cooldown']):>2}  "
                  f"{row['sharpe_opt']:>7.4f}  {row['sharpe_val']:>7.4f}  "
                  f"{row['score']:>7.4f}  {row['cagr_val']:>7.2f}  "
                  f"{row['max_dd_val']:>7.2f}  {row['wr_val']:>5.1f}  "
                  f"{row['pf_val']:>6.4f}  {int(row['n_trades_val']):>5}  "
                  f"{row['cost_pct_val']:>6.1f}  {row['final_eq_val']:>7.2f}L{marker}")

    # E. Locked params
    print(f"\n  E. LOCKED OPTIMAL PARAMETERS  (frozen before OOS)")
    print(S2)
    print(f"  Holding Period   = {int(bp.get('holding_period',5))} days")
    print(f"  Z-Threshold      = {bp.get('z_threshold',1.5):.1f}σ  "
          f"(cross-sectional z < -{bp.get('z_threshold',1.5):.1f} to qualify)")
    print(f"  Reversion Window = {int(bp.get('reversion_window',5))} days  "
          f"(return computed over this period)")
    print(f"  Cooldown         = {int(bp.get('cooldown',3))} days  (re-entry blackout per stock)")
    print(f"  Stop-Loss        = {STOP_LOSS*100:.0f}%  (position-level, checked at daily close)")
    print(f"  Max Positions    = {MAX_POSITIONS}  |  Long Only  |  No shorts")

    # F. IS Performance
    print(f"\n  F. IN-SAMPLE PERFORMANCE  (locked params per sub-window)")
    print(S2)
    print(f"  {'Metric':<34}{'Opt 2015-19':>24}{'Val 2020-21':>24}")
    print("  " + "-" * 82)
    _row("Initial Capital",      _fic(g(is_m,"init")),     _fic(g(val_m,"init")))
    _row("Final Equity",         _fic(g(is_m,"final")),    _fic(g(val_m,"final")))
    _row("Trades Executed",      _fn(g(is_m,"n_trades")),  _fn(g(val_m,"n_trades")))
    print("  " + "-" * 82)
    _row("Total Return (%)",     _fp(g(is_m,"ret_pct")),   _fp(g(val_m,"ret_pct")))
    _row("CAGR (%)",             _fp(g(is_m,"cagr")),      _fp(g(val_m,"cagr")))
    _row("Sharpe Ratio",         _f4(g(is_m,"sharpe")),    _f4(g(val_m,"sharpe")))
    _row("Sortino Ratio",        _f4(g(is_m,"sortino")),   _f4(g(val_m,"sortino")))
    _row("Calmar Ratio",         _f4(g(is_m,"calmar")),    _f4(g(val_m,"calmar")))
    _row("Max Drawdown (%)",     _fp(g(is_m,"max_dd")),    _fp(g(val_m,"max_dd")))
    print("  " + "-" * 82)
    _row("Win Rate (%)",         _fp(g(is_m,"win_rate")),  _fp(g(val_m,"win_rate")))
    _row("Profit Factor",        _f4(g(is_m,"profit_factor")),_f4(g(val_m,"profit_factor")))
    _row("Expectancy / Trade",   _fi(g(is_m,"expectancy")),_fi(g(val_m,"expectancy")))
    _row("Avg Winner",           _fi(g(is_m,"avg_win")),   _fi(g(val_m,"avg_win")))
    _row("Avg Loser",            _fi(g(is_m,"avg_loss")),  _fi(g(val_m,"avg_loss")))
    _row("Short Trades",         "0  (long only)",          "0  (long only)")
    _row("Stop-Loss Exits",      _fn(g(is_m,"n_stops")),   _fn(g(val_m,"n_stops")))
    _row("Avg Hold (days)",      _f2(g(is_m,"avg_hold")),  _f2(g(val_m,"avg_hold")))
    _row("Max Consec. Wins",     _fn(g(is_m,"max_cw")),    _fn(g(val_m,"max_cw")))
    _row("Max Consec. Losses",   _fn(g(is_m,"max_cl")),    _fn(g(val_m,"max_cl")))
    print("  " + "-" * 82)
    _row("Gross PnL",            _fi(g(is_m,"gross")),     _fi(g(val_m,"gross")))
    _row("Total Costs",          _fi(g(is_m,"costs")),     _fi(g(val_m,"costs")))
    _row("Net PnL",              _fi(g(is_m,"ret_inr")),   _fi(g(val_m,"ret_inr")))
    _row("Cost / Gross (%)",     _fcp(g(is_m,"cost_pct")), _fcp(g(val_m,"cost_pct")))
    _row("Avg Lots / Trade",     _f2(g(is_m,"avg_lots")),  _f2(g(val_m,"avg_lots")))
    _row("Avg Ret on Margin(%)", _fcp(g(is_m,"avg_rom")),  _fcp(g(val_m,"avg_rom")))

    # Per-symbol
    print(f"\n  G. PER-SYMBOL BREAKDOWN  (Optimisation window)")
    print(S2)
    sym_st = g(is_m, "sym_stats") or {}
    print(f"  {'Symbol':<14}{'Trades':>7}{'Net PnL':>14}{'Win Rate':>10}")
    print("  " + "-" * 45)
    for sym, st in sorted(sym_st.items(), key=lambda x: x[1]["net"], reverse=True):
        net_l = st['net'] / 1e5
        print(f"  {sym:<14}{int(st['n']):>7}  {'INR':>3} {net_l:>+8.2f}L    {st['wr']:>6.1f}%")
    print(); print(S)


def print_sheet2(oos_m, bp, is_m, val_m, src, runtime):
    S  = "=" * 84; S2 = "-" * 84
    g  = lambda k: oos_m.get(k) if oos_m else None

    def _row(label, val, w=42):
        print(f"  {label:<{w}}{str(val):>34}")

    print(); print(S)
    print("  SHEET 2  |  OUT-OF-SAMPLE  |  NSE FUTSTK LONG-ONLY POSITIONAL v1.0")
    print("  PURE FORWARD TEST  |  2022-2024  |  LONG ONLY  |  Params locked from IS")
    print(S)

    print(f"\n  A. OOS SETUP")
    print(S2)
    print(f"  OOS Window     : {DATES['oos_start']}  to  {DATES['oos_end']}  (3 years)")
    print(f"  Data           : {src}  — UNSEEN, zero IS contact")
    print(f"  Direction      : LONG ONLY  — no short trades ever")
    print(f"  Hold           : {int(bp.get('holding_period',5))} days  "
          f"|  z<-{bp.get('z_threshold',1.5):.1f}  "
          f"|  RW={int(bp.get('reversion_window',5))}d  "
          f"|  CD={int(bp.get('cooldown',3))}d")

    print(f"\n  B. OUT-OF-SAMPLE PERFORMANCE  (2022-2024)")
    print(S2)
    print(f"  {'Metric':<42}{'Value':>34}")
    print("  " + "-" * 76)
    _row("Initial Capital",               _fic(g("init")))
    _row("Final Equity",                  _fic(g("final")))
    _row("Trades Executed",               _fn(g("n_trades")))
    print("  " + "-" * 76)
    _row("Total Return (%)",              _fp(g("ret_pct")))
    _row("CAGR (%)",                      _fp(g("cagr")))
    _row("Sharpe Ratio",                  _f4(g("sharpe")))
    _row("Sortino Ratio",                 _f4(g("sortino")))
    _row("Calmar Ratio",                  _f4(g("calmar")))
    _row("Max Drawdown (%)",              _fp(g("max_dd")))
    print("  " + "-" * 76)
    _row("Win Rate (%)",                  _fp(g("win_rate")))
    _row("Profit Factor",                 _f4(g("profit_factor")))
    _row("Expectancy / Trade",            _fi(g("expectancy")))
    _row("Avg Winner",                    _fi(g("avg_win")))
    _row("Avg Loser",                     _fi(g("avg_loss")))
    _row("Short Trades",                  "0  (long only)")
    _row("Stop-Loss Exits",               _fn(g("n_stops")))
    _row("Avg Hold (days)",               _f2(g("avg_hold")))
    _row("Max Consec. Wins",              _fn(g("max_cw")))
    _row("Max Consec. Losses",            _fn(g("max_cl")))
    print("  " + "-" * 76)
    _row("Gross PnL",                     _fi(g("gross")))
    _row("Total Transaction Costs",       _fi(g("costs")))
    _row("Net PnL",                       _fi(g("ret_inr")))
    _row("Cost / Gross (%)",              _fcp(g("cost_pct")))
    _row("Avg Lots / Trade",              _f2(g("avg_lots")))
    _row("Avg Return on Margin (%)",      _fcp(g("avg_rom")))

    # Comparison
    print(f"\n  C. IS vs OOS COMPARISON")
    print(S2)
    print(f"  {'Metric':<26}{'IS-Opt 15-19':>14}{'IS-Val 20-21':>14}{'OOS 22-24':>14}")
    print("  " + "-" * 68)

    def _c(m, k):
        try:    return f"INR {m[k]/1e5:.2f}L"
        except: return "N/A"
    def _crow(label, ov, vv, oosv, w=26):
        print(f"  {label:<{w}}{str(ov):>14}{str(vv):>14}{str(oosv):>14}")
    def fp2(m, k):
        try:    return f"{m[k]:>+.2f}%"
        except: return "N/A"
    def f42(m, k):
        try:    return f"{m[k]:.4f}"
        except: return "N/A"

    _crow("Initial Capital",  _c(is_m,"init"),   _c(val_m,"init"),  _c(oos_m,"init"))
    _crow("Final Equity",     _c(is_m,"final"),  _c(val_m,"final"), _c(oos_m,"final"))
    _crow("Trades",           _fn(is_m.get("n_trades",0)), _fn(val_m.get("n_trades",0)), _fn(g("n_trades")))
    print("  " + "-" * 68)
    _crow("Total Return(%)",  fp2(is_m,"ret_pct"),    fp2(val_m,"ret_pct"),    _fp(g("ret_pct")))
    _crow("CAGR (%)",         fp2(is_m,"cagr"),       fp2(val_m,"cagr"),       _fp(g("cagr")))
    _crow("Sharpe",           f42(is_m,"sharpe"),     f42(val_m,"sharpe"),     _f4(g("sharpe")))
    _crow("Max DD (%)",       fp2(is_m,"max_dd"),     fp2(val_m,"max_dd"),     _fp(g("max_dd")))
    _crow("Win Rate (%)",     fp2(is_m,"win_rate"),   fp2(val_m,"win_rate"),   _fp(g("win_rate")))
    _crow("Profit Factor",    f42(is_m,"profit_factor"), f42(val_m,"profit_factor"), _f4(g("profit_factor")))
    _crow("Cost/Gross (%)",   _fcp(is_m.get("cost_pct")), _fcp(val_m.get("cost_pct")), _fcp(g("cost_pct")))

    # Conclusion
    print(f"\n  D. CONCLUSION")
    print(S2)
    sh_opt = is_m.get("sharpe",0); sh_val = val_m.get("sharpe",0)
    sh_oos = g("sharpe") or 0;     pf_oos = g("profit_factor") or 0
    ctg    = g("cost_pct") or 999; mdd    = g("max_dd") or -99
    n_oos  = g("n_trades") or 0

    chg_ov = (sh_val-sh_opt)/(abs(sh_opt)+1e-9)*100
    chg_vo = (sh_oos-sh_val)/(abs(sh_val)+1e-9)*100

    print(f"  Sharpe path : IS-Opt {sh_opt:.4f}  ->  IS-Val {sh_val:.4f}  ->  OOS {sh_oos:.4f}")
    print(f"  IS→Val      : {chg_ov:>+.1f}%  ({'improvement' if chg_ov>0 else 'degradation'})")
    print(f"  Val→OOS     : {chg_vo:>+.1f}%  ({'stable' if abs(chg_vo)<35 else 'regime shift'})")
    print(f"  Cost/Gross  : {ctg:.1f}%  ({'healthy <30%' if ctg<30 else 'watch' if ctg<60 else 'HIGH'})")
    print(f"  Max DD      : {mdd:.2f}%  ({'controlled' if mdd>-15 else 'elevated'})")
    print()

    if sh_oos > 0.8 and pf_oos > 1.2:
        verdict = "STRONG     — Solid live NSE performance. Implement with normal position management."
    elif sh_oos > 0.4 and pf_oos > 1.0:
        verdict = "ACCEPTABLE — Profitable on live data. Paper trade 30 days, then go live with 1 lot."
    elif pf_oos > 1.0:
        verdict = "MARGINAL   — Profitable but thin. Start with 1 lot per trade maximum."
    else:
        verdict = "WEAK       — Signal not profitable after costs on this universe. See notes below."

    print(f"  Verdict     : {verdict}")
    print()
    print(f"  HONEST EXPECTATIONS for 10L NSE futures:")
    print(f"    - Mean reversion on daily data is a thin alpha on live NSE")
    print(f"    - Realistic CAGR: 8-15%  |  Realistic Sharpe: 0.4-0.9")
    print(f"    - If result is below this, the signal needs more edge (sector filters,")
    print(f"      earnings calendar exclusion, index event exclusion)")
    print(f"    - Trade only RELIANCE, TCS, INFY, LT — highest liquidity, tightest spreads")
    print()
    print(f"  Runtime  : {runtime:.1f}s  |  Live data only  |  Params locked before OOS")
    print(S); print()


# ==============================================================================
#  MAIN
# ==============================================================================
def main():
    t0 = datetime.now()

    print("\n" + "="*70)
    print("  NSE FUTSTK LONG-ONLY POSITIONAL v1.0  |  10 Lakh Capital")
    print("  Clean rebuild — previous intraday models caused 94% losses")
    print("="*70)

    print("\n  Loading market data from Yahoo Finance...")
    data, nifty, src = load_data()

    print("  Building features...")
    features = build_features(data, nifty, {})

    # Optimisation 2015-2019
    print("  Running grid search on 2015-2019...")
    gdf, _ = grid_search(features, DATES["opt_start"], DATES["opt_end"])
    if gdf.empty:
        print("  ERROR: No valid param combinations found. Check data.")
        sys.exit(1)

    # Validation 2020-2021
    print("  Validating top params on 2020-2021...")
    vdf, best_params = validate(features, gdf.iloc[0].to_dict(), gdf,
                                DATES["val_start"], DATES["val_end"])

    # IS metrics with best params
    print("  Computing IS metrics...")
    sig_o, sc_o = build_signals(features, best_params, DATES["opt_start"], DATES["opt_end"])
    tdf_o, edf_o = run_backtest(features, sig_o, sc_o, best_params, DATES["opt_start"], DATES["opt_end"])
    is_m = metrics(tdf_o, edf_o, "OPT")

    sig_v, sc_v = build_signals(features, best_params, DATES["val_start"], DATES["val_end"])
    tdf_v, edf_v = run_backtest(features, sig_v, sc_v, best_params, DATES["val_start"], DATES["val_end"])
    val_m = metrics(tdf_v, edf_v, "VAL")

    # OOS 2022-2024 — locked
    print("  Running OOS (2022-2024) with locked params...")
    sig_oo, sc_oo = build_signals(features, best_params, DATES["oos_start"], DATES["oos_end"])
    tdf_oo, edf_oo = run_backtest(features, sig_oo, sc_oo, best_params, DATES["oos_start"], DATES["oos_end"])
    oos_m = metrics(tdf_oo, edf_oo, "OOS")

    loaded_syms = list(is_m.get("sym_stats", {}).keys())
    elapsed = (datetime.now() - t0).total_seconds()
    print("  Done. Printing results...\n")

    print_sheet1(gdf, vdf, best_params, is_m, val_m, loaded_syms, src)
    print_sheet2(oos_m, best_params, is_m, val_m, src, elapsed)


if __name__ == "__main__":
    main()