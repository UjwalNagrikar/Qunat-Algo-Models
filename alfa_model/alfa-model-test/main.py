#!/usr/bin/env python3
"""
================================================================================
  NSE FUTSTK BACKTESTING ENGINE  v5.0
  Strategy : Cross-Sectional Mean Reversion  |  Futures (FUTSTK) Mechanics
  Universe : 12 NSE Large-Cap Stocks  |  2010 - 2024

  IS / OOS TWO-PHASE DESIGN
  ─────────────────────────────────────────────────────────────────────────────
  Optimisation : 2010-2017  (8 yr)  grid-search  →  rank by Sharpe
  Validation   : 2018-2021  (4 yr)  top-N re-scored  →  pick winner by combined score
  OOS          : 2023-2024  (2 yr)  LOCKED params, pure unseen-data test

  v5.0 FIXES vs v4.0  (live NSE diagnostics)
  ─────────────────────────────────────────────────────────────────────────────
  FIX 1  MIN-TRADE GUARD
         Grid combos with < MIN_TRADES trades on Opt window are rejected.
         Previous: z=2.0 won with 22 trades → metrics are pure noise.
         v5.0: any combo below 80 trades is discarded before ranking.

  FIX 2  GRID EXPANDED
         z_threshold : [0.50, 0.75, 1.00, 1.50]  (removed 2.0, added 0.50)
         vol_threshold: [0.025, 0.035, 0.045]     (added 0.045 for NSE live)
         holding_period: [3, 5, 7, 10]
         cooldown_days : [3, 5]
         Total: 4x4x3x2 = 96 combinations

  FIX 3  TREND FILTER SOFTENED
         Default OFF. Real NSE cross-sectional dispersion is lower than
         synthetic, so trend + high-z eliminates almost all signals.
         Grid optionally tests with/without trend filter.

  FIX 4  OUTPUT — initial capital, final capital, trade count per phase
         Every phase now prints: start equity, end equity, trades executed.

  FIX 5  DEGRADATION LABEL CORRECTED
         IS-Opt < IS-Val Sharpe = good OOS generalisation, not overfitting.
         The label now correctly reads the direction.
================================================================================
"""
import warnings, itertools
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# ── NSE FUTSTK Lot Sizes ──────────────────────────────────────────────────────
FUTSTK_LOTS = {
    "RELIANCE":   250,  "TCS":        150,  "HDFCBANK":   550,
    "INFY":       300,  "ICICIBANK": 1375,  "HINDUNILVR": 300,
    "ITC":       3200,  "SBIN":      1500,  "AXISBANK":  1200,
    "BHARTIARTL": 950,  "LT":         150,  "KOTAKBANK":  400,
}

# ── Window Definitions ────────────────────────────────────────────────────────
DATES = {
    "data_start": "2010-01-01",
    "data_end":   "2024-12-31",
    "opt_start":  "2010-01-01",   # Optimisation window (8 yr)
    "opt_end":    "2017-12-31",
    "val_start":  "2018-01-01",   # Validation window (4 yr)
    "val_end":    "2021-12-31",
    "oos_start":  "2023-01-01",   # OOS window (2 yr) — NEVER used in IS
    "oos_end":    "2024-12-31",
}

# ── Master Config ─────────────────────────────────────────────────────────────
CFG = {
    "symbols": [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
        "HINDUNILVR.NS","ITC.NS","SBIN.NS","AXISBANK.NS",
        "BHARTIARTL.NS","LT.NS","KOTAKBANK.NS",
    ],

    # ── Grid (Optimisation window only) ──────────────────────────────────
    "param_grid": {
        "holding_period": [3, 5, 7, 10],
        "z_threshold":    [0.50, 0.75, 1.00, 1.50],   # removed 2.0 (too sparse on live NSE)
        "vol_threshold":  [0.025, 0.035, 0.045],       # added 0.045 for live NSE
        "cooldown_days":  [3, 5],
    },

    # ── FIX 1: minimum trades per optimisation window ─────────────────────
    # Any param combo generating fewer than this is statistically unreliable
    "min_trades_opt": 80,

    # ── Fixed params ──────────────────────────────────────────────────────
    "initial_capital":  1000000,
    "long_pct":         0.25,
    "short_pct":        0.25,
    "max_positions":    6,
    "margin_pct":       0.20,
    "risk_pct":         0.020,
    "max_margin_pct":   0.12,
    "use_trend_filter": False,   # FIX 3: OFF by default for live NSE
    "ma_window":        20,
    "vol_window":       10,
    "stop_loss_pct":    None,    # time-exit only
    "n_top_candidates": 10,

    # ── NSE Futures cost model ─────────────────────────────────────────────
    "brokerage_flat": 20.0,   "brokerage_pct": 0.0003,
    "stt_sell_pct":   0.0001, "exchange_pct":  0.000019,
    "sebi_pct":       0.000001,"stamp_buy_pct": 0.00002,
    "gst_pct":        0.18,   "slippage_pct":  0.0002,
}


# ==============================================================================
#  DATA
# ==============================================================================
def _try_yfinance(cfg):
    frames = []
    for sym in cfg["symbols"]:
        try:
            raw = yf.download(sym, start=DATES["data_start"], end=DATES["data_end"],
                              auto_adjust=True, progress=False, threads=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
            if raw.empty or len(raw) < 500:
                continue
            raw = raw[["open","high","low","close","volume"]].replace(0, np.nan)
            raw = raw.dropna(subset=["open","close"])
            raw.index = pd.to_datetime(raw.index)
            raw.index.name = "date"
            raw["symbol"] = sym.replace(".NS","")
            frames.append(raw)
        except Exception:
            pass
    return frames if len(frames) >= 8 else None


def _synthetic(cfg):
    np.random.seed(2010)
    dates = pd.bdate_range(DATES["data_start"], DATES["data_end"])
    T     = len(dates)
    syms  = [s.replace(".NS","") for s in cfg["symbols"]]
    mkt   = np.random.normal(0.00012, 0.008, T)
    for s,e,a in [(750,790,-0.004),(1700,1740,-0.003),
                  (2580,2640,-0.007),(2650,2700,0.005),(3000,3040,-0.002)]:
        mkt[s:e] += a + np.random.normal(0, abs(a)*0.4, e-s)
    seed = {
        "RELIANCE":1050,"TCS":750,"HDFCBANK":350,"INFY":650,"ICICIBANK":430,
        "HINDUNILVR":250,"ITC":120,"SBIN":250,"AXISBANK":260,
        "BHARTIARTL":320,"LT":1600,"KOTAKBANK":380,
    }
    frames = []
    for sym in syms:
        s0    = seed.get(sym, 500)
        idvol = np.random.uniform(0.009, 0.018)
        beta  = np.random.uniform(0.60, 1.40)
        drift = np.random.uniform(-0.00008, 0.00030)
        ac    = np.random.uniform(-0.45, -0.22)
        idio  = np.zeros(T)
        for t in range(1, T):
            idio[t] = ac*idio[t-1] + idvol*np.random.randn()
        ret   = mkt*beta + idio + drift
        close = s0 * np.exp(np.cumsum(ret))
        intra = np.abs(np.random.normal(0.008, 0.004, T)).clip(0.002, 0.05)
        open_ = np.concatenate([[s0], close[:-1]]) * np.exp(np.random.normal(0, 0.003, T))
        high  = np.maximum(open_, close) * (1 + intra*0.6)
        low   = np.minimum(open_, close) * (1 - intra*0.4)
        df = pd.DataFrame({
            "open":np.round(open_,2),"high":np.round(high,2),
            "low":np.round(low,2),"close":np.round(close,2),
            "volume":np.random.lognormal(15,0.9,T).astype(int),
            "symbol":sym
        }, index=dates)
        df.index.name = "date"
        frames.append(df)
    return frames


def load_data(cfg):
    frames = _try_yfinance(cfg)
    live   = frames is not None
    if not live:
        frames = _synthetic(cfg)
    data = (pd.concat(frames).reset_index()
            .set_index(["date","symbol"]).sort_index()
            .replace(0, np.nan).dropna(subset=["open","close"]))
    src = "Yahoo Finance (live)" if live else "Synthetic (negative-AC AR1, NSE price levels)"
    return data, src


# ==============================================================================
#  FEATURES  (computed once on full history, sliced per window)
# ==============================================================================
def build_features(data, cfg):
    close  = data["close"].unstack("symbol").sort_index().ffill(limit=3)
    open_  = data["open"].unstack("symbol").sort_index().ffill(limit=3)
    ret1   = close.pct_change(1)
    ret3   = close.pct_change(3)
    vol10  = ret1.rolling(cfg["vol_window"]).std()
    ma     = close.rolling(cfg["ma_window"]).mean()

    def _csz(df):
        mu  = df.mean(axis=1)
        sig = df.std(axis=1).replace(0, np.nan)
        return df.sub(mu, axis=0).div(sig, axis=0)

    z_comp  = (_csz(ret1) + _csz(ret3)) / 2.0
    cs_rank = ret1.rank(axis=1, ascending=True, pct=True)
    return {"close":close,"open":open_,"vol10":vol10,"ma":ma,
            "z_comp":z_comp,"cs_rank":cs_rank}


# ==============================================================================
#  SIGNAL GENERATION
# ==============================================================================
def build_signals(features, params, s_date, e_date):
    def _w(df): return df.loc[(df.index >= s_date) & (df.index <= e_date)].copy()

    rnk  = _w(features["cs_rank"])
    zc   = _w(features["z_comp"])
    v10  = _w(features["vol10"])
    cl   = _w(features["close"])
    ma   = _w(features["ma"])

    lp = params.get("long_pct",         0.25)
    sp = params.get("short_pct",        0.25)
    vt = params.get("vol_threshold",   0.035)
    zt = params.get("z_threshold",      0.75)
    tf = params.get("use_trend_filter", False)

    hi   = v10 > vt
    rnk2 = rnk.where(~hi)
    zc2  = zc.where(~hi)

    lc = (rnk2 <= lp)       & (zc2 <= -zt)
    sc = (rnk2 >= (1 - sp)) & (zc2 >=  zt)

    if tf:
        lc = lc & (cl > ma).fillna(False)
        sc = sc & (cl < ma).fillna(False)

    sig = (lc.fillna(False).astype(np.int8)
           - sc.fillna(False).astype(np.int8))
    return sig, zc2.abs()


# ==============================================================================
#  COST MODEL
# ==============================================================================
def _rt_cost(ep, xp, lot, nl, direction, cfg):
    ten = ep*lot*nl
    tex = xp*lot*nl
    def _oc(t, side):
        b = min(cfg["brokerage_flat"], t*cfg["brokerage_pct"])
        e = t*cfg["exchange_pct"]
        s = t*cfg["sebi_pct"]
        g = (b+e)*cfg["gst_pct"]
        stamp = t*cfg["stamp_buy_pct"] if side == "buy"  else 0.0
        stt   = t*cfg["stt_sell_pct"]  if side == "sell" else 0.0
        return b+e+s+g+stamp+stt
    if direction == 1:
        c = _oc(ten,"buy") + _oc(tex,"sell")
    else:
        c = _oc(ten,"sell") + _oc(tex,"buy")
    return c + (ten+tex)*cfg["slippage_pct"]


# ==============================================================================
#  BACKTEST ENGINE
# ==============================================================================
def _trec(tid, sym, pos, dates, i, rsn, xp, gp, cost):
    net = gp - cost
    return {
        "trade_id":     tid,     "symbol":       sym,
        "direction":    "LONG" if pos["dir"]==1 else "SHORT",
        "entry_date":   dates[pos["i"]],  "exit_date": dates[i],
        "entry_price":  round(pos["ep"],2),"exit_price": round(xp,2),
        "lot_size":     pos["lot"],        "num_lots":   pos["nl"],
        "holding_days": i - pos["i"],      "exit_reason":rsn,
        "margin_used":  round(pos["margin"],2),
        "gross_pnl":    round(gp,2),  "cost":    round(cost,2),
        "net_pnl":      round(net,2),
        "ret_pct":      round((xp/pos["ep"]-1)*pos["dir"]*100, 4),
        "rom_pct":      round(net/pos["margin"]*100, 4) if pos["margin"]>0 else 0.0,
    }


def run_backtest(features, signals, ret_score, params, s_date, e_date):
    def _w(df): return df.loc[(df.index >= s_date) & (df.index <= e_date)]

    op  = _w(features["open"])
    cp  = _w(features["close"])
    sig = _w(signals)
    rs  = _w(ret_score)

    dates    = op.index.tolist()
    init_cap = float(params["initial_capital"])
    free_cap = init_cap
    positions = {}; cooldown = {}; trades = []; eq_rec = []

    hp    = int(params.get("holding_period",  3))
    sl    = params.get("stop_loss_pct",    None)
    maxp  = int(params.get("max_positions",   6))
    cd    = int(params.get("cooldown_days",   3))
    m_pct = params.get("margin_pct",       0.20)
    r_pct = params.get("risk_pct",         0.015)
    mm    = params.get("max_margin_pct",   0.12)

    for i, date in enumerate(dates):
        dop = op.loc[date]
        dcp = cp.loc[date]

        # Mark-to-market
        locked = unreal = 0.0
        for sym, pos in positions.items():
            locked += pos["margin"]
            px = dcp.get(sym, np.nan)
            if not np.isnan(px):
                unreal += (px - pos["ep"]) * pos["lot"] * pos["nl"] * pos["dir"]

        cur_eq = free_cap + locked + unreal
        eq_rec.append({"date": date, "equity": cur_eq, "n_open": len(positions)})

        # Optional stop-loss
        if sl is not None:
            for sym in list(positions):
                pos = positions[sym]
                px  = dcp.get(sym, np.nan)
                if np.isnan(px): continue
                tr = (px - pos["ep"]) / pos["ep"] * pos["dir"]
                if tr < -sl:
                    positions.pop(sym)
                    gp   = (px - pos["ep"]) * pos["lot"] * pos["nl"] * pos["dir"]
                    cost = _rt_cost(pos["ep"],px,pos["lot"],pos["nl"],pos["dir"],params)
                    free_cap += pos["margin"] + gp - cost
                    cooldown[sym] = i
                    trades.append(_trec(len(trades)+1,sym,pos,dates,i,"STOP",px,gp,cost))

        # Time exit
        for sym in [s for s,p in list(positions.items()) if i >= p["i"] + hp]:
            pos = positions.pop(sym)
            xp  = dop.get(sym, pos["ep"])
            if np.isnan(xp) or xp == 0: xp = pos["ep"]
            gp   = (xp - pos["ep"]) * pos["lot"] * pos["nl"] * pos["dir"]
            cost = _rt_cost(pos["ep"],xp,pos["lot"],pos["nl"],pos["dir"],params)
            free_cap += pos["margin"] + gp - cost
            cooldown[sym] = i
            trades.append(_trec(len(trades)+1,sym,pos,dates,i,"HOLD",xp,gp,cost))

        # Entry
        if i == 0: continue
        prev = dates[i-1]
        if prev not in sig.index: continue
        sr    = sig.loc[prev]
        cands = sr[sr != 0]
        if cands.empty: continue
        if prev in rs.index:
            sc2   = rs.loc[prev][cands.index].fillna(0)
            cands = cands.loc[sc2.sort_values(ascending=False).index]

        for sym, direction in cands.items():
            if len(positions) >= maxp: break
            if sym in positions: continue
            if i - cooldown.get(sym, -9999) < cd: continue
            ep = dop.get(sym, np.nan)
            if np.isnan(ep) or ep <= 0: continue
            lot = FUTSTK_LOTS.get(sym, 500)
            mpl = ep * lot * m_pct
            if mpl <= 0: continue
            alloc    = min(cur_eq * r_pct, cur_eq * mm)
            num_lots = max(1, int(alloc / mpl))
            tot_mg   = num_lots * mpl
            while tot_mg > cur_eq * mm and num_lots > 1:
                num_lots -= 1; tot_mg = num_lots * mpl
            if tot_mg > free_cap * 0.93:
                num_lots = max(1, int(free_cap * 0.90 / mpl))
                tot_mg   = num_lots * mpl
            if tot_mg > free_cap or num_lots < 1: continue
            free_cap -= tot_mg
            positions[sym] = {"i":i,"ep":ep,"nl":num_lots,
                              "lot":lot,"dir":int(direction),"margin":tot_mg}

    # Force-close at last close
    ld = dates[-1]; lc = cp.loc[ld]
    for sym, pos in list(positions.items()):
        xp = lc.get(sym, pos["ep"])
        if np.isnan(xp) or xp == 0: xp = pos["ep"]
        gp   = (xp - pos["ep"]) * pos["lot"] * pos["nl"] * pos["dir"]
        cost = _rt_cost(pos["ep"],xp,pos["lot"],pos["nl"],pos["dir"],params)
        free_cap += pos["margin"] + gp - cost
        trades.append(_trec(len(trades)+1,sym,pos,dates,len(dates)-1,"EOD",xp,gp,cost))

    tdf = pd.DataFrame(trades)
    edf = pd.DataFrame(eq_rec).set_index("date")
    return tdf, edf


# ==============================================================================
#  METRICS
# ==============================================================================
def _mc(lst, v):
    best = cur = 0
    for x in lst:
        cur = cur+1 if x==v else 0; best = max(best, cur)
    return best


def compute_metrics(tdf, edf, params, label=""):
    init  = params["initial_capital"]
    eq    = edf["equity"]
    final = eq.iloc[-1]
    nyrs  = (edf.index[-1] - edf.index[0]).days / 365.25

    rp   = (final / init - 1) * 100
    ri   = final - init
    cagr = ((final / init) ** (1 / max(nyrs, 0.01)) - 1) * 100
    dr   = eq.pct_change().dropna()
    sh   = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
    dn   = dr[dr < 0]
    so   = dr.mean() / dn.std() * np.sqrt(252) if len(dn) > 1 else 0.0
    rm   = eq.cummax()
    dds  = (eq - rm) / rm
    mdd  = dds.min() * 100
    cal  = cagr / abs(mdd) if mdd != 0 else 0.0

    if tdf.empty:
        return {}

    win  = tdf[tdf["net_pnl"] > 0]
    los  = tdf[tdf["net_pnl"] <= 0]
    wr   = len(win) / len(tdf) * 100
    gp   = win["net_pnl"].sum()       if len(win) > 0 else 0.0
    gl   = abs(los["net_pnl"].sum())  if len(los) > 0 else 1e-9
    pf   = gp / gl
    exp_ = tdf["net_pnl"].mean()
    gt   = tdf["gross_pnl"].sum()
    ct   = tdf["cost"].sum()
    ctg  = abs(ct) / abs(gt) * 100 if gt != 0 else 999.0
    ws   = (tdf["net_pnl"] > 0).astype(int).tolist()
    lg   = tdf[tdf["direction"] == "LONG"]
    sh_  = tdf[tdf["direction"] == "SHORT"]
    stp  = (tdf[tdf["exit_reason"] == "STOP"]
            if "exit_reason" in tdf.columns else pd.DataFrame())

    return {
        "label":label, "initial_capital":init, "final_equity":final,
        "total_ret_pct":rp, "total_ret_inr":ri, "cagr":cagr,
        "sharpe":sh, "sortino":so, "calmar":cal, "max_dd":mdd,
        "n_trades":len(tdf), "n_long":len(lg), "n_short":len(sh_),
        "n_stops":len(stp), "win_rate":wr, "profit_factor":pf,
        "expectancy":exp_,
        "avg_win":  win["net_pnl"].mean() if len(win) > 0 else 0.0,
        "avg_loss": los["net_pnl"].mean() if len(los) > 0 else 0.0,
        "avg_hold": tdf["holding_days"].mean(),
        "max_cw":   _mc(ws,1), "max_cl": _mc(ws,0),
        "gross_pnl":gt, "total_costs":ct, "cost_to_gross":ctg,
        "long_pnl": lg["net_pnl"].sum(),
        "short_pnl":sh_["net_pnl"].sum(),
        "avg_lots": tdf["num_lots"].mean()   if "num_lots"  in tdf.columns else 0.0,
        "avg_rom":  tdf["rom_pct"].mean()    if "rom_pct"   in tdf.columns else 0.0,
    }


# ==============================================================================
#  OPTIMISATION  (2010-2017)
# ==============================================================================
def run_optimisation(features, cfg):
    pg     = cfg["param_grid"]
    keys   = list(pg.keys())
    combos = list(itertools.product(*[pg[k] for k in keys]))
    s, e   = DATES["opt_start"], DATES["opt_end"]
    min_t  = cfg.get("min_trades_opt", 80)
    results = []

    for vals in combos:
        p = {**cfg,
             **dict(zip(keys, vals)),
             "holding_period": int(vals[keys.index("holding_period")]),
             "cooldown_days":  int(vals[keys.index("cooldown_days")]),
             "stop_loss_pct":  None}
        try:
            sigs, sc   = build_signals(features, p, s, e)
            tdf, edf   = run_backtest(features, sigs, sc, p, s, e)
            # FIX 1: reject low-trade combos
            if tdf.empty or len(tdf) < min_t:
                continue
            m = compute_metrics(tdf, edf, p, "OPT")
            row = dict(zip(keys, vals))
            row.update({
                "sharpe_opt":   round(m["sharpe"],         4),
                "calmar_opt":   round(m["calmar"],         4),
                "cagr_opt":     round(m["cagr"],           2),
                "max_dd_opt":   round(m["max_dd"],         2),
                "wr_opt":       round(m["win_rate"],       2),
                "pf_opt":       round(m["profit_factor"],  4),
                "n_trades_opt": m["n_trades"],
                "cost_pct_opt": round(m["cost_to_gross"],  1),
                "final_eq_opt": round(m["final_equity"],   0),
            })
            results.append(row)
        except Exception:
            pass

    if not results:
        return pd.DataFrame(), []

    gdf = (pd.DataFrame(results)
           .sort_values(["sharpe_opt","calmar_opt"], ascending=False)
           .reset_index(drop=True))
    return gdf, gdf.head(cfg["n_top_candidates"]).to_dict("records")


# ==============================================================================
#  VALIDATION  (2018-2021)
# ==============================================================================
def run_validation(features, cfg, top_candidates):
    s, e   = DATES["val_start"], DATES["val_end"]
    pg_keys= list(cfg["param_grid"].keys())
    scored  = []

    for cand in top_candidates:
        p = {**cfg,
             **{k: v for k, v in cand.items() if k in pg_keys},
             "holding_period": int(cand["holding_period"]),
             "cooldown_days":  int(cand["cooldown_days"]),
             "stop_loss_pct":  None}
        try:
            sigs, sc = build_signals(features, p, s, e)
            tdf, edf = run_backtest(features, sigs, sc, p, s, e)
            if tdf.empty or len(tdf) < 5:
                continue
            m = compute_metrics(tdf, edf, p, "VAL")
            row = {k: v for k, v in cand.items() if k in pg_keys}
            row.update({
                "sharpe_opt":   cand.get("sharpe_opt", 0),
                "sharpe_val":   round(m["sharpe"],         4),
                "calmar_val":   round(m["calmar"],         4),
                "cagr_val":     round(m["cagr"],           2),
                "max_dd_val":   round(m["max_dd"],         2),
                "wr_val":       round(m["win_rate"],       2),
                "pf_val":       round(m["profit_factor"],  4),
                "n_trades_val": m["n_trades"],
                "final_eq_val": round(m["final_equity"],   0),
            })
            s_opt = max(cand.get("sharpe_opt", 0), 0.001)
            s_val = max(m["sharpe"], 0.001)
            row["combined_score"] = round((s_opt * s_val) ** 0.5, 4)
            scored.append(row)
        except Exception:
            pass

    if not scored:
        fallback = {**cfg, "holding_period":3, "z_threshold":0.75,
                    "vol_threshold":0.035, "cooldown_days":3, "stop_loss_pct":None}
        return pd.DataFrame(), fallback

    vdf = (pd.DataFrame(scored)
           .sort_values("combined_score", ascending=False)
           .reset_index(drop=True))
    w = vdf.iloc[0]
    bp = {**cfg,
          "holding_period": int(w["holding_period"]),
          "z_threshold":    float(w["z_threshold"]),
          "vol_threshold":  float(w["vol_threshold"]),
          "cooldown_days":  int(w["cooldown_days"]),
          "stop_loss_pct":  None}
    return vdf, bp


# ==============================================================================
#  SHEET 1  —  IN-SAMPLE
# ==============================================================================
def print_sheet1(opt_df, val_df, best_params, is_m, val_m, cfg, src):
    SEP  = "=" * 84
    SEP2 = "-" * 84

    def fp(v):
        try:    return f"{v:>+.2f}%"
        except: return "N/A"
    def fi(v):
        try:    return f"INR {v:>+,.0f}"
        except: return "N/A"
    def fic(v):
        try:    return f"INR {v:>,.0f}"
        except: return "N/A"
    def f4(v):
        try:    return f"{v:.4f}"
        except: return "N/A"
    def fn(v):
        try:    return f"{int(v):,}"
        except: return "N/A"
    def f2(v):
        try:    return f"{v:.2f}"
        except: return "N/A"
    def fcp(v):
        try:    return f"{v:.1f}%"
        except: return "N/A"

    g = lambda m, k: m.get(k) if isinstance(m, dict) else None

    def _row(label, ov, vv, w=34):
        print(f"  {label:<{w}}{str(ov):>22}{str(vv):>22}")

    print(); print(SEP)
    print("  SHEET 1  |  IN-SAMPLE  |  NSE FUTSTK ENGINE v5.0  |  Universe Capital")
    print(SEP)

    # A. Setup
    print(f"\n  A. SETUP & DATA")
    print(SEP2)
    print(f"  Data Source     : {src}")
    print(f"  Universe        : {len(cfg['symbols'])} stocks  |  "
          + "  ".join(s.replace('.NS','') for s in cfg['symbols']))
    print(f"  Initial Capital : INR {cfg['initial_capital']:,.0f}")
    print(f"  Margin Rate     : {cfg['margin_pct']*100:.0f}%  |  "
          f"Risk/Trade: {cfg['risk_pct']*100:.1f}%  |  Max Positions: {cfg['max_positions']}")
    print(f"  Signal          : Composite z-score = avg(z_1day + z_3day)")
    print(f"  Vol Filter      : 10-day rolling std cap")
    print(f"  Trend Filter    : {'20-day MA' if cfg['use_trend_filter'] else 'OFF (default for live NSE — avoids signal depletion)'}")
    print(f"  Stop-Loss       : Disabled — time exit only")
    print(f"  Futures PnL     : (Exit − Entry) × Lot × Num Lots × Direction")
    print(f"  Cost Model      : Brokerage + STT(0.01% sell) + Exchange + SEBI + GST + Slippage")
    print()
    print(f"  IN-SAMPLE WINDOWS:")
    print(f"    Optimisation  : {DATES['opt_start']}  to  {DATES['opt_end']}  [8 years]")
    print(f"    Validation    : {DATES['val_start']}  to  {DATES['val_end']}  [4 years]")
    print(f"  OOS WINDOW      : {DATES['oos_start']}  to  {DATES['oos_end']}  [2 years  NEVER touched in IS]")
    print(f"\n  Grid size    : {len(opt_df) if not opt_df.empty else 'N/A'} valid combinations "
          f"(min {cfg['min_trades_opt']} trades filter applied)")
    print(f"  Param keys   : {list(cfg['param_grid'].keys())}")

    # B. Optimisation Grid
    print(f"\n  B. OPTIMISATION GRID  (2010-2017)  |  Top 15 by Sharpe")
    print(SEP2)
    print(f"  {'Rk':>2}  {'Hold':>4}  {'Z-Thr':>5}  {'Vol%':>4}  {'CD':>2}"
          f"  {'Sharpe':>7}  {'Calmar':>7}  {'CAGR%':>6}  {'MaxDD%':>7}"
          f"  {'WR%':>5}  {'PF':>6}  {'Trades':>6}  {'Cost%':>5}  {'FinalEq(L)':>10}")
    print("  " + "." * 80)
    for i, row in opt_df.head(15).iterrows():
        feq_l = row["final_eq_opt"] / 100000
        print(f"  {i+1:>2}  {int(row['holding_period']):>4}  "
              f"{row['z_threshold']:>5.2f}  {row['vol_threshold']*100:>4.1f}  "
              f"{int(row['cooldown_days']):>2}  "
              f"{row['sharpe_opt']:>7.4f}  {row['calmar_opt']:>7.4f}  "
              f"{row['cagr_opt']:>6.2f}  {row['max_dd_opt']:>7.2f}  "
              f"{row['wr_opt']:>5.1f}  {row['pf_opt']:>6.4f}  "
              f"{int(row['n_trades_opt']):>6}  {row['cost_pct_opt']:>5.1f}  "
              f"{feq_l:>9.2f}L")
    print(f"\n  Criterion : Sharpe (primary), Calmar (secondary)")
    print(f"  Min trades: {cfg['min_trades_opt']} — combos below this rejected (prevents noise-fit on tiny samples)")
    print(f"  Top {cfg['n_top_candidates']} forwarded to Validation scoring")

    # C. Validation Scoring
    print(f"\n  C. VALIDATION SCORING  (2018-2021)  |  Top-N re-tested, winner by combined score")
    print(SEP2)
    print(f"  Combined Score = sqrt(IS_Sharpe × Val_Sharpe)  [geometric mean, penalises IS overfitting]")
    print()
    print(f"  {'Rk':>2}  {'Hold':>4}  {'Z-Thr':>5}  {'Vol%':>4}  {'CD':>2}"
          f"  {'IS_Sh':>7}  {'V_Sh':>7}  {'Comb':>7}"
          f"  {'VCAGR%':>7}  {'VDD%':>6}  {'VWR%':>5}  {'VPF':>6}"
          f"  {'VTrades':>7}  {'VFinalEq(L)':>11}")
    print("  " + "." * 80)
    for i, row in val_df.iterrows():
        marker = "  <- WINNER" if i == 0 else ""
        feq_v = row.get("final_eq_val", 0) / 100000
        print(f"  {i+1:>2}  {int(row['holding_period']):>4}  "
              f"{row['z_threshold']:>5.2f}  {row['vol_threshold']*100:>4.1f}  "
              f"{int(row['cooldown_days']):>2}  "
              f"{row['sharpe_opt']:>7.4f}  {row['sharpe_val']:>7.4f}  "
              f"{row['combined_score']:>7.4f}  "
              f"{row['cagr_val']:>7.2f}  {row['max_dd_val']:>6.2f}  "
              f"{row['wr_val']:>5.1f}  {row['pf_val']:>6.4f}  "
              f"{int(row['n_trades_val']):>7}  {feq_v:>10.2f}L{marker}")

    # D. Locked Params
    print(f"\n  D. LOCKED OPTIMAL PARAMETERS  (frozen — never changed after this point)")
    print(SEP2)
    print(f"  Holding Period    = {int(best_params['holding_period'])} days")
    print(f"  Z-Score Threshold = {best_params['z_threshold']:.2f}σ  (|composite z| > threshold to enter)")
    print(f"  Vol Filter        = {best_params['vol_threshold']*100:.1f}%  (skip stocks with 10d vol above this)")
    print(f"  Cooldown          = {int(best_params['cooldown_days'])} days  (no re-entry on same stock)")
    print(f"  Trend Filter      = {'ON — 20d MA' if best_params.get('use_trend_filter') else 'OFF'}")
    print(f"  Stop-Loss         = Disabled  (time-exit only)")
    print(f"  Max Positions     = {cfg['max_positions']}")

    # E. IS Performance — FIX 4: initial + final capital + trades per window
    print(f"\n  E. IN-SAMPLE PERFORMANCE  (locked params applied per sub-window)")
    print(SEP2)
    print(f"  {'Metric':<34}{'Opt 2010-17':>22}{'Val 2018-21':>22}")
    print("  " + "-" * 78)

    # Capital summary block first
    _row("Initial Capital",     fic(g(is_m,"initial_capital")),  fic(g(val_m,"initial_capital")))
    _row("Final Equity",        fic(g(is_m,"final_equity")),     fic(g(val_m,"final_equity")))
    _row("Trades Executed",     fn(g(is_m,"n_trades")),          fn(g(val_m,"n_trades")))
    print("  " + "-" * 78)
    _row("Total Return (%)",    fp(g(is_m,"total_ret_pct")),     fp(g(val_m,"total_ret_pct")))
    _row("CAGR (%)",            fp(g(is_m,"cagr")),              fp(g(val_m,"cagr")))
    _row("Sharpe Ratio",        f4(g(is_m,"sharpe")),            f4(g(val_m,"sharpe")))
    _row("Sortino Ratio",       f4(g(is_m,"sortino")),           f4(g(val_m,"sortino")))
    _row("Calmar Ratio",        f4(g(is_m,"calmar")),            f4(g(val_m,"calmar")))
    _row("Max Drawdown (%)",    fp(g(is_m,"max_dd")),            fp(g(val_m,"max_dd")))
    print("  " + "-" * 78)
    _row("Win Rate (%)",        fp(g(is_m,"win_rate")),          fp(g(val_m,"win_rate")))
    _row("Profit Factor",       f4(g(is_m,"profit_factor")),     f4(g(val_m,"profit_factor")))
    _row("Expectancy / Trade",  fi(g(is_m,"expectancy")),        fi(g(val_m,"expectancy")))
    _row("Avg Winner",          fi(g(is_m,"avg_win")),           fi(g(val_m,"avg_win")))
    _row("Avg Loser",           fi(g(is_m,"avg_loss")),          fi(g(val_m,"avg_loss")))
    _row("  Long  Trades",      fn(g(is_m,"n_long")),            fn(g(val_m,"n_long")))
    _row("  Short Trades",      fn(g(is_m,"n_short")),           fn(g(val_m,"n_short")))
    _row("Avg Hold (days)",     f2(g(is_m,"avg_hold")),          f2(g(val_m,"avg_hold")))
    _row("Max Consec. Wins",    fn(g(is_m,"max_cw")),            fn(g(val_m,"max_cw")))
    _row("Max Consec. Losses",  fn(g(is_m,"max_cl")),            fn(g(val_m,"max_cl")))
    print("  " + "-" * 78)
    _row("Gross PnL",           fi(g(is_m,"gross_pnl")),         fi(g(val_m,"gross_pnl")))
    _row("Total Costs",         fi(g(is_m,"total_costs")),       fi(g(val_m,"total_costs")))
    _row("Net PnL",             fi(g(is_m,"total_ret_inr")),     fi(g(val_m,"total_ret_inr")))
    _row("Cost / Gross (%)",    fcp(g(is_m,"cost_to_gross")),    fcp(g(val_m,"cost_to_gross")))
    _row("Long Side Net PnL",   fi(g(is_m,"long_pnl")),          fi(g(val_m,"long_pnl")))
    _row("Short Side Net PnL",  fi(g(is_m,"short_pnl")),         fi(g(val_m,"short_pnl")))
    _row("Avg Lots / Trade",    f2(g(is_m,"avg_lots")),          f2(g(val_m,"avg_lots")))
    _row("Avg Ret on Margin(%)",fcp(g(is_m,"avg_rom")),          fcp(g(val_m,"avg_rom")))
    print(); print(SEP)


# ==============================================================================
#  SHEET 2  —  OUT-OF-SAMPLE
# ==============================================================================
def print_sheet2(oos_m, best_params, is_m, val_m, src, runtime):
    SEP  = "=" * 84
    SEP2 = "-" * 84

    def fp(v):
        try:    return f"{v:>+.2f}%"
        except: return "N/A"
    def fi(v):
        try:    return f"INR {v:>+,.0f}"
        except: return "N/A"
    def fic(v):
        try:    return f"INR {v:>,.0f}"
        except: return "N/A"
    def f4(v):
        try:    return f"{v:.4f}"
        except: return "N/A"
    def fn(v):
        try:    return f"{int(v):,}"
        except: return "N/A"
    def f2(v):
        try:    return f"{v:.2f}"
        except: return "N/A"
    def fcp(v):
        try:    return f"{v:.1f}%"
        except: return "N/A"

    g = lambda k: oos_m.get(k)

    def _row(label, val, w=42):
        print(f"  {label:<{w}}{str(val):>34}")

    print(); print(SEP)
    print("  SHEET 2  |  OUT-OF-SAMPLE  |  NSE FUTSTK ENGINE v5.0  |  Universe Capital")
    print("  PURE FORWARD TEST  |  Parameters frozen from IS — zero contact with OOS data")
    print(SEP)

    # A. OOS Setup
    print(f"\n  A. OOS SETUP")
    print(SEP2)
    print(f"  OOS Window      : {DATES['oos_start']}  to  {DATES['oos_end']}  [2 years]")
    print(f"  Data Status     : UNSEEN — no contact with IS optimisation or validation")
    print(f"  Parameters      : LOCKED from IS validation scoring, not modified")
    print(f"  Data Source     : {src}")
    print()
    print(f"  LOCKED PARAMS:")
    print(f"    Holding Period    = {int(best_params['holding_period'])} days")
    print(f"    Z-Score Threshold = {best_params['z_threshold']:.2f}σ")
    print(f"    Vol Filter        = {best_params['vol_threshold']*100:.1f}%")
    print(f"    Cooldown          = {int(best_params['cooldown_days'])} days")
    print(f"    Trend Filter      = {'ON — 20d MA' if best_params.get('use_trend_filter') else 'OFF'}")
    print(f"    Stop-Loss         = Disabled")

    # B. OOS Performance — FIX 4: initial + final + trades
    print(f"\n  B. OUT-OF-SAMPLE PERFORMANCE  (2023-2024)")
    print(SEP2)
    print(f"  {'Metric':<42}{'Value':>34}")
    print("  " + "-" * 76)
    _row("Initial Capital",               fic(g("initial_capital")))
    _row("Final Equity",                  fic(g("final_equity")))
    _row("Trades Executed",               fn(g("n_trades")))
    print("  " + "-" * 76)
    _row("Total Return (%)",              fp(g("total_ret_pct")))
    _row("CAGR (%)",                      fp(g("cagr")))
    _row("Sharpe Ratio",                  f4(g("sharpe")))
    _row("Sortino Ratio",                 f4(g("sortino")))
    _row("Calmar Ratio",                  f4(g("calmar")))
    _row("Max Drawdown (%)",              fp(g("max_dd")))
    print("  " + "-" * 76)
    _row("Win Rate (%)",                  fp(g("win_rate")))
    _row("Profit Factor",                 f4(g("profit_factor")))
    _row("Expectancy / Trade",            fi(g("expectancy")))
    _row("Avg Winner",                    fi(g("avg_win")))
    _row("Avg Loser",                     fi(g("avg_loss")))
    _row("  Long  Trades",                fn(g("n_long")))
    _row("  Short Trades",                fn(g("n_short")))
    _row("  Stop-Loss Exits",             fn(g("n_stops")))
    _row("Avg Hold (days)",               f2(g("avg_hold")))
    _row("Max Consec. Wins",              fn(g("max_cw")))
    _row("Max Consec. Losses",            fn(g("max_cl")))
    print("  " + "-" * 76)
    _row("Gross PnL  (futures formula)",  fi(g("gross_pnl")))
    _row("Total Transaction Costs",       fi(g("total_costs")))
    _row("Net PnL",                       fi(g("total_ret_inr")))
    _row("Cost / Gross (%)",              fcp(g("cost_to_gross")))
    _row("Long Side Net PnL",             fi(g("long_pnl")))
    _row("Short Side Net PnL",            fi(g("short_pnl")))
    _row("Avg Lots / Trade",              f2(g("avg_lots")))
    _row("Avg Return on Margin (%)",      fcp(g("avg_rom")))

    # C. Comparison
    print(f"\n  C. IS vs OOS COMPARISON")
    print(SEP2)
    print(f"  {'Metric':<24}{'IS-Opt 10-17':>14}{'IS-Val 18-21':>14}{'OOS 23-24':>14}")
    print("  " + "-" * 66)

    def _crow(label, ov, vv, oosv, w=24):
        print(f"  {label:<{w}}{str(ov):>14}{str(vv):>14}{str(oosv):>14}")

    def fp2(m, k):
        try:    return f"{m[k]:>+.2f}%"
        except: return "N/A"
    def f42(m, k):
        try:    return f"{m[k]:.4f}"
        except: return "N/A"
    def fic2(m, k):
        try:    return f"{m[k]/1e5:.2f}L"
        except: return "N/A"

    _crow("Initial Capital", fic2(is_m,"initial_capital"),  fic2(val_m,"initial_capital"),  fic2(oos_m,"initial_capital"))
    _crow("Final Equity",    fic2(is_m,"final_equity"),     fic2(val_m,"final_equity"),     fic2(oos_m,"final_equity"))
    _crow("Trades Executed", fn(is_m.get("n_trades",0)),    fn(val_m.get("n_trades",0)),    fn(g("n_trades")))
    print("  " + "-" * 66)
    _crow("Total Return (%)", fp2(is_m,"total_ret_pct"), fp2(val_m,"total_ret_pct"), fp(g("total_ret_pct")))
    _crow("CAGR (%)",         fp2(is_m,"cagr"),          fp2(val_m,"cagr"),          fp(g("cagr")))
    _crow("Sharpe",           f42(is_m,"sharpe"),        f42(val_m,"sharpe"),        f4(g("sharpe")))
    _crow("Sortino",          f42(is_m,"sortino"),       f42(val_m,"sortino"),       f4(g("sortino")))
    _crow("Max DD (%)",       fp2(is_m,"max_dd"),        fp2(val_m,"max_dd"),        fp(g("max_dd")))
    _crow("Win Rate (%)",     fp2(is_m,"win_rate"),      fp2(val_m,"win_rate"),      fp(g("win_rate")))
    _crow("Profit Factor",    f42(is_m,"profit_factor"), f42(val_m,"profit_factor"), f4(g("profit_factor")))

    # D. Conclusion — FIX 5: corrected degradation label
    print(f"\n  D. CONCLUSION")
    print(SEP2)
    sh_opt = is_m.get("sharpe", 0)
    sh_val = val_m.get("sharpe", 0)
    sh_oos = g("sharpe") or 0
    pf_oos = g("profit_factor") or 0
    ctg    = g("cost_to_gross") or 999
    n_oos  = g("n_trades") or 0

    # FIX 5: sign-correct degradation
    chg_ov  = (sh_val - sh_opt) / (abs(sh_opt) + 1e-9) * 100  # positive = improvement
    chg_vo  = (sh_oos - sh_val) / (abs(sh_val) + 1e-9) * 100

    print(f"  Sharpe path : IS-Opt {sh_opt:.4f}  →  IS-Val {sh_val:.4f}  →  OOS {sh_oos:.4f}")
    print(f"  IS-Opt → Val: {chg_ov:>+.1f}%  "
          f"({'improvement — Val generalised better than Opt' if chg_ov > 0 else 'degradation — some IS overfitting'})")
    print(f"  IS-Val → OOS: {chg_vo:>+.1f}%  "
          f"({'stable' if abs(chg_vo) < 40 else 'drop — regime shift or limited OOS sample'})")
    print()

    if sh_oos > 0.8 and pf_oos > 1.2:
        verdict = "STRONG    — Robust on completely unseen 2-year forward window."
    elif sh_oos > 0.4 and pf_oos > 1.0:
        verdict = "SOLID     — Positive OOS PnL with acceptable risk-adjusted return."
    elif sh_oos > 0 and pf_oos > 1.0:
        verdict = "MARGINAL  — Profitable OOS but weak Sharpe. Monitor live carefully."
    else:
        verdict = "WEAK      — Strategy does not hold on OOS data. Revisit alpha."

    fit_note  = ("Low — IS-Val Sharpe did not inflate vs IS-Opt"
                 if chg_ov >= -20 else "CAUTION — IS-Opt Sharpe inflated; Val degraded")
    cost_note = (f"Cost/Gross = {ctg:.1f}%  →  "
                 + ("healthy (<50%)" if ctg < 50 else "HIGH (>50%) — edge thinner than costs"))
    trade_note= (f"OOS trades = {n_oos}  →  "
                 + ("adequate sample" if n_oos >= 30 else
                    "small sample — interpret metrics with caution"))

    print(f"  Verdict    : {verdict}")
    print(f"  Fit check  : {fit_note}")
    print(f"  Cost check : {cost_note}")
    print(f"  Sample     : {trade_note}")
    print()
    print(f"  Lot Sizes  : " + "  ".join(f"{k}={v}" for k,v in FUTSTK_LOTS.items()))
    print(f"  Runtime    : {runtime:.1f}s  |  No lookahead  |  Params locked before OOS")
    print(SEP); print()


# ==============================================================================
#  MAIN
# ==============================================================================
def main():
    t0  = datetime.now()
    cfg = CFG

    data, src = load_data(cfg)
    features  = build_features(data, cfg)

    # IS: Optimisation window (2010-2017)
    opt_df, top_candidates = run_optimisation(features, cfg)

    # IS: Validation window (2018-2021) — pick winner
    val_df, best_params = run_validation(features, cfg, top_candidates)

    # Compute IS sub-window metrics with best params
    so, eo = DATES["opt_start"], DATES["opt_end"]
    sigs_o, sc_o = build_signals(features, best_params, so, eo)
    tdf_o, edf_o = run_backtest(features, sigs_o, sc_o, best_params, so, eo)
    is_m = compute_metrics(tdf_o, edf_o, best_params, "OPT")

    sv, ev = DATES["val_start"], DATES["val_end"]
    sigs_v, sc_v = build_signals(features, best_params, sv, ev)
    tdf_v, edf_v = run_backtest(features, sigs_v, sc_v, best_params, sv, ev)
    val_m = compute_metrics(tdf_v, edf_v, best_params, "VAL")

    # OOS: 2023-2024 — LOCKED params, pure unseen test
    soo, eoo = DATES["oos_start"], DATES["oos_end"]
    sigs_oo, sc_oo = build_signals(features, best_params, soo, eoo)
    tdf_oo, edf_oo = run_backtest(features, sigs_oo, sc_oo, best_params, soo, eoo)
    oos_m = compute_metrics(tdf_oo, edf_oo, best_params, "OOS")

    elapsed = (datetime.now() - t0).total_seconds()
    print_sheet1(opt_df, val_df, best_params, is_m, val_m, cfg, src)
    print_sheet2(oos_m, best_params, is_m, val_m, src, elapsed)


if __name__ == "__main__":
    main()