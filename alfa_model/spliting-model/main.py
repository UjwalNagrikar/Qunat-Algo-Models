#!/usr/bin/env python3
"""
================================================================================
  UNIVERSE CAPITAL  |  WALK-FORWARD BACKTESTING ENGINE  v2.1
  Strategy   : Cross-Sectional Mean Reversion Alpha  (PATCHED FOR LIVE NSE)
  Universe   : 25 NSE Large-Cap  |  Daily OHLCV

  WALK-FORWARD REGIME PIPELINE
  Training   : 2010 - 2018  Grid-search optimal hyperparameters
  Validation : 2019 - 2021  Out-of-sample proof  [params LOCKED]
  Execution  : 2022 - 2024  Final live simulation  [params LOCKED]

  v2.1 PATCHES vs v2.0  (based on live NSE diagnostics)
  ────────────────────────────────────────────────────────────────────────────
  1. STOP-LOSS REMOVED from grid search options.
     Mean-reversion P&L depends on the bounce — a 4% tight stop exits before
     the reversal and doubles round-trip cost.  v2.1 uses TIME-EXIT only.
     Stop-loss remains as an *optional* circuit-breaker at 8% (configurable).

  2. COMPOSITE SIGNAL: average of z(1d) + z(3d) return z-scores.
     Real NSE daily returns have lower cross-sectional dispersion than
     synthetic data.  A single 1-day z-score fires on noise.  The composite
     smooths noise and raises the effective signal-to-cost ratio.

  3. GRID NOW EXPLICITLY TESTS holding_period 2, 3 (excluded 5).
     On live data, 5-day hold on a 3-day-mean-reversion signal costs too much.

  4. VOL FILTER tightened from 3.5% to 2.5% default (live NSE is more volatile).
     Grid also tests 2.0% and 3.0% as options.

  5. COST TRANSPARENCY: total cost and cost-to-gross ratio printed per phase.
     When cost/gross > 60%, strategy needs wider signal edge.

  Short-leg: modelled as FUTSTK (NSE single-stock futures).
================================================================================
"""
import warnings, os, itertools
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

# ── Theme ─────────────────────────────────────────────────────────────────────
DARK  = "#0a0e1a"; NAVY  = "#0d1b2e"; PANEL = "#111827"
GOLD  = "#d4af37"; GREEN = "#00b37d"; RED   = "#e05c5c"
BLUE  = "#4a9eda"; AMBER = "#f39c12"; GREY  = "#8892a0"; WHITE = "#e8eaf0"
PURP  = "#9b59b6"

PHASE_SPANS = {
    "Training":   ("2010-01-01", "2018-12-31"),
    "Validation": ("2019-01-01", "2021-12-31"),
    "Execution":  ("2022-01-01", "2024-12-31"),
}
PHASE_COLS = {"Training": BLUE, "Validation": AMBER, "Execution": GREEN}

plt.rcParams.update({
    "figure.facecolor": DARK,  "axes.facecolor":   PANEL,
    "axes.edgecolor":   "#1e2a3a", "axes.labelcolor": GREY,
    "xtick.color":      GREY,  "ytick.color":       GREY,
    "text.color":       WHITE, "grid.color":        "#1e2a3a",
    "grid.linewidth":   0.55,  "font.family":       "monospace",
    "axes.spines.top":  False, "axes.spines.right": False,
})

# ==============================================================================
#  CONFIGURATION  v2.1
# ==============================================================================
BASE_CFG = {
    "symbols": [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
        "HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
        "LT.NS","AXISBANK.NS","BAJFINANCE.NS","WIPRO.NS","TITAN.NS",
        "MARUTI.NS","SUNPHARMA.NS","NTPC.NS","POWERGRID.NS","NESTLEIND.NS",
        "ONGC.NS","HCLTECH.NS","TECHM.NS","ULTRACEMCO.NS","ASIANPAINT.NS",
    ],
    "full_start": "2010-01-01",
    "full_end":   "2024-12-31",
    "initial_capital": 2_000_000,

    # ── v2.1 Grid: shorter holds, no stop, tighter vol ──────────────────────
    "param_grid": {
        "holding_period": [2, 3, 4],        # removed 5 (too long for 3d AC)
        "z_composite":    [0.50, 0.75, 1.0],# composite 1d+3d z-score threshold
        "vol_threshold":  [0.020, 0.025, 0.030], # tighter than v2.0
        "cooldown_days":  [3, 5],
    },

    # Fixed
    "long_pct":       0.20,   "short_pct":     0.20,
    "max_positions":  8,      "vol_window":    10,
    "risk_pct":       0.015,  "max_pos_pct":   0.14,

    # ── v2.1: Stop-loss is a safety net only (8%), not a strategy tool ───────
    # Set to None to disable entirely, or a float like 0.08 for 8%
    "stop_loss_pct": None,         # DISABLED — mean reversion needs the bounce

    # Transaction costs (NSE futures proxy — unchanged from v2.0)
    "brokerage_flat": 20.0, "brokerage_pct": 0.0003,
    "stt_sell_pct":   0.0001, "exchange_pct":  0.000019,
    "sebi_pct":       0.000001, "stamp_buy_pct": 0.00002,
    "gst_pct":        0.18, "slippage_pct":  0.0002,
}


# ==============================================================================
#  DATA LOADING  (Yahoo Finance primary, synthetic fallback)
# ==============================================================================
def _try_yfinance(cfg):
    frames = []
    for sym in cfg["symbols"]:
        try:
            raw = yf.download(sym, start=cfg["full_start"], end=cfg["full_end"],
                              auto_adjust=True, progress=False, threads=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            raw.columns = [c.lower().replace(" ","_") for c in raw.columns]
            if raw.empty or len(raw) < 500: continue
            raw = raw[["open","high","low","close","volume"]].replace(0,np.nan)
            raw = raw.dropna(subset=["open","close"])
            raw.index = pd.to_datetime(raw.index); raw.index.name = "date"
            raw["symbol"] = sym.replace(".NS","")
            frames.append(raw)
        except Exception:
            pass
    return frames if len(frames) >= 10 else None


def _synthetic_nse(cfg):
    np.random.seed(2010)
    dates = pd.bdate_range(cfg["full_start"], cfg["full_end"])
    T     = len(dates)
    mkt   = np.random.normal(0.00012, 0.008, T)
    for s,e,adj in [(750,790,-0.004),(1700,1740,-0.003),
                    (2580,2640,-0.007),(2650,2700,0.005),(3000,3040,-0.002)]:
        mkt[s:e] += adj + np.random.normal(0, abs(adj)*0.4, e-s)
    seed = {
        "RELIANCE":1050,"TCS":750,"HDFCBANK":350,"INFY":650,"ICICIBANK":430,
        "HINDUNILVR":250,"ITC":120,"SBIN":250,"BHARTIARTL":320,"KOTAKBANK":380,
        "LT":1600,"AXISBANK":260,"BAJFINANCE":180,"WIPRO":380,"TITAN":270,
        "MARUTI":1400,"SUNPHARMA":280,"NTPC":200,"POWERGRID":95,"NESTLEIND":2600,
        "ONGC":320,"HCLTECH":380,"TECHM":270,"ULTRACEMCO":900,"ASIANPAINT":800,
    }
    frames = []
    for sym, s0 in seed.items():
        idvol = np.random.uniform(0.009, 0.018)
        beta  = np.random.uniform(0.60, 1.40)
        drift = np.random.uniform(-0.00008, 0.00030)
        ac    = np.random.uniform(-0.45, -0.22)
        idio  = np.zeros(T)
        for t in range(1, T):
            idio[t] = ac*idio[t-1] + idvol*np.random.randn()
        ret   = mkt*beta + idio + drift
        close = s0 * np.exp(np.cumsum(ret))
        intra = np.abs(np.random.normal(0.008,0.004,T)).clip(0.002,0.05)
        open_ = np.concatenate([[s0],close[:-1]])*np.exp(np.random.normal(0,0.003,T))
        high  = np.maximum(open_,close)*(1+intra*0.6)
        low   = np.minimum(open_,close)*(1-intra*0.4)
        df    = pd.DataFrame({"open":np.round(open_,2),"high":np.round(high,2),
                              "low":np.round(low,2),"close":np.round(close,2),
                              "volume":np.random.lognormal(15,.9,T).astype(int),
                              "symbol":sym}, index=dates)
        df.index.name = "date"; frames.append(df)
    return frames


def load_data(cfg):
    print("\n" + "="*74)
    print("  STEP 1  |  LOADING MARKET DATA  (2010-2024)")
    print("="*74)
    print("  Trying Yahoo Finance...")
    frames = _try_yfinance(cfg)
    if not frames:
        print("  Network blocked -- synthetic NSE fallback")
        frames = _synthetic_nse(cfg)
        src = "Synthetic (negative-AC AR1)"
    else:
        src = "Yahoo Finance (live)"
    for f in frames:
        s=f["symbol"].iloc[0]
        print(f"  OK  {s:<14} {len(f):>4} days | INR {f['close'].iloc[-1]:>10,.2f}")
    data = pd.concat(frames).reset_index().set_index(["date","symbol"]).sort_index()
    data = data.replace(0,np.nan).dropna(subset=["open","close"])
    nd = data.index.get_level_values("date").nunique()
    ns = data.index.get_level_values("symbol").nunique()
    print(f"\n  Universe: {ns} stocks | {nd} days | source: {src}\n")
    return data


# ==============================================================================
#  FEATURES  (v2.1: adds 3-day return and composite z-score)
# ==============================================================================
def build_features(data):
    close = data["close"].unstack("symbol").sort_index().ffill(limit=3)
    open_ = data["open"].unstack("symbol").sort_index().ffill(limit=3)
    ret1  = close.pct_change(1)
    ret3  = close.pct_change(3)
    vol10 = ret1.rolling(10).std()

    # Per-row cross-sectional z-scores (vectorised, no Python loop)
    def cs_z(ret_df):
        mu  = ret_df.mean(axis=1)
        sig = ret_df.std(axis=1).replace(0, np.nan)
        return ret_df.sub(mu, axis=0).div(sig, axis=0)

    z1 = cs_z(ret1)
    z3 = cs_z(ret3)
    # Composite = equal-weight average of the two z-scores
    # Only defined where BOTH are available (z3 requires ≥4 days of history)
    z_comp = (z1 + z3) / 2.0

    # Percentile rank (0-1) used for long/short bucket selection
    cs_rank = ret1.rank(axis=1, ascending=True, pct=True)

    print(f"  Features: ret1, ret3, vol10, cs_z1, cs_z3, z_composite, cs_rank")
    print(f"  Matrix  : {close.shape}  (dates x stocks)\n")
    return {"close":close,"open":open_,"ret1":ret1,"ret3":ret3,
            "vol10":vol10,"z_comp":z_comp,"z1":z1,"cs_rank":cs_rank}


# ==============================================================================
#  COST MODEL  (unchanged)
# ==============================================================================
def _oc(tv, cfg, side):
    b = min(cfg["brokerage_flat"], tv*cfg["brokerage_pct"])
    return (b + tv*cfg["exchange_pct"] + tv*cfg["sebi_pct"]
            + (b + tv*cfg["exchange_pct"])*cfg["gst_pct"]
            + (tv*cfg["stamp_buy_pct"] if side=="buy"  else 0)
            + (tv*cfg["stt_sell_pct"]  if side=="sell" else 0))

def rtc(ep,xp,sh,direction,cfg):
    ten=ep*sh; tex=xp*sh
    c = (_oc(ten,cfg,"buy") +_oc(tex,cfg,"sell") if direction==1
         else _oc(ten,cfg,"sell")+_oc(tex,cfg,"buy"))
    return c + (ten+tex)*cfg["slippage_pct"]


# ==============================================================================
#  SIGNAL BUILDER  v2.1  (composite z-score, vectorised)
# ==============================================================================
def build_signals(features, cfg, s_date, e_date):
    """
    Composite signal = average of 1d and 3d cross-sectional z-scores.
    This reduces the impact of single-day outlier moves (earnings, splits,
    index rebalances) that create false mean-reversion signals.

    Long  : bottom long_pct by cs_rank  AND  composite_z <= -z_thresh
    Short : top   short_pct by cs_rank  AND  composite_z >=  z_thresh
    """
    cs_rank = features["cs_rank"]
    z_comp  = features["z_comp"]
    vol10   = features["vol10"]

    mask   = (cs_rank.index >= s_date) & (cs_rank.index <= e_date)
    rnk    = cs_rank.loc[mask].copy()
    zc     = z_comp.loc[mask].copy()
    v10    = vol10.loc[mask].copy()

    lp = cfg.get("long_pct",  0.20)
    sp = cfg.get("short_pct", 0.20)
    vt = cfg.get("vol_threshold", 0.025)
    zt = cfg.get("z_composite",   0.75)  # key param name changed to z_composite

    # Vol mask: NaN out high-vol stocks on each day (they distort rankings)
    hi_vol  = v10 > vt
    rnk_adj = rnk.copy(); rnk_adj[hi_vol] = np.nan
    zc_adj  = zc.copy();  zc_adj[hi_vol]  = np.nan

    # Long: bottom lp AND z <= -zt (composite z-score confirms oversold)
    # Short: top sp AND z >= +zt (composite z-score confirms overbought)
    long_sig  = (rnk_adj <= lp)       & (zc_adj <= -zt)
    short_sig = (rnk_adj >= (1 - sp)) & (zc_adj >= zt)

    signals   = (long_sig.astype(np.int8) - short_sig.astype(np.int8))
    signals   = signals.fillna(0).astype(np.int8)
    ret_score = zc_adj.abs()   # priority: most extreme composite z first
    return signals, ret_score


# ==============================================================================
#  BACKTEST ENGINE  v2.1  (stop-loss optional, default OFF)
# ==============================================================================
def run_backtest(features, signals, ret_score, cfg, s_date, e_date):
    open_p  = features["open"]
    close_p = features["close"]
    op      = open_p[ (open_p.index  >= s_date) & (open_p.index  <= e_date)]
    cp      = close_p[(close_p.index >= s_date) & (close_p.index <= e_date)]
    sig     = signals[(signals.index  >= s_date) & (signals.index  <= e_date)]

    dates    = op.index.tolist()
    cash     = float(cfg["initial_capital"])
    positions= {}; cooldown = {}; trades = []; equity_rec = []
    stop_exits=0
    hold_per = int(cfg.get("holding_period", 3))
    sl_pct   = cfg.get("stop_loss_pct", None)  # None = disabled
    max_pos  = int(cfg.get("max_positions", 8))
    cd_days  = int(cfg.get("cooldown_days", 3))

    for i, date in enumerate(dates):
        dop = op.loc[date]; dcp = cp.loc[date]

        locked = unrealised = 0.0
        for sym, pos in positions.items():
            locked += pos["ep"]*pos["sh"]
            px = dcp.get(sym, np.nan)
            if not np.isnan(px):
                unrealised += (px-pos["ep"])*pos["sh"]*pos["dir"]

        equity_rec.append({"date":date,"cash":cash,
                           "equity":cash+locked+unrealised,"n_open":len(positions)})
        cur_eq = cash + locked + unrealised

        # Optional stop-loss (safety net only, default disabled)
        if sl_pct is not None:
            for sym in list(positions):
                pos=positions[sym]; px=dcp.get(sym,np.nan)
                if np.isnan(px): continue
                tr=(px-pos["ep"])/pos["ep"]*pos["dir"]
                if tr < -sl_pct:
                    positions.pop(sym)
                    raw=(px-pos["ep"])*pos["sh"]*pos["dir"]
                    c=rtc(pos["ep"],px,pos["sh"],pos["dir"],cfg)
                    cash+=pos["ep"]*pos["sh"]+raw-c
                    cooldown[sym]=i; stop_exits+=1
                    trades.append(_rec(len(trades)+1,sym,pos["dir"],
                                  dates[pos["i"]],date,pos["ep"],px,
                                  pos["sh"],i-pos["i"],"STOP",raw,c))

        # Time exit
        for sym in [s for s,p in list(positions.items()) if i>=p["i"]+hold_per]:
            pos=positions.pop(sym)
            xp=dop.get(sym,pos["ep"])
            if np.isnan(xp): xp=pos["ep"]
            raw=(xp-pos["ep"])*pos["sh"]*pos["dir"]
            c=rtc(pos["ep"],xp,pos["sh"],pos["dir"],cfg)
            cash+=pos["ep"]*pos["sh"]+raw-c; cooldown[sym]=i
            trades.append(_rec(len(trades)+1,sym,pos["dir"],
                          dates[pos["i"]],date,pos["ep"],xp,
                          pos["sh"],i-pos["i"],"HOLD",raw,c))

        # Entry
        if i==0: continue
        prev=dates[i-1]
        if prev not in sig.index: continue
        sr=sig.loc[prev]; cands=sr[sr!=0]
        if cands.empty: continue
        if prev in ret_score.index:
            sc=ret_score.loc[prev][cands.index].fillna(0)
            cands=cands.loc[sc.sort_values(ascending=False).index]

        for sym,direction in cands.items():
            if len(positions)>=max_pos: break
            if sym in positions: continue
            if i-cooldown.get(sym,-9999)<cd_days: continue
            ep=dop.get(sym,np.nan)
            if np.isnan(ep) or ep<=0: continue
            alloc=min(cur_eq*cfg["risk_pct"],cur_eq*cfg["max_pos_pct"])
            shares=max(1,int(alloc/ep))
            needed=ep*shares
            if needed>cash*0.97 or shares<1: continue
            cash-=needed
            positions[sym]={"i":i,"ep":ep,"sh":shares,"dir":int(direction)}

    # EOD force-close
    ld=dates[-1]; lc=cp.loc[ld]
    for sym,pos in list(positions.items()):
        xp=lc.get(sym,pos["ep"])
        if np.isnan(xp): xp=pos["ep"]
        raw=(xp-pos["ep"])*pos["sh"]*pos["dir"]
        c=rtc(pos["ep"],xp,pos["sh"],pos["dir"],cfg)
        cash+=pos["ep"]*pos["sh"]+raw-c
        trades.append(_rec(len(trades)+1,sym,pos["dir"],
                      dates[pos["i"]],ld,pos["ep"],xp,
                      pos["sh"],len(dates)-1-pos["i"],"EOD",raw,c))

    tdf=pd.DataFrame(trades)
    edf=pd.DataFrame(equity_rec).set_index("date")
    return tdf, edf, stop_exits


def _rec(tid,sym,dir_,ed,xd,ep,xp,sh,hold,reason,raw,c):
    net=raw-c
    return {"trade_id":tid,"symbol":sym,
            "direction":"LONG" if dir_==1 else "SHORT",
            "entry_date":ed,"exit_date":xd,
            "entry_price":round(ep,2),"exit_price":round(xp,2),
            "shares":sh,"holding_days":hold,"exit_reason":reason,
            "gross_pnl":round(raw,2),"cost":round(c,2),
            "net_pnl":round(net,2),
            "ret_pct":round((xp/ep-1)*dir_*100,4)}


# ==============================================================================
#  METRICS  (added cost transparency)
# ==============================================================================
def _mc(lst,v):
    b=c=0
    for x in lst:
        c=c+1 if x==v else 0; b=max(b,c)
    return b


def compute_metrics(tdf,edf,cfg,phase=""):
    init=cfg["initial_capital"]; eq=edf["equity"]
    final=eq.iloc[-1]; nyrs=(edf.index[-1]-edf.index[0]).days/365.25
    rp=(final/init-1)*100; ri=final-init
    cagr=((final/init)**(1/max(nyrs,0.01))-1)*100
    daily=eq.pct_change().dropna()
    sh=daily.mean()/daily.std()*np.sqrt(252) if daily.std()>0 else 0
    dn=daily[daily<0]
    so=daily.mean()/dn.std()*np.sqrt(252) if len(dn)>1 else 0
    rm=eq.cummax(); dds=(eq-rm)/rm; mdd=dds.min()*100
    cal=cagr/abs(mdd) if mdd!=0 else 0
    if tdf.empty: return {}
    win=tdf[tdf["net_pnl"]>0]; los=tdf[tdf["net_pnl"]<=0]
    wr=len(win)/len(tdf)*100
    gp=win["net_pnl"].sum() if len(win)>0 else 0
    gl=abs(los["net_pnl"].sum()) if len(los)>0 else 1e-9
    pf=gp/gl; exp=tdf["net_pnl"].mean()
    gross_pnl_total = tdf["gross_pnl"].sum()
    cost_total      = tdf["cost"].sum()
    cost_to_gross   = (abs(cost_total)/abs(gross_pnl_total)*100
                       if gross_pnl_total!=0 else 999)
    avg_w=win["net_pnl"].mean() if len(win)>0 else 0
    avg_l=los["net_pnl"].mean() if len(los)>0 else 0
    ws=(tdf["net_pnl"]>0).astype(int).tolist()
    lg=tdf[tdf["direction"]=="LONG"]; sh_=tdf[tdf["direction"]=="SHORT"]
    st=(tdf[tdf["exit_reason"]=="STOP"]
        if "exit_reason" in tdf.columns else pd.DataFrame())
    return {
        "phase":phase,"total_ret_pct":rp,"total_ret_inr":ri,"cagr":cagr,
        "sharpe":sh,"sortino":so,"calmar":cal,"max_dd":mdd,
        "n_trades":len(tdf),"n_long":len(lg),"n_short":len(sh_),
        "n_stops":len(st),"win_rate":wr,
        "wr_long":(lg["net_pnl"]>0).mean()*100 if len(lg)>0 else 0,
        "wr_short":(sh_["net_pnl"]>0).mean()*100 if len(sh_)>0 else 0,
        "profit_factor":pf,"expectancy":exp,
        "avg_win":avg_w,"avg_loss":avg_l,
        "wl_ratio":abs(avg_w/avg_l) if avg_l!=0 else 0,
        "avg_hold":tdf["holding_days"].mean(),
        "max_consec_wins":_mc(ws,1),"max_consec_losses":_mc(ws,0),
        "gross_pnl":gross_pnl_total,"total_costs":cost_total,
        "cost_to_gross_pct":cost_to_gross,
        "long_pnl":lg["net_pnl"].sum(),"short_pnl":sh_["net_pnl"].sum(),
        "eq_series":eq,"dd_series":dds,"trades_df":tdf,
    }


# ==============================================================================
#  GRID SEARCH  (v2.1: z_composite key, no stop in loop)
# ==============================================================================
def run_grid_search(features, cfg):
    print("="*74)
    print("  STEP 2  |  GRID SEARCH  v2.1  (Training 2010-2018)")
    print("="*74)
    print("  KEY CHANGES vs v2.0:")
    print("    - Composite 1d+3d z-score signal (smoother, fewer false entries)")
    print("    - Stop-loss DISABLED (time-exit only)")
    print("    - Shorter hold periods tested (2,3,4 days)")
    print("    - Tighter vol filter options (2.0%, 2.5%, 3.0%)\n")

    pg    = cfg["param_grid"]
    keys  = list(pg.keys())
    combos= list(itertools.product(*[pg[k] for k in keys]))
    s, e  = PHASE_SPANS["Training"]
    print(f"  Combinations: {len(combos)} | Objective: Sharpe (primary), Calmar (secondary)")
    print(f"  Period      : {s} to {e}\n")

    results = []
    for vals in combos:
        params = {**cfg, **dict(zip(keys, vals))}
        params["holding_period"] = int(params["holding_period"])
        params["cooldown_days"]  = int(params["cooldown_days"])
        params["stop_loss_pct"]  = None   # always disabled in grid search
        try:
            sigs, sc = build_signals(features, params, s, e)
            tdf, edf, _ = run_backtest(features, sigs, sc, params, s, e)
            if tdf.empty or len(tdf) < 20: continue
            m   = compute_metrics(tdf, edf, params, "Training")
            row = dict(zip(keys, vals))
            row.update({"sharpe":round(m["sharpe"],4),
                        "calmar":round(m["calmar"],4),
                        "cagr":round(m["cagr"],2),
                        "max_dd":round(m["max_dd"],2),
                        "win_rate":round(m["win_rate"],2),
                        "n_trades":m["n_trades"],
                        "pf":round(m["profit_factor"],4),
                        "cost_pct":round(m["cost_to_gross_pct"],1)})
            results.append(row)
        except Exception:
            pass

    if not results:
        print("  ERROR: No valid combinations. Check data.\n")
        return {**cfg}, pd.DataFrame()

    grid_df = (pd.DataFrame(results)
               .sort_values(["sharpe","calmar"], ascending=False)
               .reset_index(drop=True))

    print("  Top 10 combinations (Training period):")
    print(tabulate(grid_df.head(10), headers="keys",
                   tablefmt="simple", showindex=True, floatfmt=".3f"))
    print()

    best   = grid_df.iloc[0]
    bparams = {**cfg}
    for k in keys:
        bparams[k] = int(best[k]) if k in ["holding_period","cooldown_days"] else best[k]
    bparams["stop_loss_pct"] = None  # always off

    print("  BEST PARAMS (locked for Validation + Execution):")
    for k in keys:
        print(f"    {k:<22} = {bparams[k]}")
    print(f"    stop_loss_pct       = disabled (v2.1 policy)")
    print(f"    -> Sharpe  = {best['sharpe']:.4f}")
    print(f"    -> Calmar  = {best['calmar']:.4f}")
    print(f"    -> CAGR    = {best['cagr']:.2f}%")
    print(f"    -> MaxDD   = {best['max_dd']:.2f}%")
    print(f"    -> CostRatio = {best['cost_pct']:.1f}% of gross\n")

    return bparams, grid_df


# ==============================================================================
#  PHASE COMPARISON  (added cost/gross transparency)
# ==============================================================================
def print_comparison(pm, best_params):
    print("\n" + "="*74)
    print("  WALK-FORWARD REGIME SUMMARY  v2.1  (params locked after Training)")
    print("="*74)
    rows = []
    for label, fn in [
        ("Period",              lambda m: PHASE_SPANS[m["phase"]][0][:4]+"-"+PHASE_SPANS[m["phase"]][1][:4]),
        # intial captal 
        ("Initial Capital",     lambda m: f"INR {m['total_ret_inr']-m['total_ret_pct']/100*m['total_ret_inr']:>+,.0f}"),
        ("Final Equity",        lambda m: f"INR {m['total_ret_inr']:>+,.0f}"),
        ("Total Return %",      lambda m: f"{m['total_ret_pct']:>+.2f}%"),
        ("CAGR",                lambda m: f"{m['cagr']:>+.2f}%"),
        ("Sharpe Ratio",        lambda m: f"{m['sharpe']:.4f}"),
        ("Sortino Ratio",       lambda m: f"{m['sortino']:.4f}"),
        ("Calmar Ratio",        lambda m: f"{m['calmar']:.4f}"),
        ("Max Drawdown",        lambda m: f"{m['max_dd']:.2f}%"),
        ("Win Rate",            lambda m: f"{m['win_rate']:.2f}%"),
        ("Profit Factor",       lambda m: f"{m['profit_factor']:.4f}"),
        ("Expectancy/Trade",    lambda m: f"INR {m['expectancy']:>+.0f}"),
        ("Total Trades",        lambda m: f"{m['n_trades']:,}"),
        ("Stop Exits",          lambda m: f"{m['n_stops']:,}"),
        ("Avg Hold (days)",     lambda m: f"{m['avg_hold']:.2f}"),
        ("Gross PnL",           lambda m: f"INR {m['gross_pnl']:>+,.0f}"),
        ("Total Costs",         lambda m: f"INR {m['total_costs']:,.0f}"),
        ("Cost / Gross %",      lambda m: f"{m['cost_to_gross_pct']:.1f}%"),
        ("Net PnL",             lambda m: f"INR {m['total_ret_inr']:>+,.0f}"),
    ]:
        row = [label]
        for p in ["Training","Validation","Execution"]:
            try: row.append(fn(pm[p]))
            except: row.append("N/A")
        rows.append(row)

    print(tabulate(rows,
        headers=["Metric","Training (2010-18)","Validation (2019-21)","Execution (2022-24)"],
        tablefmt="simple", colalign=("left","right","right","right")))

    # Consistency
    tr_s=pm["Training"]["sharpe"]; va_s=pm["Validation"]["sharpe"]
    ex_s=pm["Execution"]["sharpe"]
    deg=(tr_s-va_s)/(abs(tr_s)+1e-9)*100
    print(f"\n  CONSISTENCY ANALYSIS:")
    print(f"    Training  Sharpe : {tr_s:.4f}")
    print(f"    Validation Sharpe: {va_s:.4f}  (degradation: {deg:.1f}%)")
    print(f"    Execution  Sharpe: {ex_s:.4f}")

    # Cost health check
    tr_cp = pm["Training"]["cost_to_gross_pct"]
    va_cp = pm["Validation"]["cost_to_gross_pct"]
    ex_cp = pm["Execution"]["cost_to_gross_pct"]
    print(f"\n  COST HEALTH CHECK:")
    print(f"    Training  cost/gross: {tr_cp:.1f}%  "
          f"({'OK' if tr_cp < 60 else 'HIGH -- signal edge too thin'})")
    print(f"    Validation cost/gross:{va_cp:.1f}%")
    print(f"    Execution  cost/gross:{ex_cp:.1f}%")
    print(f"    Rule: cost/gross < 60% = viable.  >80% = uneconomic.\n")

    if deg<30 and va_s>0.5:      v="ROBUST"
    elif va_s>0:                  v="ACCEPTABLE"
    else:                         v="CAUTION -- strategy not profitable OOS"
    print(f"  Verdict: {v}\n")


# ==============================================================================
#  TRADE LOG
# ==============================================================================
def print_trade_log(tdf, phase, max_rows=25):
    print(f"\n  --- {phase.upper()} PHASE  |  SAMPLE TRADE LOG ---")
    df=tdf.sort_values("exit_date").copy()
    for c in ["entry_date","exit_date"]:
        df[c]=pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
    df["pnl"]=df["net_pnl"].apply(lambda x: f"{'+'if x>=0 else ''}{x:,.0f}")
    df["g_pnl"]=df["gross_pnl"].apply(lambda x: f"{'+'if x>=0 else ''}{x:,.0f}")
    cols=["trade_id","symbol","direction","entry_date","exit_date",
          "entry_price","exit_price","shares","holding_days","g_pnl","pnl","ret_pct"]
    hdrs=["#","Symbol","Dir","Entry","Exit","EntPx","ExPx","Qty","Hold","GrossPnL","NetPnL","Ret%"]
    print(tabulate(df[cols].head(max_rows),headers=hdrs,
                   tablefmt="simple",showindex=False,floatfmt=".2f"))
    if len(df)>max_rows: print(f"  ... {len(df)-max_rows:,} more trades ...")


# ==============================================================================
#  DASHBOARD  (8-panel)
# ==============================================================================
def plot_dashboard(pm, grid_df, best_params, cfg):
    print("\n" + "="*74)
    print("  STEP 6  |  GENERATING WALK-FORWARD DASHBOARD  v2.1")
    print("="*74)

    PHASES=["Training","Validation","Execution"]
    PCOLS=[BLUE, AMBER, GREEN]
    cmap_rg=LinearSegmentedColormap.from_list("rg",[RED,PANEL,GREEN])
    mon_nm=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig=plt.figure(figsize=(24,28),facecolor=DARK)
    fig.suptitle(
        "WALK-FORWARD v2.1  |  CROSS-SECTIONAL MEAN REVERSION  |  NSE 25  |  LIVE DATA",
        fontsize=13,color=GOLD,fontweight="bold",y=0.993,fontfamily="monospace")
    fig.text(0.5,0.988,
        "UNiverse Capital  |  Composite Signal (1d+3d)  |  Stop-Loss Disabled  "
        "|  Training 2010-18  |  Val 2019-21  |  Exec 2022-24",
        ha="center",fontsize=7.5,color=GREY,fontfamily="monospace")

    gs=gridspec.GridSpec(5,3,figure=fig,hspace=0.52,wspace=0.30,
                         top=0.982,bottom=0.03,left=0.06,right=0.97)

    # Row 0: equity per phase
    for col,(phase,pc) in enumerate(zip(PHASES,PCOLS)):
        ax=fig.add_subplot(gs[0,col])
        eq=pm[phase]["eq_series"]; base=cfg["initial_capital"]
        ax.fill_between(eq.index,eq/1e5,base/1e5,where=eq>=base,alpha=0.13,color=GREEN)
        ax.fill_between(eq.index,eq/1e5,base/1e5,where=eq< base,alpha=0.13,color=RED)
        ax.plot(eq.index,eq/1e5,color=pc,lw=1.7)
        ax.axhline(base/1e5,color=GREY,lw=0.7,ls="--",alpha=0.5)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:.1f}L"))
        s0,e0=PHASE_SPANS[phase]
        ax.set_title(f"{phase.upper()}\n{s0[:4]}–{e0[:4]}",color=pc,fontsize=10,
                     fontweight="bold",pad=6)
        ax.grid(True,alpha=0.3)
        m=pm[phase]; fv=eq.iloc[-1]; clr=GREEN if fv>=base else RED
        txt=f"{m['total_ret_pct']:>+.1f}%  Sh:{m['sharpe']:.2f}  DD:{m['max_dd']:.1f}%"
        ax.text(0.03,0.93,txt,transform=ax.transAxes,fontsize=7,color=clr,
                fontweight="bold",
                bbox=dict(facecolor=NAVY,edgecolor=pc,alpha=0.9,boxstyle="round,pad=0.3"))

    # Row 1: drawdown
    for col,(phase,pc) in enumerate(zip(PHASES,PCOLS)):
        ax=fig.add_subplot(gs[1,col])
        dd=pm[phase]["dd_series"]
        ax.fill_between(dd.index,dd*100,0,alpha=0.65,color=RED)
        ax.plot(dd.index,dd*100,color=RED,lw=0.7)
        ax.set_title(f"DRAWDOWN  |  {phase}",color=WHITE,fontsize=9,pad=5)
        ax.set_ylabel("%",color=GREY,fontsize=8); ax.grid(True,alpha=0.3)
        mdd_v=dd.min()*100
        ax.annotate(f"  {mdd_v:.2f}%",xy=(dd.idxmin(),mdd_v),
                    color=RED,fontsize=8,fontweight="bold")

    # Row 2 L: combined overlay
    ax_comb=fig.add_subplot(gs[2,:2])
    for phase,pc in zip(PHASES,PCOLS):
        eq=pm[phase]["eq_series"]; rb=eq/eq.iloc[0]*100
        ax_comb.plot(eq.index,rb,color=pc,lw=1.7,label=phase)
        ax_comb.fill_between(eq.index,rb,100,alpha=0.06,color=pc)
    ax_comb.axhline(100,color=GREY,lw=0.8,ls="--",alpha=0.5)
    for phase in ["Validation","Execution"]:
        sep=pd.Timestamp(PHASE_SPANS[phase][0])
        ax_comb.axvline(sep,color=GREY,lw=1.0,ls=":",alpha=0.5)
    ax_comb.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:.0f}"))
    ax_comb.set_title("EQUITY OVERLAY  (each phase rebased to 100)",
                      color=WHITE,fontsize=10,pad=7,fontweight="bold")
    ax_comb.legend(fontsize=8,facecolor=PANEL,edgecolor=GREY); ax_comb.grid(True,alpha=0.32)

    # Row 2 R: grid scatter
    ax_gs=fig.add_subplot(gs[2,2])
    if not grid_df.empty:
        sh_v=grid_df["sharpe"].values; cal_v=grid_df["calmar"].clip(-5,12).values
        sc=ax_gs.scatter(sh_v,cal_v,c=grid_df["win_rate"].values,
                         cmap="RdYlGn",s=28,alpha=0.80,vmin=40,vmax=72)
        ax_gs.axvline(0,color=GREY,lw=0.7,ls="--",alpha=0.6)
        ax_gs.axhline(0,color=GREY,lw=0.7,ls="--",alpha=0.6)
        bsh=grid_df.iloc[0]["sharpe"]; bcal=grid_df.iloc[0]["calmar"]
        ax_gs.scatter([bsh],[bcal],color=GOLD,s=130,zorder=5,marker="*",label="Best")
        ax_gs.legend(fontsize=7,facecolor=PANEL,edgecolor=GREY)
        plt.colorbar(sc,ax=ax_gs,label="Win Rate %",fraction=0.035,pad=0.04)
    ax_gs.set_xlabel("Sharpe",color=GREY); ax_gs.set_ylabel("Calmar",color=GREY)
    ax_gs.set_title("GRID SEARCH  v2.1\n(composite signal, no stop)",
                    color=WHITE,fontsize=9,pad=5,fontweight="bold")
    ax_gs.grid(True,alpha=0.3)

    # Row 3: monthly heatmaps
    for col,(phase,pc) in enumerate(zip(PHASES,PCOLS)):
        ax=fig.add_subplot(gs[3,col])
        eq=pm[phase]["eq_series"]
        mon=eq.resample("ME").last().pct_change().dropna()
        dfm=mon.to_frame("ret")
        dfm["year"]=dfm.index.year; dfm["month"]=dfm.index.month
        try:
            pv=dfm.pivot(index="year",columns="month",values="ret")
            pv.columns=[mon_nm[c-1] for c in pv.columns]
            vm=max(abs(pv.values[~np.isnan(pv.values)]).max(),0.005)
            im=ax.imshow(pv.values,aspect="auto",cmap=cmap_rg,vmin=-vm,vmax=vm)
            ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns,fontsize=6,rotation=30)
            ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index.astype(str),fontsize=7)
            for r in range(pv.shape[0]):
                for c2 in range(pv.shape[1]):
                    v=pv.values[r,c2]
                    if not np.isnan(v):
                        ax.text(c2,r,f"{v*100:.1f}%",ha="center",va="center",fontsize=5.2,
                                color=WHITE if abs(v)>vm*0.4 else GREY,fontweight="bold")
        except Exception: pass
        ax.set_title(f"MONTHLY RETURNS  |  {phase}",color=pc,fontsize=9,pad=5,fontweight="bold")

    # Row 4: P&L distributions + cost bar
    for col,(phase,pc) in enumerate(zip(PHASES,PCOLS)):
        ax=fig.add_subplot(gs[4,col])
        pnl=pm[phase]["trades_df"]["net_pnl"].values
        wins=pnl[pnl>0]; loss=pnl[pnl<=0]
        ax.hist(loss,bins=30,color=RED,  alpha=0.75,label=f"L:{len(loss)}")
        ax.hist(wins,bins=30,color=GREEN,alpha=0.75,label=f"W:{len(wins)}")
        ax.axvline(0,         color=WHITE,lw=0.8,ls="--")
        ax.axvline(pnl.mean(),color=GOLD, lw=1.2,ls="--",
                   label=f"Avg:{pnl.mean():,.0f}")
        cp=pm[phase]["cost_to_gross_pct"]
        ax.set_title(f"P&L DIST  |  {phase}\ncost/gross={cp:.0f}%",
                     color=pc,fontsize=9,pad=5,fontweight="bold")
        ax.set_xlabel("Net PnL (INR)",color=GREY,fontsize=8)
        ax.legend(fontsize=7,facecolor=PANEL,edgecolor=GREY); ax.grid(True,alpha=0.3)

    # Stats box
    bp_txt=(
        "  v2.1 OPTIMAL PARAMS\n"
        f"  hold_period = {int(best_params['holding_period'])} days\n"
        f"  z_composite = {best_params['z_composite']:.2f}\n"
        f"  vol_max     = {best_params['vol_threshold']*100:.1f}%\n"
        f"  cooldown    = {int(best_params['cooldown_days'])} days\n"
        f"  stop_loss   = disabled\n\n"
        + "\n".join(
            f"  {p:<12}  Sh:{pm[p]['sharpe']:.2f}  "
            f"DD:{pm[p]['max_dd']:.1f}%  WR:{pm[p]['win_rate']:.1f}%"
            for p in PHASES)
    )
    fig.text(0.775,0.410,bp_txt,fontsize=7.8,color=WHITE,fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.7",facecolor=NAVY,
                       edgecolor=GOLD,linewidth=1.2,alpha=0.97))

    out="/mnt/user-data/outputs/walkforward_v21.png"
    os.makedirs(os.path.dirname(out),exist_ok=True)
    fig.savefig(out,dpi=155,bbox_inches="tight",facecolor=DARK)
    plt.close(fig)
    print(f"  Dashboard saved -> {out}\n")
    return out


# ==============================================================================
#  MAIN
# ==============================================================================
def main():
    t0=datetime.now()
    print("\n" + "#"*74)
    print("#  UNIVERSE CAPITAL  |  WALK-FORWARD ENGINE  v2.1  (LIVE NSE PATCH)")
    print("#  Fixes: composite signal, no stop-loss, shorter holds, tighter vol")
    print("#  Training 2010-18  |  Validation 2019-21  |  Execution 2022-24")
    print("#"*74)

    data    = load_data(BASE_CFG)
    print("  Computing features (composite 1d+3d z-score)...")
    features= build_features(data)

    best_params, grid_df = run_grid_search(features, BASE_CFG)

    print("="*74)
    print("  STEP 3-5  |  PHASE EXECUTION  (params locked)")
    print("="*74)

    phase_metrics={}
    for phase in ["Training","Validation","Execution"]:
        s,e=PHASE_SPANS[phase]
        cfg={**best_params,"initial_capital":BASE_CFG["initial_capital"]}
        print(f"\n  [{phase}]  {s}  to  {e}")
        sigs,sc=build_signals(features,cfg,s,e)
        tdf,edf,stops=run_backtest(features,sigs,sc,cfg,s,e)
        m=compute_metrics(tdf,edf,cfg,phase)
        print(f"    Trades  : {m['n_trades']:,}  |  Stop exits: {m['n_stops']}")
        print(f"    Return  : {m['total_ret_pct']:>+.2f}%  ({m['cagr']:>+.2f}% CAGR)")
        print(f"    Sharpe  : {m['sharpe']:.4f}  |  Sortino: {m['sortino']:.4f}"
              f"  |  Calmar: {m['calmar']:.4f}")
        print(f"    MaxDD   : {m['max_dd']:.2f}%")
        print(f"    WinRate : {m['win_rate']:.2f}%  |  PF: {m['profit_factor']:.4f}"
              f"  |  Exp: INR {m['expectancy']:>+.0f}")
        print(f"    Gross   : INR {m['gross_pnl']:>+,.0f}"
              f"  |  Costs: INR {m['total_costs']:,.0f}"
              f"  |  Cost/Gross: {m['cost_to_gross_pct']:.1f}%")
        print(f"    Net PnL : INR {m['total_ret_inr']:>+,.0f}")
        phase_metrics[phase]=m

    print_comparison(phase_metrics, best_params)

    print("="*74)
    print("  TRADE LOGS  (sample)")
    print("="*74)
    for phase in ["Training","Validation","Execution"]:
        print_trade_log(phase_metrics[phase]["trades_df"],phase,max_rows=15)

    plot_dashboard(phase_metrics,grid_df,best_params,BASE_CFG)

    elapsed=(datetime.now()-t0).total_seconds()
    print(f"  Runtime: {elapsed:.1f}s")
    print("#"*74+"\n")
    return phase_metrics, best_params, grid_df


if __name__=="__main__":
    main()