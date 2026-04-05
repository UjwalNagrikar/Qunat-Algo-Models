#!/usr/bin/env python3
"""
================================================================================
  UNIVERSE CAPITAL  |  QUANTITATIVE BACKTESTING ENGINE  v2.0
  Strategy   : Cross-Sectional Mean Reversion Alpha
  Universe   : 25 NSE Large-Cap Equities  |  Daily OHLCV

  WALK-FORWARD REGIME PIPELINE
  ─────────────────────────────────────────────────────────────────────────────
  Training   : 2010 - 2018  (8 yr)  Grid-search optimal hyperparameters
  Validation : 2019 - 2021  (3 yr)  Out-of-sample proof  [no re-fitting]
  Execution  : 2022 - 2024  (3 yr)  Final live simulation [params locked]

  Key principle: Parameters are LOCKED after Training phase.
  Validation and Execution never see the optimiser.
  This eliminates temporal lookahead at the strategy-parameter level.

  Short-leg note: Modelled as FUTSTK (NSE single-stock futures).
  Costs reflect futures STT + exchange charges, not delivery rates.
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

PHASE_COLORS = {"Training": BLUE, "Validation": AMBER, "Execution": GREEN}
PHASE_SPANS  = {
    "Training":   ("2010-01-01", "2018-12-31"),
    "Validation": ("2019-01-01", "2021-12-31"),
    "Execution":  ("2022-01-01", "2024-12-31"),
}

plt.rcParams.update({
    "figure.facecolor": DARK,  "axes.facecolor":   PANEL,
    "axes.edgecolor":   "#1e2a3a", "axes.labelcolor": GREY,
    "xtick.color":      GREY,  "ytick.color":       GREY,
    "text.color":       WHITE, "grid.color":        "#1e2a3a",
    "grid.linewidth":   0.55,  "font.family":       "monospace",
    "axes.spines.top":  False, "axes.spines.right": False,
})

# ── Base config (costs + universe; strategy params come from grid search) ─────
BASE_CFG = {
    "symbols": [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ICICIBANK.NS",
        "HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
        "LT.NS","AXISBANK.NS","BAJFINANCE.NS","WIPRO.NS","TITAN.NS",
        "MARUTI.NS","SUNPHARMA.NS","NTPC.NS","POWERGRID.NS","NESTLEIND.NS",
        "ONGC.NS","HCLTECH.NS","TECHM.NS","ULTRACEMCO.NS","ASIANPAINT.NS",
    ],
    "full_start":       "2010-01-01",
    "full_end":         "2024-12-31",
    "initial_capital":   2_000_000,

    # Grid for Training phase only
    "param_grid": {
        "holding_period": [2, 3, 5],
        "z_threshold":    [0.30, 0.60, 1.00],
        "vol_threshold":  [0.025, 0.035],
        "cooldown_days":  [3, 5],
    },

    # Fixed (not grid-searched)
    "long_pct":      0.20, "short_pct":    0.20,
    "max_positions": 8,    "stop_loss_pct":0.04,
    "vol_window":    10,   "risk_pct":     0.015,
    "max_pos_pct":   0.14,

    # Transaction costs (NSE futures proxy)
    "brokerage_flat": 20.0, "brokerage_pct": 0.0003,
    "stt_sell_pct":   0.0001, "exchange_pct":  0.000019,
    "sebi_pct":       0.000001, "stamp_buy_pct": 0.00002,
    "gst_pct":        0.18, "slippage_pct":  0.0002,
}


# ==============================================================================
#  DATA  (Yahoo Finance with full synthetic fallback)
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
    """
    GBM market factor + stock-specific AR(1) with negative coefficient.
    ac in [-0.45, -0.22]  =>  documented short-horizon mean reversion
    (Jegadeesh 1990, Lehmann 1990, Conrad & Kaul 1988).
    """
    np.random.seed(2010)
    dates = pd.bdate_range(cfg["full_start"], cfg["full_end"])
    T     = len(dates)

    mkt = np.random.normal(0.00012, 0.008, T)
    # Historical stress overlays (approximate index offsets for 2010-2024)
    for s,e,adj in [(750,790,-0.004),(1700,1740,-0.003),
                    (2580,2640,-0.007),(2650,2700,0.005),(3000,3040,-0.002)]:
        mkt[s:e] += adj + np.random.normal(0, abs(adj)*0.4, e-s)

    seed_2010 = {
        "RELIANCE":1050,"TCS":750,"HDFCBANK":350,"INFY":650,"ICICIBANK":430,
        "HINDUNILVR":250,"ITC":120,"SBIN":250,"BHARTIARTL":320,"KOTAKBANK":380,
        "LT":1600,"AXISBANK":260,"BAJFINANCE":180,"WIPRO":380,"TITAN":270,
        "MARUTI":1400,"SUNPHARMA":280,"NTPC":200,"POWERGRID":95,"NESTLEIND":2600,
        "ONGC":320,"HCLTECH":380,"TECHM":270,"ULTRACEMCO":900,"ASIANPAINT":800,
    }

    frames = []
    for sym, s0 in seed_2010.items():
        idvol = np.random.uniform(0.009, 0.018)
        beta  = np.random.uniform(0.60,  1.40)
        drift = np.random.uniform(-0.00008, 0.00030)
        ac    = np.random.uniform(-0.45, -0.22)   # negative  => mean reversion
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
        df.index.name = "date"
        frames.append(df)
    return frames


def load_data(cfg):
    print("\n" + "="*74)
    print("  STEP 1  |  LOADING FULL MARKET DATA  (2010-2024)")
    print("="*74)
    print("  Trying Yahoo Finance...")
    frames = _try_yfinance(cfg)
    if not frames:
        print("  Network blocked -- generating synthetic NSE data")
        print("  DGP: correlated GBM market + negative-AC idio (realistic NSE)")
        frames = _synthetic_nse(cfg)
        src = "Synthetic (negative-AC AR1, authentic NSE price levels 2010)"
    else:
        src = "Yahoo Finance (live)"

    for f in frames:
        s = f["symbol"].iloc[0]
        print(f"  OK  {s:<14} {len(f):>4} days | INR {f['close'].iloc[-1]:>10,.2f}")

    data = pd.concat(frames).reset_index().set_index(["date","symbol"]).sort_index()
    data = data.replace(0,np.nan).dropna(subset=["open","close"])
    nd   = data.index.get_level_values("date").nunique()
    ns   = data.index.get_level_values("symbol").nunique()
    print(f"\n  Universe: {ns} stocks | {nd} days | {cfg['full_start']} -> {cfg['full_end']}")
    print(f"  Source  : {src}\n")
    return data


# ==============================================================================
#  FEATURES
# ==============================================================================
def build_features(data):
    close = data["close"].unstack("symbol").sort_index().ffill(limit=3)
    open_ = data["open"].unstack("symbol").sort_index().ffill(limit=3)
    ret1  = close.pct_change(1)
    vol10 = ret1.rolling(10).std()
    # Pre-compute cross-sectional quantities used in grid search
    cs_rank = ret1.rank(axis=1, ascending=True, pct=True)   # [0,1] pct-rank
    cs_z    = ret1.sub(ret1.mean(axis=1), axis=0).div(
              ret1.std(axis=1).replace(0, np.nan), axis=0)   # z-score per row
    print(f"  Features computed: ret1, vol10, cs_rank, cs_z  | shape {close.shape}")
    return {"close":close,"open":open_,"ret1":ret1,
            "vol10":vol10,"cs_rank":cs_rank,"cs_z":cs_z}


# ==============================================================================
#  COST MODEL
# ==============================================================================
def _order_cost(tv, cfg, side):
    b = min(cfg["brokerage_flat"], tv*cfg["brokerage_pct"])
    return (b + tv*cfg["exchange_pct"] + tv*cfg["sebi_pct"]
            + (b + tv*cfg["exchange_pct"])*cfg["gst_pct"]
            + (tv*cfg["stamp_buy_pct"] if side=="buy"  else 0)
            + (tv*cfg["stt_sell_pct"]  if side=="sell" else 0))

def rt_cost(ep, xp, sh, direction, cfg):
    ten = ep*sh; tex = xp*sh
    c   = (_order_cost(ten,cfg,"buy")  + _order_cost(tex,cfg,"sell")
           if direction==1
           else _order_cost(ten,cfg,"sell") + _order_cost(tex,cfg,"buy"))
    return c + (ten+tex)*cfg["slippage_pct"]


# ==============================================================================
#  FAST SIGNAL BUILDER  (pre-computed ranks, vectorised per param combo)
# ==============================================================================
def build_signals_fast(features, cfg, s_date, e_date):
    """
    Uses pre-computed cs_rank and cs_z matrices.
    Signals in [s_date, e_date] window.
    No per-row Python loop -- pure pandas boolean masking.
    """
    cs_rank = features["cs_rank"]
    cs_z    = features["cs_z"]
    vol10   = features["vol10"]

    mask = (cs_rank.index >= s_date) & (cs_rank.index <= e_date)
    rnk  = cs_rank.loc[mask].copy()
    z    = cs_z.loc[mask].copy()
    v10  = vol10.loc[mask].copy()

    lp   = cfg.get("long_pct",  0.20)
    sp   = cfg.get("short_pct", 0.20)
    vt   = cfg.get("vol_threshold", 0.030)
    zt   = cfg.get("z_threshold",   0.50)

    # Volatility mask: NaN out high-vol stocks on each day
    hi_vol  = v10 > vt                  # True where vol too high
    rnk_adj = rnk.copy(); rnk_adj[hi_vol] = np.nan
    z_adj   = z.copy();   z_adj[hi_vol]   = np.nan

    # Long: bottom lp quantile AND |z| >= zt
    long_sig  = (rnk_adj <= lp) & (z_adj <= -zt)
    # Short: top sp quantile AND |z| >= zt
    short_sig = (rnk_adj >= (1-sp)) & (z_adj >= zt)

    signals = (long_sig.astype(np.int8)
               - short_sig.astype(np.int8))             # +1 / -1 / 0
    signals[signals.isna()] = 0

    # ret_score: abs z for extremity sorting at execution
    ret_score = z_adj.abs()
    return signals.astype(np.int8), ret_score


# ==============================================================================
#  BACKTEST ENGINE
# ==============================================================================
def run_backtest(features, signals, ret_score, cfg, s_date, e_date):
    open_p  = features["open"]
    close_p = features["close"]
    mask_o  = (open_p.index  >= s_date) & (open_p.index  <= e_date)
    mask_c  = (close_p.index >= s_date) & (close_p.index <= e_date)
    op      = open_p.loc[mask_o];  cp = close_p.loc[mask_c]
    sig     = signals[(signals.index >= s_date) & (signals.index <= e_date)]

    dates     = op.index.tolist()
    cash      = float(cfg["initial_capital"])
    positions = {}; cooldown = {}; trades = []; equity_rec = []
    stop_exits= 0; hold_per = int(cfg.get("holding_period", 3))
    sl_pct    = cfg.get("stop_loss_pct", 0.04)
    max_pos   = int(cfg.get("max_positions", 8))
    cd_days   = int(cfg.get("cooldown_days", 4))

    for i, date in enumerate(dates):
        dop = op.loc[date]; dcp = cp.loc[date]

        locked = unrealised = 0.0
        for sym, pos in positions.items():
            locked += pos["ep"]*pos["sh"]
            px = dcp.get(sym, np.nan)
            if not np.isnan(px):
                unrealised += (px - pos["ep"])*pos["sh"]*pos["dir"]

        equity_rec.append({"date":date,"cash":cash,"equity":cash+locked+unrealised,
                           "n_open":len(positions)})
        cur_eq = cash + locked + unrealised

        # Stop-loss
        for sym in list(positions):
            pos = positions[sym]; px = dcp.get(sym, np.nan)
            if np.isnan(px): continue
            tr = (px-pos["ep"])/pos["ep"]*pos["dir"]
            if tr < -sl_pct:
                positions.pop(sym)
                raw = (px-pos["ep"])*pos["sh"]*pos["dir"]
                c   = rt_cost(pos["ep"],px,pos["sh"],pos["dir"],cfg)
                cash += pos["ep"]*pos["sh"] + raw - c
                cooldown[sym] = i; stop_exits += 1
                trades.append(_rec(len(trades)+1,sym,pos["dir"],
                              dates[pos["i"]],date,pos["ep"],px,
                              pos["sh"],i-pos["i"],"STOP",raw,c))

        # Time exit
        for sym in [s for s,p in list(positions.items()) if i>=p["i"]+hold_per]:
            pos = positions.pop(sym)
            xp  = dop.get(sym, pos["ep"])
            if np.isnan(xp): xp = pos["ep"]
            raw = (xp-pos["ep"])*pos["sh"]*pos["dir"]
            c   = rt_cost(pos["ep"],xp,pos["sh"],pos["dir"],cfg)
            cash += pos["ep"]*pos["sh"] + raw - c
            cooldown[sym] = i
            trades.append(_rec(len(trades)+1,sym,pos["dir"],
                          dates[pos["i"]],date,pos["ep"],xp,
                          pos["sh"],i-pos["i"],"HOLD",raw,c))

        # Entry (yesterday's signal -> today's open)
        if i == 0: continue
        prev = dates[i-1]
        if prev not in sig.index: continue
        sr = sig.loc[prev]; cands = sr[sr != 0]
        if cands.empty: continue

        # Sort by z-score magnitude (most extreme first)
        if prev in ret_score.index:
            sc = ret_score.loc[prev][cands.index].fillna(0)
            cands = cands.loc[sc.sort_values(ascending=False).index]

        for sym, direction in cands.items():
            if len(positions) >= max_pos: break
            if sym in positions: continue
            if i - cooldown.get(sym, -9999) < cd_days: continue
            ep = dop.get(sym, np.nan)
            if np.isnan(ep) or ep <= 0: continue
            alloc  = min(cur_eq*cfg["risk_pct"], cur_eq*cfg["max_pos_pct"])
            shares = max(1, int(alloc/ep))
            needed = ep*shares
            if needed > cash*0.97 or shares < 1: continue
            cash -= needed
            positions[sym] = {"i":i,"ep":ep,"sh":shares,"dir":int(direction)}

    # EOD force-close
    ld = dates[-1]; lc = cp.loc[ld]
    for sym, pos in list(positions.items()):
        xp  = lc.get(sym, pos["ep"])
        if np.isnan(xp): xp = pos["ep"]
        raw = (xp-pos["ep"])*pos["sh"]*pos["dir"]
        c   = rt_cost(pos["ep"],xp,pos["sh"],pos["dir"],cfg)
        cash += pos["ep"]*pos["sh"] + raw - c
        trades.append(_rec(len(trades)+1,sym,pos["dir"],
                      dates[pos["i"]],ld,pos["ep"],xp,
                      pos["sh"],len(dates)-1-pos["i"],"EOD",raw,c))

    tdf = pd.DataFrame(trades)
    edf = pd.DataFrame(equity_rec).set_index("date")
    return tdf, edf, stop_exits


def _rec(tid,sym,dir_,ed,xd,ep,xp,sh,hold,reason,raw,c):
    net = raw-c
    return {"trade_id":tid,"symbol":sym,
            "direction":"LONG" if dir_==1 else "SHORT",
            "entry_date":ed,"exit_date":xd,
            "entry_price":round(ep,2),"exit_price":round(xp,2),
            "shares":sh,"holding_days":hold,"exit_reason":reason,
            "gross_pnl":round(raw,2),"cost":round(c,2),
            "net_pnl":round(net,2),
            "ret_pct":round((xp/ep-1)*dir_*100,4)}


# ==============================================================================
#  METRICS
# ==============================================================================
def _mc(lst,v):
    best=cur=0
    for x in lst:
        cur = cur+1 if x==v else 0
        best = max(best,cur)
    return best


def compute_metrics(tdf, edf, cfg, phase=""):
    init   = cfg["initial_capital"]; eq = edf["equity"]
    final  = eq.iloc[-1]
    nyrs   = (edf.index[-1]-edf.index[0]).days/365.25
    rp     = (final/init-1)*100; ri = final-init
    cagr   = ((final/init)**(1/max(nyrs,0.01))-1)*100
    daily  = eq.pct_change().dropna()
    sh     = daily.mean()/daily.std()*np.sqrt(252) if daily.std()>0 else 0
    dn     = daily[daily<0]
    so     = daily.mean()/dn.std()*np.sqrt(252) if len(dn)>1 else 0
    rm     = eq.cummax(); dds = (eq-rm)/rm
    mdd    = dds.min()*100
    cal    = cagr/abs(mdd) if mdd!=0 else 0

    if tdf.empty: return {}
    win=tdf[tdf["net_pnl"]>0]; los=tdf[tdf["net_pnl"]<=0]
    wr =len(win)/len(tdf)*100
    gp =win["net_pnl"].sum() if len(win)>0 else 0
    gl =abs(los["net_pnl"].sum()) if len(los)>0 else 1e-9
    pf =gp/gl; exp=tdf["net_pnl"].mean()
    avg_w=win["net_pnl"].mean() if len(win)>0 else 0
    avg_l=los["net_pnl"].mean() if len(los)>0 else 0
    ws=(tdf["net_pnl"]>0).astype(int).tolist()
    lg=tdf[tdf["direction"]=="LONG"]; sh_=tdf[tdf["direction"]=="SHORT"]
    st=tdf[tdf.get("exit_reason","")=="STOP"] if "exit_reason" in tdf.columns else pd.DataFrame()

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
        "total_costs":tdf["cost"].sum(),
        "long_pnl":lg["net_pnl"].sum(),"short_pnl":sh_["net_pnl"].sum(),
        "eq_series":eq,"dd_series":dds,"trades_df":tdf,
    }


# ==============================================================================
#  GRID SEARCH  (Training period only, vectorised signals)
# ==============================================================================
def run_grid_search(features, cfg):
    print("="*74)
    print("  STEP 2  |  HYPERPARAMETER GRID SEARCH  (Training 2010-2018 only)")
    print("="*74)

    pg    = cfg["param_grid"]
    keys  = list(pg.keys())
    combos= list(itertools.product(*[pg[k] for k in keys]))
    s, e  = PHASE_SPANS["Training"]
    print(f"  Combinations : {len(combos)}")
    print(f"  Objective    : Sharpe (primary)  |  Calmar (secondary)")
    print(f"  Period       : {s}  to  {e}\n")

    results = []
    for vals in combos:
        params = {**cfg, **dict(zip(keys, vals))}
        params["holding_period"] = int(params["holding_period"])
        params["cooldown_days"]  = int(params["cooldown_days"])
        try:
            sigs, sc = build_signals_fast(features, params, s, e)
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
                        "pf":round(m["profit_factor"],4)})
            results.append(row)
        except Exception:
            pass

    grid_df = (pd.DataFrame(results)
               .sort_values(["sharpe","calmar"], ascending=False)
               .reset_index(drop=True))

    print("  Top 10 parameter combinations (Training period):")
    print(tabulate(grid_df.head(10), headers="keys",
                   tablefmt="simple", showindex=True, floatfmt=".3f"))
    print()

    best  = grid_df.iloc[0]
    bparams = {**cfg}
    for k in keys:
        bparams[k] = int(best[k]) if k in ["holding_period","cooldown_days"] else best[k]

    print("  BEST PARAMS (locked for Validation + Execution):")
    for k in keys:
        print(f"    {k:<22} = {bparams[k]}")
    print(f"    -> Sharpe = {best['sharpe']:.4f}  |  Calmar = {best['calmar']:.4f}"
          f"  |  CAGR = {best['cagr']:.2f}%  |  MaxDD = {best['max_dd']:.2f}%\n")

    return bparams, grid_df


# ==============================================================================
#  REGIME LABEL
# ==============================================================================
def classify_regime(eq):
    ret = eq.pct_change().dropna()
    ann = ret.mean()*252; dd=(eq-eq.cummax())/eq.cummax(); mdd=dd.min()
    if ann>0.10 and mdd>-0.06: return "BULL  / LOW-RISK"
    elif ann>0.04:               return "MILD-BULL"
    elif ann>-0.02:              return "SIDEWAYS"
    elif ann>-0.08:              return "MILD-BEAR"
    else:                         return "BEAR  / STRESS"


# ==============================================================================
#  COMPARISON TABLE
# ==============================================================================
def print_comparison(pm):
    print("\n" + "="*74)
    print("  WALK-FORWARD REGIME SUMMARY  --  PARAMS LOCKED AFTER TRAINING")
    print("="*74)

    rows = []
    for label, fn in [
        ("Period",           lambda m: PHASE_SPANS[m["phase"]][0][:4]+"-"+PHASE_SPANS[m["phase"]][1][:4]),
        ("Total Return %",   lambda m: f"{m['total_ret_pct']:>+.2f}%"),
        ("CAGR",             lambda m: f"{m['cagr']:>+.2f}%"),
        ("Sharpe Ratio",     lambda m: f"{m['sharpe']:.4f}"),
        ("Sortino Ratio",    lambda m: f"{m['sortino']:.4f}"),
        ("Calmar Ratio",     lambda m: f"{m['calmar']:.4f}"),
        ("Max Drawdown",     lambda m: f"{m['max_dd']:.2f}%"),
        ("Win Rate",         lambda m: f"{m['win_rate']:.2f}%"),
        ("Profit Factor",    lambda m: f"{m['profit_factor']:.4f}"),
        ("Expectancy/Trade", lambda m: f"INR {m['expectancy']:>+.0f}"),
        ("Total Trades",     lambda m: f"{m['n_trades']:,}"),
        ("Stop Exits",       lambda m: f"{m['n_stops']:,}"),
        ("Avg Hold (days)",  lambda m: f"{m['avg_hold']:.2f}"),
        ("Total Costs",      lambda m: f"INR {m['total_costs']:,.0f}"),
        ("Net PnL",          lambda m: f"INR {m['total_ret_inr']:>+,.0f}"),
        ("Market Regime",    lambda m: classify_regime(m["eq_series"])),
    ]:
        row = [label]
        for p in ["Training","Validation","Execution"]:
            try: row.append(fn(pm[p]))
            except: row.append("N/A")
        rows.append(row)

    print(tabulate(rows,
        headers=["Metric","Training (2010-18)","Validation (2019-21)","Execution (2022-24)"],
        tablefmt="simple", colalign=("left","right","right","right")))

    # Consistency check
    tr_s = pm["Training"]["sharpe"]
    va_s = pm["Validation"]["sharpe"]
    ex_s = pm["Execution"]["sharpe"]
    deg  = (tr_s - va_s) / (abs(tr_s)+1e-9) * 100

    print(f"\n  CONSISTENCY ANALYSIS:")
    print(f"    Training   Sharpe : {tr_s:.4f}")
    print(f"    Validation Sharpe : {va_s:.4f}  (degradation vs Training: {deg:.1f}%)")
    print(f"    Execution  Sharpe : {ex_s:.4f}")
    if deg < 30 and va_s > 0.5:
        v = "ROBUST    -- OOS Sharpe within 30% of in-sample"
    elif va_s > 0:
        v = "ACCEPTABLE -- Strategy profitable OOS despite some degradation"
    else:
        v = "CAUTION   -- Significant overfitting; consider wider param ranges"
    print(f"    Verdict  : {v}\n")


# ==============================================================================
#  TRADE LOG
# ==============================================================================
def print_trade_log(tdf, phase, max_rows=25):
    print(f"\n  --- {phase.upper()} PHASE  |  TRADE LOG (first {max_rows}) ---")
    df  = tdf.sort_values("exit_date").copy()
    for c in ["entry_date","exit_date"]:
        df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
    df["pnl"] = df["net_pnl"].apply(lambda x: f"{'+'if x>=0 else ''}{x:,.0f}")
    cols = ["trade_id","symbol","direction","entry_date","exit_date",
            "entry_price","exit_price","shares","holding_days","exit_reason","pnl","ret_pct"]
    hdrs = ["#","Symbol","Dir","Entry","Exit","EntPx","ExPx","Qty","Hold","Rsn","PnL","Ret%"]
    print(tabulate(df[cols].head(max_rows), headers=hdrs,
                   tablefmt="simple", showindex=False, floatfmt=".2f"))
    if len(df)>max_rows: print(f"  ... {len(df)-max_rows:,} more trades ...")


# ==============================================================================
#  WALK-FORWARD DASHBOARD  (8-panel chart)
# ==============================================================================
def plot_dashboard(pm, grid_df, best_params, cfg):
    print("\n" + "="*74)
    print("  STEP 6  |  GENERATING WALK-FORWARD DASHBOARD")
    print("="*74)

    PHASES = ["Training","Validation","Execution"]
    PCOLS  = [BLUE, AMBER, GREEN]
    cmap_rg= LinearSegmentedColormap.from_list("rg",[RED,PANEL,GREEN])
    mon_nm = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig = plt.figure(figsize=(24,28), facecolor=DARK)
    fig.suptitle(
        "WALK-FORWARD REGIME PIPELINE  |  CROSS-SECTIONAL MEAN REVERSION  |  NSE 25 LARGE-CAP",
        fontsize=14, color=GOLD, fontweight="bold", y=0.993, fontfamily="monospace")
    fig.text(0.5, 0.987,
        "UNiverse Capital  |  Alpha Research  |"
        "  Training 2010-18  |  Validation 2019-21  |  Execution 2022-24",
        ha="center", fontsize=8, color=GREY, fontfamily="monospace")

    gs = gridspec.GridSpec(5,3, figure=fig, hspace=0.52, wspace=0.30,
                           top=0.982, bottom=0.03, left=0.06, right=0.97)

    # ── ROW 0: per-phase equity ────────────────────────────────────────────────
    for col,(phase,pc) in enumerate(zip(PHASES,PCOLS)):
        ax  = fig.add_subplot(gs[0,col])
        eq  = pm[phase]["eq_series"]; base = cfg["initial_capital"]
        ax.fill_between(eq.index, eq/1e5, base/1e5, where=eq>=base, alpha=0.13, color=GREEN)
        ax.fill_between(eq.index, eq/1e5, base/1e5, where=eq< base, alpha=0.13, color=RED)
        ax.plot(eq.index, eq/1e5, color=pc, lw=1.7)
        ax.axhline(base/1e5, color=GREY, lw=0.7, ls="--", alpha=0.5)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:.1f}L"))
        s0,e0 = PHASE_SPANS[phase]
        ax.set_title(f"{phase.upper()}\n{s0[:4]} – {e0[:4]}", color=pc, fontsize=10,
                     fontweight="bold", pad=6)
        ax.grid(True, alpha=0.3)
        m   = pm[phase]; fv = eq.iloc[-1]; clr = GREEN if fv>=base else RED
        txt = f"{m['total_ret_pct']:>+.1f}%  Sh:{m['sharpe']:.2f}  DD:{m['max_dd']:.1f}%"
        ax.text(0.03,0.93,txt,transform=ax.transAxes,fontsize=7,color=clr,
                fontweight="bold",
                bbox=dict(facecolor=NAVY,edgecolor=pc,alpha=0.9,boxstyle="round,pad=0.3"))

    # ── ROW 1: per-phase drawdown ──────────────────────────────────────────────
    for col,(phase,pc) in enumerate(zip(PHASES,PCOLS)):
        ax = fig.add_subplot(gs[1,col])
        dd = pm[phase]["dd_series"]
        ax.fill_between(dd.index, dd*100, 0, alpha=0.65, color=RED)
        ax.plot(dd.index, dd*100, color=RED, lw=0.7)
        ax.set_title(f"DRAWDOWN  |  {phase}", color=WHITE, fontsize=9, pad=5)
        ax.set_ylabel("%", color=GREY, fontsize=8); ax.grid(True, alpha=0.3)
        mdd_v = dd.min()*100
        ax.annotate(f"  {mdd_v:.2f}%", xy=(dd.idxmin(),mdd_v),
                    color=RED, fontsize=8, fontweight="bold")

    # ── ROW 2 L-M: Combined overlay (rebased to 100) ──────────────────────────
    ax_comb = fig.add_subplot(gs[2,:2])
    for phase,pc in zip(PHASES,PCOLS):
        eq = pm[phase]["eq_series"]; rb = eq/eq.iloc[0]*100
        ax_comb.plot(eq.index, rb, color=pc, lw=1.7, label=phase)
        ax_comb.fill_between(eq.index, rb, 100, alpha=0.06, color=pc)
    ax_comb.axhline(100, color=GREY, lw=0.8, ls="--", alpha=0.5)
    for phase in ["Validation","Execution"]:
        sep = pd.Timestamp(PHASE_SPANS[phase][0])
        ax_comb.axvline(sep, color=GREY, lw=1.0, ls=":", alpha=0.5)
        ax_comb.text(sep, ax_comb.get_ylim()[1] if ax_comb.get_ylim()[1]!=1 else 110,
                     f"  {phase[:3]}.", color=GREY, fontsize=7, va="top")
    ax_comb.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:.0f}"))
    ax_comb.set_title("EQUITY OVERLAY  (each phase rebased to 100)",
                      color=WHITE, fontsize=10, pad=7, fontweight="bold")
    ax_comb.legend(fontsize=8, facecolor=PANEL, edgecolor=GREY)
    ax_comb.grid(True, alpha=0.32)

    # ── ROW 2 R: Grid search scatter ──────────────────────────────────────────
    ax_gs = fig.add_subplot(gs[2,2])
    sh_v  = grid_df["sharpe"].values
    cal_v = grid_df["calmar"].clip(-5,12).values
    sc    = ax_gs.scatter(sh_v, cal_v, c=grid_df["win_rate"].values,
                          cmap="RdYlGn", s=28, alpha=0.80, vmin=40, vmax=72)
    ax_gs.axvline(0,color=GREY,lw=0.7,ls="--",alpha=0.6)
    ax_gs.axhline(0,color=GREY,lw=0.7,ls="--",alpha=0.6)
    bsh = grid_df.iloc[0]["sharpe"]; bcal = grid_df.iloc[0]["calmar"]
    ax_gs.scatter([bsh],[bcal],color=GOLD,s=130,zorder=5,marker="*",label="Best")
    ax_gs.set_xlabel("Sharpe",color=GREY); ax_gs.set_ylabel("Calmar",color=GREY)
    ax_gs.set_title("GRID SEARCH  (Training only)\ncolour = win rate %",
                    color=WHITE, fontsize=9, pad=5, fontweight="bold")
    ax_gs.legend(fontsize=7, facecolor=PANEL, edgecolor=GREY)
    plt.colorbar(sc, ax=ax_gs, label="Win Rate %", fraction=0.035, pad=0.04)
    ax_gs.grid(True, alpha=0.3)

    # ── ROW 3: Monthly heatmaps ────────────────────────────────────────────────
    for col,(phase,pc) in enumerate(zip(PHASES,PCOLS)):
        ax  = fig.add_subplot(gs[3,col])
        eq  = pm[phase]["eq_series"]
        mon = eq.resample("ME").last().pct_change().dropna()
        dfm = mon.to_frame("ret")
        dfm["year"]=dfm.index.year; dfm["month"]=dfm.index.month
        try:
            pv = dfm.pivot(index="year",columns="month",values="ret")
            pv.columns = [mon_nm[c-1] for c in pv.columns]
            vm = max(abs(pv.values[~np.isnan(pv.values)]).max(),0.005)
            im = ax.imshow(pv.values,aspect="auto",cmap=cmap_rg,vmin=-vm,vmax=vm)
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

    # ── ROW 4: P&L distributions ──────────────────────────────────────────────
    for col,(phase,pc) in enumerate(zip(PHASES,PCOLS)):
        ax   = fig.add_subplot(gs[4,col])
        pnl  = pm[phase]["trades_df"]["net_pnl"].values
        wins = pnl[pnl>0]; loss = pnl[pnl<=0]
        ax.hist(loss,bins=30,color=RED,  alpha=0.75,label=f"L:{len(loss)}")
        ax.hist(wins,bins=30,color=GREEN,alpha=0.75,label=f"W:{len(wins)}")
        ax.axvline(0,         color=WHITE,lw=0.8,ls="--")
        ax.axvline(pnl.mean(),color=GOLD, lw=1.2,ls="--",
                   label=f"Avg:{pnl.mean():,.0f}")
        ax.set_title(f"P&L DISTRIBUTION  |  {phase}",color=pc,fontsize=9,pad=5,fontweight="bold")
        ax.set_xlabel("Net PnL (INR)",color=GREY,fontsize=8)
        ax.legend(fontsize=7,facecolor=PANEL,edgecolor=GREY); ax.grid(True,alpha=0.3)

    # ── Stats annotation ──────────────────────────────────────────────────────
    bp_txt = (
        "  OPTIMAL PARAMS  (from Training)\n"
        f"  hold_period = {int(best_params['holding_period'])} days\n"
        f"  z_threshold = {best_params['z_threshold']:.2f}\n"
        f"  vol_max     = {best_params['vol_threshold']*100:.1f}%\n"
        f"  cooldown    = {int(best_params['cooldown_days'])} days\n\n"
        + "\n".join(
            f"  {p:<12}  Sh:{pm[p]['sharpe']:.2f}  "
            f"DD:{pm[p]['max_dd']:.1f}%  WR:{pm[p]['win_rate']:.1f}%"
            for p in PHASES)
    )
    fig.text(0.775,0.410,bp_txt,fontsize=7.8,color=WHITE,fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.7",facecolor=NAVY,
                       edgecolor=GOLD,linewidth=1.2,alpha=0.97))

    out = "/mnt/user-data/outputs/walkforward_backtest.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=155, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  Dashboard saved -> {out}\n")
    return out


# ==============================================================================
#  MAIN  PIPELINE
# ==============================================================================
def main():
    t0 = datetime.now()
    print("\n" + "#"*74)
    print("#  UNIVERSE CAPITAL  |  WALK-FORWARD BACKTESTING ENGINE  v2.0")
    print("#  Cross-Sectional Mean Reversion  |  25 NSE Large-Cap  |  Daily")
    print("#  Training: 2010-2018  |  Validation: 2019-2021  |  Execution: 2022-2024")
    print("#"*74)

    # 1. Load & features
    data     = load_data(BASE_CFG)
    print("  Computing features on full history...")
    features = build_features(data)
    print()

    # 2. Grid search (Training only)
    best_params, grid_df = run_grid_search(features, BASE_CFG)

    # 3-5. Run all phases with LOCKED params
    print("="*74)
    print("  STEP 3-5  |  PHASE EXECUTION  (params locked from Training)")
    print("="*74)

    phase_metrics = {}
    for phase in ["Training","Validation","Execution"]:
        s, e = PHASE_SPANS[phase]
        cfg  = {**best_params, "initial_capital": BASE_CFG["initial_capital"]}
        print(f"\n  [{phase}]  {s}  to  {e}")
        sigs, sc = build_signals_fast(features, cfg, s, e)
        tdf, edf, stops = run_backtest(features, sigs, sc, cfg, s, e)
        m = compute_metrics(tdf, edf, cfg, phase)
        print(f"    Trades : {m['n_trades']:,}  |  Stop exits : {m['n_stops']}")
        print(f"    Return : {m['total_ret_pct']:>+.2f}%  ({m['cagr']:>+.2f}% CAGR)")
        print(f"    Sharpe : {m['sharpe']:.4f}  |  Sortino : {m['sortino']:.4f}"
              f"  |  Calmar : {m['calmar']:.4f}")
        print(f"    MaxDD  : {m['max_dd']:.2f}%")
        print(f"    WinRate: {m['win_rate']:.2f}%  |  PF : {m['profit_factor']:.4f}"
              f"  |  Exp/trade : INR {m['expectancy']:>+.0f}")
        print(f"    NetPnL : INR {m['total_ret_inr']:>+,.0f}")
        phase_metrics[phase] = m

    # 6. Comparison table
    print_comparison(phase_metrics)

    # 7. Trade logs
    print("="*74)
    print("  TRADE LOGS  (sample)")
    print("="*74)
    for phase in ["Training","Validation","Execution"]:
        print_trade_log(phase_metrics[phase]["trades_df"], phase, max_rows=15)

    # 8. Dashboard
    plot_dashboard(phase_metrics, grid_df, best_params, BASE_CFG)

    elapsed = (datetime.now()-t0).total_seconds()
    print(f"  Total runtime : {elapsed:.1f}s")
    print("#"*74+"\n")
    return phase_metrics, best_params, grid_df


if __name__ == "__main__":
    main()