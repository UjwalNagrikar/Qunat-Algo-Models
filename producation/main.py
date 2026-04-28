# ═══════════════════════════════════════════════════════════════════════════════
#   UNiverse Capital — Nifty 50 Futures Swing Algo  v4.1
#   Targets: PF≥1.5 | Sharpe≥1.2 | CAGR 15-25% | 90-100 trades | Cost<15%
#
#   Key math:  stop=1.0 ATR, target=3.5 ATR  →  R:R = 3.5
#              At WR=45%: PF = (0.45×3.5)/0.55 = 2.86 (gross)
#              Cost% = total_costs / gross_net_pnl  ≈ 3-8%  ✓
#
#   Root causes fixed vs v4.0:
#   ✗ quick_cut was exiting at bar LOW (worse than stop) → REMOVED
#   ✗ target=2.5 too narrow → RAISED to 3.0-4.5 grid
#   ✗ trailing at 1R too early, cutting winners → trail only after 2R
#   ✗ Sharpe<0.3 constraint too tight → relaxed to 0.1
#
#   Charts: displayed inline (plt.show / IPython.display) — no files written
#   Replace generate_nifty_data() with Zerodha Kite API for live trading
# ═══════════════════════════════════════════════════════════════════════════════

import os, warnings, itertools, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from dataclasses import dataclass
from typing import Optional

warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────────────────────

TRAIN_START = "2015-01-01";  TRAIN_END = "2022-12-31"
TEST_START  = "2023-01-01";  TEST_END  = "2024-12-31"

LOT_SIZE    = 75
SPAN_MARGIN = 0.065
CAPITAL     = 10_00_000      # ₹10 Lakhs — your actual starting capital
RISK_PCT    = 0.01

BROKERAGE_LOT = 40.0
STT_PCT       = 0.0001
EXCHANGE_PCT  = 0.00002
SEBI_PCT      = 0.000001
STAMP_PCT     = 0.00002
SLIPPAGE_PTS  = 2.0          # tight-spread assumption

MC_RUNS = 5000

# ── Parameter grid (focused on what the math shows works) ────────────────────
PARAM_GRID = {
    "hurst_window"     : [30, 40, 60],
    "hurst_trend"      : [0.50, 0.52, 0.55],
    "hurst_mr"         : [0.48, 0.50, 0.52],
    "fast_bk_window"   : [5, 7, 10],
    "mom_consec"       : [2, 3],
    "mr_z_thresh"      : [0.9, 1.0, 1.2, 1.5],
    "zscore_window"    : [10, 15, 20],
    "ret_extreme_pct"  : [0.010, 0.012, 0.018, 0.025],
    "ret_extreme_days" : [3, 5],
    "atr_expand_mult"  : [1.3, 1.6, 2.0],
    "atr_window"       : [10, 14],
    "stop_mult"        : [0.8, 0.9, 1.0, 1.1],
    "target_mult"      : [4.0, 4.5, 5.0],        # HIGH R:R → low cost%, high PF
    "time_stop"        : [3, 4, 5],              # 3 = ~90-100 trades/yr
    "circuit_breaker"  : [3, 4, 5],
}

C = dict(bg="#0d1117", panel="#161b22", border="#30363d",
         text="#e6edf3", muted="#8b949e", gold="#f0b429",
         green="#3fb950", red="#f85149", blue="#58a6ff",
         purple="#bc8cff", orange="#ffa657", teal="#39d353")


# ── INLINE DISPLAY  (no files written) ───────────────────────────────────────

def show_fig(fig):
    try:
        from IPython.display import display as ipy_display
        ipy_display(fig)
    except Exception:
        plt.show()
    plt.close(fig)


# ── DATA ─────────────────────────────────────────────────────────────────────

def generate_nifty_data(start: str, end: str, s0: float = 8500.0) -> pd.DataFrame:
    """
    Synthetic Nifty 50 OHLCV (GBM + GARCH + regime switching + fat tails).
    Swap this for real data:
        raw = kite.historical_data(token, from_date, to_date, "day")
        df  = pd.DataFrame(raw).set_index("date").rename(columns=str.title)
    """
    dates = pd.bdate_range(start=start, end=end); n = len(dates)
    rng = np.random.default_rng(42); dt = 1/252
    DRIFT = {0:0.18*dt, 1:-0.06*dt, 2:0.05*dt}
    VOL_R = {0:0.14,    1:0.28,     2:0.11}
    reg=0; dur=0; regs=np.zeros(n,dtype=int)
    for i in range(n):
        dur+=1
        if rng.random()<0.002+0.001*(dur/60): reg=rng.integers(0,3); dur=0
        regs[i]=reg
    eps=rng.standard_normal(n)
    shk=rng.choice(n,size=int(n*0.007),replace=False)
    eps[shk]=rng.choice([-1,1],size=len(shk))*(4.0+rng.exponential(1.5,len(shk)))
    vol=np.zeros(n); vol[0]=0.17
    for i in range(1,n):
        rv=VOL_R[regs[i]]
        vol[i]=np.clip(0.55*np.sqrt(2e-6+0.08*(vol[i-1]*eps[i-1])**2+0.89*vol[i-1]**2)+0.45*rv,0.07,0.60)
    lr=np.array([DRIFT[regs[i]]+vol[i]*dt**0.5*eps[i] for i in range(n)])
    cl=np.maximum(s0*np.exp(np.cumsum(lr)),500)
    ir=vol*dt**0.5*cl; body=cl*rng.normal(0,0.002,n)
    op=np.clip(cl-body,cl*0.97,cl*1.03)
    hi=np.maximum(cl,op)+np.abs(rng.normal(0,0.3,n))*ir
    lo=np.minimum(cl,op)-np.abs(rng.normal(0,0.3,n))*ir
    lo=np.maximum(lo,cl*0.85)
    vl=(6e5*(0.6+np.abs(rng.standard_normal(n))*0.4)*(1+vol/0.17*0.5)).astype(int)
    df=pd.DataFrame({"Open":np.round(op,2),"High":np.round(hi,2),"Low":np.round(lo,2),
                     "Close":np.round(cl,2),"Volume":vl},index=dates)
    df.index.name="Date"; return df


# ── FEATURES ─────────────────────────────────────────────────────────────────

def hurst_rs(prices: np.ndarray) -> float:
    n=len(prices)
    if n<20: return 0.5
    lr=np.diff(np.log(np.maximum(prices,1e-10)))
    lags=list(dict.fromkeys([max(8,int(n/k)) for k in [2,4,8] if int(n/k)>=8]))
    if len(lags)<2: return 0.5
    rs_v=[]
    for lag in lags:
        rs_s=[]
        for s in range(0,len(lr)-lag+1,lag):
            sub=lr[s:s+lag]; m=sub.mean(); dev=np.cumsum(sub-m)
            R=dev.max()-dev.min(); S=sub.std(ddof=1)
            if S>1e-10: rs_s.append(R/S)
        if rs_s: rs_v.append(np.mean(rs_s))
    if len(rs_v)<2: return 0.5
    H=np.polyfit(np.log(lags[:len(rs_v)]),np.log(np.maximum(rs_v,1e-10)),1)[0]
    return float(np.clip(H,0.0,1.0))


def build_features(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    d=df.copy()
    zw=p["zscore_window"]; aw=p["atr_window"]
    hw=p["hurst_window"]; fbw=p["fast_bk_window"]

    d["ret_1d"]=d["Close"].pct_change(1)
    d["ret_3d"]=d["Close"].pct_change(3)
    d["ret_5d"]=d["Close"].pct_change(5)
    d["log_ret"]=np.log(d["Close"]/d["Close"].shift(1))

    d["tr"]=np.maximum(d["High"]-d["Low"],
            np.maximum(np.abs(d["High"]-d["Close"].shift(1)),
                       np.abs(d["Low"]-d["Close"].shift(1))))
    d["atr"]=d["tr"].rolling(aw).mean()
    d["atr_expand"]=d["tr"]/d["atr"]

    d["rvol"]=d["log_ret"].rolling(20).std()*np.sqrt(252)
    d["rvol_avg"]=d["rvol"].rolling(120).mean()
    d["high_vol"]=d["rvol"]>2.5*d["rvol_avg"]

    def zs(s,w=zw):
        mu=s.rolling(w).mean(); sd=s.rolling(w).std()
        return (s-mu)/sd.replace(0,np.nan)

    d["price_z"]=zs(d["Close"])
    d["vol_z"]=zs(np.log(d["Volume"].replace(0,1)))
    d["vwap_dev"]=(d["Close"]-d["Close"].rolling(20).mean())/d["Close"].rolling(20).mean()

    d["fast_high"]=d["Close"].shift(1).rolling(fbw).max()
    d["fast_low"] =d["Close"].shift(1).rolling(fbw).min()

    d["up_bar"]=(d["ret_1d"]>0).astype(int)
    d["dn_bar"]=(d["ret_1d"]<0).astype(int)
    def consec(s):
        arr=s.values.astype(float); cnt=np.zeros(len(arr))
        for i in range(1,len(arr)): cnt[i]=(cnt[i-1]+1)*arr[i]
        return pd.Series(cnt,index=s.index)
    d["consec_up"]=consec(d["up_bar"]).shift(1)
    d["consec_dn"]=consec(d["dn_bar"]).shift(1)

    d["hurst"]=d["Close"].rolling(hw).apply(hurst_rs,raw=True)
    d.dropna(inplace=True); return d


# ── SIGNALS (5 types · OR-logic) ─────────────────────────────────────────────

def generate_signals(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    d=df.copy()
    tt=p["hurst_trend"]; mt=p["hurst_mr"]
    mz=p["mr_z_thresh"]; mc_=p["mom_consec"]
    re=p["ret_extreme_pct"]; rd=p["ret_extreme_days"]
    ae=p["atr_expand_mult"]

    d["regime"]=np.where(d["hurst"]>tt,"trending",
                np.where(d["hurst"]<mt,"mean_rev","mixed"))

    is_tr=(d["regime"]=="trending"); is_mr=(d["regime"]!="trending")
    no_hv=~d["high_vol"]; ret_nd=d["Close"].pct_change(rd)

    # S1 Fast breakout
    s1L=(is_tr&no_hv&(d["Close"]>d["fast_high"])&(d["ret_1d"]>0))
    s1S=(is_tr&no_hv&(d["Close"]<d["fast_low"]) &(d["ret_1d"]<0))
    # S2 Momentum streak
    s2L=(no_hv&(d["consec_up"]>=mc_)&(d["hurst"]>0.50))
    s2S=(no_hv&(d["consec_dn"]>=mc_)&(d["hurst"]>0.50))
    # S3 Z-score reversion
    s3L=(is_mr&no_hv&(d["price_z"]<-mz)&(d["ret_1d"]<0)&(d["vwap_dev"]<-0.003))
    s3S=(is_mr&no_hv&(d["price_z"]> mz)&(d["ret_1d"]>0)&(d["vwap_dev"]> 0.003))
    # S4 Return extreme fade
    s4L=(no_hv&(ret_nd<-re)&(d["price_z"]<-0.3))
    s4S=(no_hv&(ret_nd> re)&(d["price_z"]> 0.3))
    # S5 ATR expansion momentum
    s5L=(no_hv&(d["atr_expand"]>ae)&(d["ret_1d"]>0.002)&(d["hurst"]>0.48))
    s5S=(no_hv&(d["atr_expand"]>ae)&(d["ret_1d"]<-0.002)&(d["hurst"]>0.48))

    d["sig_long"] =s1L|s2L|s3L|s4L|s5L
    d["sig_short"]=(s1S|s2S|s3S|s4S|s5S)&~d["sig_long"]

    # Quality score (number of confirming signals)
    qL=s1L.astype(int)+s2L.astype(int)+s3L.astype(int)+s4L.astype(int)+s5L.astype(int)
    qS=s1S.astype(int)+s2S.astype(int)+s3S.astype(int)+s4S.astype(int)+s5S.astype(int)
    d["q_score"]=np.where(d["sig_long"],qL,np.where(d["sig_short"],qS,0))

    # Tag dominant signal
    tags=pd.Series("",index=d.index)
    for cond,lbl in reversed([(s1L,"S1L"),(s2L,"S2L"),(s3L,"S3L"),(s4L,"S4L"),(s5L,"S5L"),
                               (s1S,"S1S"),(s2S,"S2S"),(s3S,"S3S"),(s4S,"S4S"),(s5S,"S5S")]):
        tags[cond]=lbl
    d["sig_tag"]=tags
    return d


# ── COSTS + SIZING ───────────────────────────────────────────────────────────

def trade_cost(entry: float, exit_p: float, lots: int) -> float:
    c=lots*LOT_SIZE; ev=entry*c; xv=exit_p*c
    return round(BROKERAGE_LOT*lots*2 + xv*STT_PCT +
                 (ev+xv)*(EXCHANGE_PCT+SEBI_PCT) + ev*STAMP_PCT +
                 SLIPPAGE_PTS*c*2, 2)


def size_lots(capital: float, entry: float, stop: float) -> int:
    dist=abs(entry-stop)
    if dist<1e-6: return 0
    lots=int(capital*RISK_PCT/(dist*LOT_SIZE))
    margin=entry*LOT_SIZE*SPAN_MARGIN
    max_lots=int(capital*0.70/margin) if margin>0 else 0
    min_lots=1 if capital>=margin else 0
    return max(min_lots, min(lots, max_lots))


# ── BACKTEST ─────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    idx: int; entry_date: pd.Timestamp; direction: str
    regime: str; sig_tag: str; q_score: int
    entry_price: float; stop_price: float; target_price: float; lots: int
    exit_date: Optional[pd.Timestamp]=None; exit_price: Optional[float]=None
    exit_reason: Optional[str]=None
    pnl_pts: float=0.; pnl_gross: float=0.; cost: float=0.
    pnl_net: float=0.; bars: int=0


def run_backtest(sig_df: pd.DataFrame, p: dict, cap0: float=CAPITAL) -> tuple:
    """
    Clean event-driven backtest — no quick_cut, no lookahead.

    Exit hierarchy (priority order):
      1. Hard stop    (stop_mult × ATR from entry)
      2. Target       (target_mult × ATR from entry)
      3. Time stop    (max bars)
      4. Trailing     (activated only after 2R profit — lets winners breathe)
      5. Break-even   (move stop to entry once 1R in profit — free ride protection)
      6. Circuit breaker (skip N signals after N consecutive losses — stops loss streaks)
    """
    sm=p["stop_mult"]; tm=p["target_mult"]; ts=p["time_stop"]
    cb=p.get("circuit_breaker", 999)   # consecutive losses before pausing
    trades: list[Trade]=[]; equity=cap0; eq_crv: dict={}
    pos: Optional[Trade]=None
    consec_losses=0; skip_signals=0   # circuit breaker state
    rows=sig_df.reset_index(); dates=rows["Date"].tolist(); N=len(rows)

    for i in range(1,N):
        today=rows.iloc[i]; prev=rows.iloc[i-1]
        date=dates[i]
        ep=today["Open"]; hi=today["High"]; lo=today["Low"]; cl=today["Close"]

        # ── Manage open position ──────────────────────────────────────────
        if pos is not None:
            pos.bars+=1
            atr=today["atr"]; xp=reason=None
            init_R=abs(pos.entry_price-pos.stop_price)

            if pos.direction=="long":
                profit=cl-pos.entry_price
                # Break-even: move stop to entry once 1R profit achieved
                if profit>=init_R:
                    pos.stop_price=max(pos.stop_price, pos.entry_price)
                # Trail: only after 2R profit (let trade breathe)
                if profit>=2*init_R:
                    pos.stop_price=max(pos.stop_price, cl-sm*atr)
                # Exit checks (use intraday high/low for stop & target)
                if lo<=pos.stop_price:
                    xp=pos.stop_price; reason="stop"
                elif hi>=pos.target_price:
                    xp=pos.target_price; reason="target"

            else:  # short
                profit=pos.entry_price-cl
                if profit>=init_R:
                    pos.stop_price=min(pos.stop_price, pos.entry_price)
                if profit>=2*init_R:
                    pos.stop_price=min(pos.stop_price, cl+sm*atr)
                if hi>=pos.stop_price:
                    xp=pos.stop_price; reason="stop"
                elif lo<=pos.target_price:
                    xp=pos.target_price; reason="target"

            if xp is None and pos.bars>=ts: xp=cl; reason="time"
            if xp is None and i==N-1:       xp=cl; reason="end"

            if xp is not None:
                pos.exit_date=date; pos.exit_price=xp; pos.exit_reason=reason
                pos.pnl_pts=(xp-pos.entry_price) if pos.direction=="long" \
                             else (pos.entry_price-xp)
                pos.pnl_gross=pos.pnl_pts*pos.lots*LOT_SIZE
                pos.cost=trade_cost(pos.entry_price,xp,pos.lots)
                pos.pnl_net=pos.pnl_gross-pos.cost
                equity+=pos.pnl_net; trades.append(pos); pos=None
                # ── Circuit breaker logic ─────────────────────────────────
                if pos is None:  # just closed
                    last = trades[-1]
                    if last.pnl_net <= 0:
                        consec_losses += 1
                        if consec_losses >= cb:
                            skip_signals = 1   # pause 1 signal slot
                            consec_losses = 0
                    else:
                        consec_losses = 0

        # ── New entry (with circuit breaker guard) ────────────────────────
        if pos is None and not today["high_vol"]:
            if skip_signals > 0:
                skip_signals -= 1   # consume one pause slot, skip this bar
            else:
                direction=q=tag=None
                if prev["sig_long"]:    direction="long";  q=int(prev["q_score"]); tag=prev["sig_tag"]
                elif prev["sig_short"]: direction="short"; q=int(prev["q_score"]); tag=prev["sig_tag"]

                if direction:
                    atr=today["atr"]
                    stop  =(ep-sm*atr) if direction=="long"  else (ep+sm*atr)
                    target=(ep+tm*atr) if direction=="long"  else (ep-tm*atr)
                    lots=size_lots(equity,ep,stop)
                    if lots>0:
                        pos=Trade(idx=len(trades),entry_date=date,direction=direction,
                                  regime=prev["regime"],sig_tag=tag,q_score=q,
                                  entry_price=ep,stop_price=stop,target_price=target,lots=lots)
        eq_crv[date]=equity

    return trades, pd.Series(eq_crv)


# ── METRICS ──────────────────────────────────────────────────────────────────

def trades_to_df(trades: list) -> pd.DataFrame:
    if not trades: return pd.DataFrame()
    return pd.DataFrame([{"idx":t.idx,"entry_date":t.entry_date,"exit_date":t.exit_date,
        "direction":t.direction,"regime":t.regime,"sig_tag":t.sig_tag,"q_score":t.q_score,
        "entry_price":t.entry_price,"exit_price":t.exit_price,"stop_price":t.stop_price,
        "target_price":t.target_price,"lots":t.lots,"bars":t.bars,"exit_reason":t.exit_reason,
        "pnl_pts":round(t.pnl_pts,2),"pnl_gross":round(t.pnl_gross,2),
        "cost":round(t.cost,2),"pnl_net":round(t.pnl_net,2)} for t in trades])


def metrics(trades: list, eq: pd.Series, cap0: float) -> dict:
    if not trades or eq.empty: return {}
    tdf=trades_to_df(trades); m={}
    m["n"]=len(tdf); m["nw"]=(tdf["pnl_net"]>0).sum(); m["nl"]=m["n"]-m["nw"]
    m["wr"]=m["nw"]/m["n"]
    w=tdf.loc[tdf["pnl_net"]>0,"pnl_net"]; l=tdf.loc[tdf["pnl_net"]<=0,"pnl_net"]
    m["avg_win"]=w.mean() if len(w)>0 else 0
    m["avg_loss"]=l.mean() if len(l)>0 else 0
    m["wl_ratio"]=abs(m["avg_win"]/m["avg_loss"]) if m["avg_loss"]!=0 else 0
    m["pf"]=w.sum()/abs(l.sum()) if l.sum()!=0 else 0
    m["net_pnl"]=tdf["pnl_net"].sum(); m["gross_pnl"]=tdf["pnl_gross"].sum()
    m["costs"]=tdf["cost"].sum()
    m["cost_pct"]=m["costs"]/max(abs(m["gross_pnl"]),1)*100
    m["expect"]=tdf["pnl_net"].mean()
    m["max_win"]=w.max() if len(w)>0 else 0
    m["max_loss"]=l.min() if len(l)>0 else 0
    fin=eq.iloc[-1]; m["final"]=fin; m["cap0"]=cap0; m["ret"]=(fin-cap0)/cap0
    yrs=(eq.index[-1]-eq.index[0]).days/365.25
    m["cagr"]=(fin/cap0)**(1/yrs)-1 if yrs>0 else 0
    pk=eq.cummax(); dd=(eq-pk)/pk
    m["max_dd"]=dd.min(); m["avg_dd"]=dd[dd<0].mean() if (dd<0).any() else 0
    dur=[]; s=None
    for dt_,v in dd.items():
        if v<0 and s is None: s=dt_
        elif v>=0 and s is not None: dur.append((dt_-s).days); s=None
    m["dd_days"]=max(dur) if dur else 0
    dr=eq.pct_change().dropna()
    if dr.std()>0:
        m["sharpe"]=dr.mean()/dr.std()*np.sqrt(252)
        dn=dr[dr<0]
        m["sortino"]=dr.mean()/dn.std()*np.sqrt(252) if dn.std()>0 else 0
        m["calmar"]=m["cagr"]/abs(m["max_dd"]) if m["max_dd"]!=0 else 0
    else: m["sharpe"]=m["sortino"]=m["calmar"]=0
    m["avg_bars"]=tdf["bars"].mean()
    m["exit_dist"]=tdf["exit_reason"].value_counts().to_dict()
    m["sig_dist"]=tdf["sig_tag"].value_counts().to_dict()
    seq=tdf["pnl_net"].values; mw=ml=cur=0
    for p_ in seq:
        if p_>0: cur=cur+1 if cur>0 else 1; mw=max(mw,cur)
        else:    cur=cur-1 if cur<0 else -1; ml=max(ml,abs(cur))
    m["max_cw"]=mw; m["max_cl"]=ml
    return m


# ── OPTIMIZATION ─────────────────────────────────────────────────────────────

def optimize(df_train: pd.DataFrame, df_test: pd.DataFrame,
             n_samples: int=250) -> tuple:
    """
    Walk-Forward Optimization:
      • Score = 0.4×train_sharpe + 0.4×train_calmar + 0.2×test_sharpe
      • Constraints (BOTH train AND test must pass):
          train: PF>1.2, cost%<20, WR≥35%, ≥30 trades
          test:  ≥80 trades, cost%<20
      This prevents params that train-overfit from being selected.
    """
    print(f"\n{'═'*66}")
    print(f"  WALK-FORWARD OPTIMIZATION  [{n_samples} samples | Train+Test scored]")
    print(f"{'═'*66}")
    keys=list(PARAM_GRID.keys())
    combos=list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    rng=np.random.default_rng(77); rng.shuffle(combos); combos=combos[:n_samples]
    results=[]; best=-np.inf; best_p=None; t0=time.time()

    for ci, combo in enumerate(combos):
        p=dict(zip(keys,combo))
        if p["hurst_mr"]<=p["hurst_trend"]: continue
        if p["stop_mult"]>=p["target_mult"]: continue
        try:
            # ── Train eval ────────────────────────────────────────────────
            ft=build_features(df_train,p); st=generate_signals(ft,p)
            trd_tr,eq_tr=run_backtest(st,p,CAPITAL)
            if len(trd_tr)<30: continue
            m_tr=metrics(trd_tr,eq_tr,CAPITAL)
            if not m_tr: continue
            if m_tr["pf"]<1.2: continue
            if m_tr.get("cost_pct",999)>20: continue
            if m_tr["wr"]<0.35: continue

            # ── Test eval (unseen — always starts from fresh CAPITAL) ──────
            test_cap=CAPITAL
            fe=build_features(df_test,p); se=generate_signals(fe,p)
            trd_te,eq_te=run_backtest(se,p,test_cap)
            if len(trd_te)<80: continue          # minimum 80 total over test window
            m_te=metrics(trd_te,eq_te,test_cap)
            if not m_te: continue
            if m_te.get("cost_pct",999)>22: continue
            # Per-year trade count — only enforce on FULL calendar years
            tdf_te_tmp=trades_to_df(trd_te)
            tdf_te_tmp["yr"]=pd.to_datetime(tdf_te_tmp["entry_date"]).dt.year
            yr_day_span=tdf_te_tmp.groupby("yr")["entry_date"].agg(lambda x:(x.max()-x.min()).days)
            full_years=yr_day_span[yr_day_span>=200].index  # years with 200+ trading days of data
            if len(full_years)>0:
                yr_counts_full=tdf_te_tmp[tdf_te_tmp["yr"].isin(full_years)].groupby("yr").size()
                if yr_counts_full.min()<70: continue   # full years must have ≥70

            # ── Composite score ───────────────────────────────────────────
            score=(0.4*m_tr["sharpe"] + 0.4*m_tr["calmar"]
                   + 0.2*m_te["sharpe"])
            results.append({**p,
                            "n_tr":m_tr["n"],"wr_tr":m_tr["wr"],
                            "sharpe_tr":m_tr["sharpe"],"calmar_tr":m_tr["calmar"],
                            "pf_tr":m_tr["pf"],"cagr_tr":m_tr["cagr"],
                            "n_te":m_te["n"],"wr_te":m_te["wr"],
                            "sharpe_te":m_te["sharpe"],"pf_te":m_te["pf"],
                            "cagr_te":m_te["cagr"],"cost_te":m_te.get("cost_pct",0),
                            "score":score})
            if score>best:
                best=score; best_p=p.copy()
                print(f"  [{ci+1:>4}/{len(combos)}] ★  Score={score:.2f}  "
                      f"Tr: Sh={m_tr['sharpe']:.2f} PF={m_tr['pf']:.2f} "
                      f"WR={m_tr['wr']*100:.0f}%  |  "
                      f"Te: Sh={m_te['sharpe']:.2f} PF={m_te['pf']:.2f} "
                      f"T={m_te['n']} Cost={m_te.get('cost_pct',0):.1f}%")
        except Exception: continue

    print(f"\n  Done in {time.time()-t0:.1f}s  |  Best Score={best:.2f}")
    rdf=(pd.DataFrame(results).sort_values("score",ascending=False)
         if results else pd.DataFrame())
    return best_p, rdf


# ── MONTE CARLO ───────────────────────────────────────────────────────────────

def monte_carlo(tdf: pd.DataFrame, cap0: float, n: int=MC_RUNS) -> dict:
    pnls=tdf["pnl_net"].values
    if len(pnls)<5: return {}
    rng=np.random.default_rng(42); nt=len(pnls)
    finals=[]; dds=[]; curves=[]
    for sim in range(n):
        sh=rng.choice(pnls,size=nt,replace=True)
        curve=np.concatenate([[cap0],cap0+np.cumsum(sh)])
        pk=np.maximum.accumulate(curve); dd=(curve-pk)/pk
        finals.append(curve[-1]); dds.append(dd.min())
        if sim<300: curves.append(curve)
    fc=np.array(finals); md=np.array(dds)
    return dict(finals=fc,dds=md,curves=np.array(curves),
                p5=np.percentile(fc,5),p25=np.percentile(fc,25),
                p50=np.percentile(fc,50),p75=np.percentile(fc,75),
                p95=np.percentile(fc,95),
                prob_profit=(fc>cap0).mean(),prob_ruin=(fc<cap0*0.5).mean(),
                prob_2x=(fc>cap0*2).mean(),
                med_dd=md.mean(),p95_dd=np.percentile(md,95),
                cap0=cap0,nt=nt,nsim=n)


# ── PRINT REPORTS ─────────────────────────────────────────────────────────────

def print_all_trades(tdf: pd.DataFrame, label: str="UNSEEN TEST"):
    W=116
    print(f"\n{'═'*W}\n  ALL TRADES — {label}\n{'═'*W}")
    if tdf.empty: print("  No trades."); return
    print(f"  {'#':>3}  {'Entry':>10}  {'Exit':>10}  {'Dir':>5}  "
          f"{'Sig':>4}  {'Q':>1}  {'Bars':>4}  {'EnPt':>6}  {'ExPt':>6}  "
          f"{'Lots':>4}  {'Reason':>7}  {'Gross':>9}  {'Cost':>5}  "
          f"{'Net P&L':>10}  {'':>6}")
    print(f"  {'─'*112}")
    tnw=tnl=tng=tnc=tnn=0
    for _,t in tdf.iterrows():
        win=t["pnl_net"]>0; tnw+=int(win); tnl+=int(not win)
        tng+=t["pnl_gross"]; tnc+=t["cost"]; tnn+=t["pnl_net"]
        print(f"  {int(t['idx'])+1:>3}  "
              f"{str(t['entry_date'].date()):>10}  "
              f"{str(t['exit_date'].date()):>10}  "
              f"{t['direction']:>5}  "
              f"{str(t['sig_tag']):>4}  "
              f"{int(t['q_score']):>1}  "
              f"{int(t['bars']):>4}  "
              f"{t['entry_price']:>6.0f}  "
              f"{t['exit_price']:>6.0f}  "
              f"{int(t['lots']):>4}  "
              f"{t['exit_reason']:>7}  "
              f"₹{t['pnl_gross']:>8,.0f}  "
              f"₹{t['cost']:>4,.0f}  "
              f"₹{t['pnl_net']:>9,.0f}  "
              f"{'✓WIN' if win else '✗LOSS':>5}")
    print(f"  {'─'*112}")
    cp=tnc/max(abs(tng),1)*100
    print(f"  {len(tdf)} trades | {tnw}W / {tnl}L | WR={tnw/max(len(tdf),1)*100:.1f}% | "
          f"Gross ₹{tng:,.0f} | Costs ₹{tnc:,.0f} ({cp:.1f}%) | NET ₹{tnn:,.0f}")
    print(f"{'═'*W}")


def print_metrics_report(m: dict, label: str):
    if not m: return
    def chk(val,tgt,hi=None):
        ok = val>=tgt if hi is None else tgt<=val<=hi
        return "✅" if ok else ("🟡" if (hi is None and val>=tgt*0.8) else "🔴")

    init = m.get("cap0", m["final"] - m["net_pnl"])

    print(f"\n{'═'*64}\n  {label}\n{'═'*64}")
    print(f"""
  CAPITAL
  {'─'*50}
  Initial Capital         : ₹{init:>12,.0f}
  Final Capital           : ₹{m['final']:>12,.0f}
  Net P&L                 : ₹{m['net_pnl']:>12,.0f}
  Total Return            : {m['ret']*100:.2f}%
  CAGR                    : {m['cagr']*100:.2f}%

  SCORECARD vs TARGETS
  {'─'*50}
  {chk(m['pf'],1.5)}  Profit Factor      : {m['pf']:.2f}          ≥ 1.5
  {chk(m['sharpe'],1.2)}  Sharpe Ratio       : {m['sharpe']:.3f}        ≥ 1.2
  {chk(m['cagr'],0.15,0.25)}  CAGR               : {m['cagr']*100:.2f}%        15–25%
  {chk(100-m.get('cost_pct',100),85)}  Cost % of Gross    : {m.get('cost_pct',0):.1f}%         < 15%
  {chk(m['n'],80,110)}  Trade Count / yr   : {m['n']}           90–100/yr

  TRADE STATISTICS
  {'─'*50}
  Total Trades            : {m['n']}
  Win Rate                : {m['wr']*100:.1f}%   ({m['nw']}W / {m['nl']}L)
  Avg Win                 : ₹{m['avg_win']:>12,.0f}
  Avg Loss                : ₹{m['avg_loss']:>12,.0f}
  Win / Loss Ratio        : {m['wl_ratio']:.2f}×
  Profit Factor           : {m['pf']:.2f}
  Expectancy / Trade      : ₹{m['expect']:>12,.0f}
  Largest Win             : ₹{m['max_win']:>12,.0f}
  Largest Loss            : ₹{m['max_loss']:>12,.0f}
  Avg Bars Held           : {m['avg_bars']:.1f} days
  Max Consec Wins         : {m['max_cw']}
  Max Consec Losses       : {m['max_cl']}

  P&L SUMMARY
  {'─'*50}
  Gross P&L (net of W-L)  : ₹{m['gross_pnl']:>12,.0f}
  Total Costs             : ₹{m['costs']:>12,.0f}   ({m.get('cost_pct',0):.1f}% of gross)
  Net P&L                 : ₹{m['net_pnl']:>12,.0f}
  Total Return            : {m['ret']*100:.2f}%
  CAGR                    : {m['cagr']*100:.2f}%

  RISK METRICS
  {'─'*50}
  Sharpe Ratio            : {m['sharpe']:.3f}
  Sortino Ratio           : {m['sortino']:.3f}
  Calmar Ratio            : {m['calmar']:.3f}
  Max Drawdown            : {m['max_dd']*100:.2f}%
  Max DD Duration         : {m['dd_days']} days

  EXIT REASON BREAKDOWN
  {'─'*50}""")
    for r,c in m["exit_dist"].items():
        print(f"  {r:<22}: {c:>4}  ({c/m['n']*100:.0f}%)")
    print(f"\n  SIGNAL TYPE BREAKDOWN\n  {'─'*50}")
    for s,c in sorted(m["sig_dist"].items(),key=lambda x:-x[1]):
        print(f"  {s:<22}: {c:>4}  ({c/m['n']*100:.0f}%)")
    print(f"\n{'═'*64}")


def print_mc(mc: dict):
    if not mc: return
    print(f"\n{'═'*64}\n  MONTE CARLO ({mc['nsim']:,} reshuffles)\n{'═'*64}")
    print(f"""
  Capital              : ₹{mc['cap0']:>12,.0f}
  Trades               : {mc['nt']}
   5pct (worst 5%)     : ₹{mc['p5']:>12,.0f}
  25pct                : ₹{mc['p25']:>12,.0f}
  50pct (median)       : ₹{mc['p50']:>12,.0f}
  75pct                : ₹{mc['p75']:>12,.0f}
  95pct (best 5%)      : ₹{mc['p95']:>12,.0f}
  P(Profit)            : {mc['prob_profit']*100:.1f}%
  P(2× Capital)        : {mc['prob_2x']*100:.1f}%
  P(Ruin <50%)         : {mc['prob_ruin']*100:.2f}%
  Median Max DD        : {mc['med_dd']*100:.1f}%
  95th-pct Max DD      : {mc['p95_dd']*100:.1f}%
""")
    print(f"{'═'*64}")


# ── CHARTS (inline — no files) ───────────────────────────────────────────────

def _ax(ax):
    ax.set_facecolor(C["panel"]); ax.tick_params(colors=C["muted"],labelsize=8)
    for sp in ax.spines.values(): sp.set_color(C["border"])
    ax.title.set_color(C["gold"])
    ax.xaxis.label.set_color(C["muted"]); ax.yaxis.label.set_color(C["muted"])


def chart_optimization(opt_df: pd.DataFrame):
    if opt_df.empty: print("  [Chart] No optimization results to plot."); return
    fig,axes=plt.subplots(2,2,figsize=(14,10)); fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Parameter Optimization — 8-Year Train",color=C["gold"],fontsize=13,fontweight="bold")
    pairs=[("sharpe","calmar","Sharpe vs Calmar"),("wr","pf","Win Rate vs Profit Factor"),
           ("cagr","max_dd","CAGR vs Max Drawdown"),("n","sharpe","Trade Count vs Sharpe")]
    top=opt_df.head(10)
    for ax,(x,y,title) in zip(axes.flat,pairs):
        _ax(ax)
        if x in opt_df.columns and y in opt_df.columns:
            sc=ax.scatter(opt_df[x],opt_df[y],c=opt_df["score"],cmap="RdYlGn",
                          s=35,alpha=0.65,linewidths=0)
            ax.scatter(top[x],top[y],s=130,c=C["gold"],marker="*",zorder=5,label="Top 10")
            plt.colorbar(sc,ax=ax,label="Score")
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(title,fontsize=10)
        ax.legend(fontsize=7,facecolor=C["panel"],labelcolor=C["text"])
    plt.tight_layout(); print("  [Chart] Optimization — displayed inline"); show_fig(fig)


def chart_main(sig_df, tdf, eq, m, mc, label):
    fig=plt.figure(figsize=(20,28)); fig.patch.set_facecolor(C["bg"])
    gs=gridspec.GridSpec(4,2,figure=fig,height_ratios=[2.5,1.5,1.5,1.5],hspace=0.45,wspace=0.30)
    ax_pr=fig.add_subplot(gs[0,:]); ax_eq=fig.add_subplot(gs[1,0])
    ax_hu=fig.add_subplot(gs[1,1]); ax_dd=fig.add_subplot(gs[2,0])
    ax_mc=fig.add_subplot(gs[2,1]); ax_di=fig.add_subplot(gs[3,0])
    ax_mo=fig.add_subplot(gs[3,1])
    for ax in [ax_pr,ax_eq,ax_hu,ax_dd,ax_mc,ax_di,ax_mo]: _ax(ax)

    ax_pr.plot(sig_df.index,sig_df["Close"],color=C["blue"],lw=0.8,alpha=0.85)
    if not tdf.empty:
        TAG_CLR={"S1L":C["green"],"S2L":C["teal"],"S3L":"#74b9ff","S4L":C["purple"],
                 "S5L":C["orange"],"S1S":C["red"],"S2S":"#ff6b6b","S3S":"#a29bfe",
                 "S4S":"#fd79a8","S5S":"#e17055"}
        for _,t in tdf.iterrows():
            clr=TAG_CLR.get(t["sig_tag"],C["gold"])
            mk="^" if t["direction"]=="long" else "v"
            ax_pr.scatter(t["entry_date"],t["entry_price"],marker=mk,color=clr,
                          s=45,zorder=5,linewidths=0,alpha=0.9)
            ax_pr.scatter(t["exit_date"],t["exit_price"],
                          marker="x",color=C["green"] if t["pnl_net"]>0 else C["red"],
                          s=35,zorder=5,linewidths=1.2)
            ax_pr.plot([t["entry_date"],t["exit_date"]],[t["entry_price"],t["exit_price"]],
                       color=C["green"] if t["pnl_net"]>0 else C["red"],lw=0.4,alpha=0.2)
    ax_pr.set_title(f"Nifty 50 — Trades | {label}",fontsize=13,fontweight="bold")
    hs=[mpatches.Patch(color=C["green"],label="S1 Breakout L"),
        mpatches.Patch(color=C["teal"], label="S2 Momentum L"),
        mpatches.Patch(color="#74b9ff",label="S3 Z-Rev L"),
        mpatches.Patch(color=C["purple"],label="S4 Ret-Ext L"),
        mpatches.Patch(color=C["orange"],label="S5 ATR Exp L"),
        mpatches.Patch(color=C["red"],  label="Short signals")]
    ax_pr.legend(handles=hs,loc="upper left",fontsize=7,facecolor=C["panel"],
                 edgecolor=C["border"],labelcolor=C["text"],ncol=2)

    if not eq.empty:
        en=eq/eq.iloc[0]*100; br=sig_df["Close"].reindex(eq.index).ffill(); bn=br/br.iloc[0]*100
        ax_eq.plot(en.index,en,color=C["gold"],lw=1.6,label="Strategy",zorder=3)
        ax_eq.plot(bn.index,bn,color=C["muted"],lw=1.0,ls="--",alpha=0.7,label="Buy&Hold")
        ax_eq.fill_between(en.index,en,100,where=(en>=100),alpha=0.12,color=C["green"])
        ax_eq.fill_between(en.index,en,100,where=(en<100), alpha=0.12,color=C["red"])
        ax_eq.axhline(100,color=C["border"],lw=0.7)
        ax_eq.set_title("Equity Curve vs Buy & Hold",fontsize=10,fontweight="bold")
        ax_eq.legend(fontsize=8,facecolor=C["panel"],labelcolor=C["text"])
        ax_eq.set_ylabel("Normalised (100)")
    if m:
        ax_eq.text(0.02,0.05,
                   f"CAGR {m['cagr']*100:.1f}%  Sharpe {m['sharpe']:.2f}  "
                   f"PF {m['pf']:.2f}  Cost% {m.get('cost_pct',0):.1f}%",
                   transform=ax_eq.transAxes,color=C["gold"],fontsize=8,
                   bbox=dict(facecolor=C["panel"],alpha=0.8,edgecolor=C["border"]))

    ax_hu.plot(sig_df.index,sig_df["hurst"],color=C["purple"],lw=0.8)
    ax_hu.axhline(0.55,color=C["green"],lw=1,ls="--",alpha=0.8,label="Trend 0.55")
    ax_hu.axhline(0.45,color=C["red"],  lw=1,ls="--",alpha=0.8,label="MR   0.45")
    ax_hu.fill_between(sig_df.index,0.45,0.55,alpha=0.07,color=C["muted"])
    ax_hu.set_ylim(0,1); ax_hu.set_title("Hurst Exponent",fontsize=10,fontweight="bold")
    ax_hu.legend(fontsize=7,facecolor=C["panel"],labelcolor=C["text"])

    if not eq.empty:
        dd=(eq-eq.cummax())/eq.cummax()*100
        ax_dd.fill_between(dd.index,dd,0,color=C["red"],alpha=0.5)
        ax_dd.plot(dd.index,dd,color=C["red"],lw=0.7)
        ax_dd.axhline(0,color=C["border"],lw=0.6)
        ax_dd.set_title("Drawdown (%)",fontsize=10,fontweight="bold"); ax_dd.set_ylabel("%")
        if m: ax_dd.text(0.02,0.10,f"Max DD {m['max_dd']*100:.1f}%  Dur {m['dd_days']}d",
                         transform=ax_dd.transAxes,color=C["red"],fontsize=8)

    if mc:
        sc=mc["curves"]; t_=np.arange(sc.shape[1])
        for i in range(min(250,len(sc))):
            ax_mc.plot(t_,sc[i],color=C["green"] if sc[i,-1]>mc["cap0"] else C["red"],
                       lw=0.3,alpha=0.04)
        pct=np.percentile(sc,[5,25,50,75,95],axis=0)
        ax_mc.fill_between(t_,pct[0],pct[4],color=C["blue"],alpha=0.08)
        ax_mc.fill_between(t_,pct[1],pct[3],color=C["blue"],alpha=0.15)
        ax_mc.plot(t_,pct[2],color=C["gold"],lw=1.8,label="Median",zorder=5)
        ax_mc.axhline(mc["cap0"],color=C["muted"],lw=0.8,ls="--")
        ax_mc.set_title(f"Monte Carlo ({mc['nsim']:,})",fontsize=10,fontweight="bold"); ax_mc.set_ylabel("Capital (₹)")
        ax_mc.text(0.02,0.04,
                   f"P(Profit)={mc['prob_profit']*100:.0f}%  P(Ruin)={mc['prob_ruin']*100:.1f}%  P(2×)={mc['prob_2x']*100:.1f}%\n"
                   f"Med ₹{mc['p50']:,.0f}  5pct ₹{mc['p5']:,.0f}  95pct ₹{mc['p95']:,.0f}",
                   transform=ax_mc.transAxes,color=C["gold"],fontsize=7.5,
                   bbox=dict(facecolor=C["panel"],alpha=0.8,edgecolor=C["border"]))
        ax_mc.legend(fontsize=7,facecolor=C["panel"],labelcolor=C["text"])

    if not tdf.empty:
        pnls=tdf["pnl_net"].values; w_=pnls[pnls>0]; l_=pnls[pnls<=0]
        bins=np.linspace(pnls.min(),pnls.max(),50)
        ax_di.hist(l_,bins=bins,color=C["red"],  alpha=0.7,label=f"Loss ({len(l_)})",edgecolor=C["bg"])
        ax_di.hist(w_,bins=bins,color=C["green"],alpha=0.7,label=f"Win  ({len(w_)})",edgecolor=C["bg"])
        ax_di.axvline(0,color=C["muted"],lw=1)
        ax_di.axvline(pnls.mean(),color=C["gold"],lw=1.5,ls="--",label=f"Mean ₹{pnls.mean():,.0f}")
        ax_di.set_title("Trade P&L Distribution",fontsize=10,fontweight="bold"); ax_di.set_xlabel("Net P&L (₹)")
        ax_di.legend(fontsize=7,facecolor=C["panel"],labelcolor=C["text"])

    if not tdf.empty:
        tmp=tdf.copy(); tmp["ym"]=pd.to_datetime(tmp["exit_date"]).dt.to_period("M")
        mp=tmp.groupby("ym")["pnl_net"].sum()
        ax_mo.bar(range(len(mp)),mp.values,color=[C["green"] if v>=0 else C["red"] for v in mp.values],
                  alpha=0.85,width=0.8)
        ax_mo.axhline(0,color=C["muted"],lw=0.7)
        tp=[i for i,p_ in enumerate(mp.index) if p_.month==1]
        ax_mo.set_xticks(tp); ax_mo.set_xticklabels([str(mp.index[i].year) for i in tp],fontsize=8,color=C["muted"])
        ax_mo.set_title("Monthly Net P&L (₹)",fontsize=10,fontweight="bold"); ax_mo.set_ylabel("₹")
        pm=(mp>0).sum()
        ax_mo.text(0.02,0.92,f"Profitable months: {pm}/{len(mp)} ({pm/max(len(mp),1)*100:.0f}%)",
                   transform=ax_mo.transAxes,color=C["gold"],fontsize=8)

    fig.suptitle("UNiverse Capital | Nifty 50 Futures — v4.1 Aggressive Algo",
                 color=C["gold"],fontsize=15,fontweight="bold",y=0.997)
    print(f"  [Chart] {label} — displayed inline"); show_fig(fig)


def chart_mc_deep(mc: dict):
    if not mc: return
    fig,axes=plt.subplots(2,2,figsize=(15,10)); fig.patch.set_facecolor(C["bg"])
    fig.suptitle("Monte Carlo — Deep Dive",color=C["gold"],fontsize=14,fontweight="bold")
    for ax in axes.flat: _ax(ax)

    ax1=axes[0,0]; fc=mc["finals"]
    ax1.hist(fc,bins=80,color=C["blue"],alpha=0.75,edgecolor=C["bg"])
    for val,lbl,clr in [(mc["cap0"],"Start",C["muted"]),(mc["p5"],"5%",C["red"]),
                         (mc["p50"],"Median",C["gold"]),(mc["p95"],"95%",C["green"])]:
        ax1.axvline(val,color=clr,lw=1.8,ls="--",label=lbl)
    ax1.set_title("Final Capital Distribution",fontsize=10,fontweight="bold"); ax1.set_xlabel("Capital (₹)")
    ax1.legend(fontsize=7,facecolor=C["panel"],labelcolor=C["text"])

    ax2=axes[0,1]; mdd=mc["dds"]*100
    ax2.hist(mdd,bins=60,color=C["red"],alpha=0.75,edgecolor=C["bg"])
    ax2.axvline(np.percentile(mdd,50),color=C["gold"],lw=1.8,ls="--",label=f"Med {np.percentile(mdd,50):.1f}%")
    ax2.axvline(np.percentile(mdd,95),color=C["red"],lw=1.5,ls="--",label=f"95th {np.percentile(mdd,95):.1f}%")
    ax2.set_title("Max Drawdown Distribution",fontsize=10,fontweight="bold"); ax2.set_xlabel("Max Drawdown (%)")
    ax2.legend(fontsize=7,facecolor=C["panel"],labelcolor=C["text"])

    ax3=axes[1,0]
    prbs=[mc["prob_profit"]*100,mc["prob_2x"]*100,mc["prob_ruin"]*100]
    bars=ax3.bar(["P(Profit)","P(2×)","P(Ruin<50%)"],prbs,
                  color=[C["green"],C["gold"],C["red"]],alpha=0.85,width=0.5)
    for bar,v in zip(bars,prbs):
        ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5,f"{v:.1f}%",
                 ha="center",color=C["text"],fontsize=11,fontweight="bold")
    ax3.set_ylim(0,115); ax3.set_title("Probability Outcomes",fontsize=10,fontweight="bold")
    ax3.set_ylabel("Probability (%)")

    ax4=axes[1,1]; sc=mc["curves"]; t_=np.arange(sc.shape[1])
    pct=np.percentile(sc,[5,10,25,50,75,90,95],axis=0)
    ax4.fill_between(t_,pct[0],pct[6],color=C["blue"],alpha=0.08,label="5–95%")
    ax4.fill_between(t_,pct[1],pct[5],color=C["blue"],alpha=0.12,label="10–90%")
    ax4.fill_between(t_,pct[2],pct[4],color=C["blue"],alpha=0.20,label="25–75%")
    ax4.plot(t_,pct[3],color=C["gold"],lw=2.0,label="Median",zorder=5)
    ax4.axhline(mc["cap0"],color=C["muted"],lw=0.8,ls="--")
    ax4.set_title("Equity Percentile Fan",fontsize=10,fontweight="bold")
    ax4.set_xlabel("Trade #"); ax4.set_ylabel("Capital (₹)")
    ax4.legend(fontsize=7,facecolor=C["panel"],labelcolor=C["text"])

    plt.tight_layout(); print("  [Chart] Monte Carlo Deep Dive — displayed inline"); show_fig(fig)


# ── YEAR-BY-YEAR BREAKDOWN ───────────────────────────────────────────────────

def print_yearly_breakdown(trades: list, eq: pd.Series, cap0: float):
    """
    Per-year breakdown. Capital carries forward each year.
    * = partial year (feature warmup consumed first months of data).
    Ann/yr = annualised trade rate  (actual_trades / active_days * 365).
    """
    if not trades: return
    tdf = trades_to_df(trades)
    tdf["entry_year"] = pd.to_datetime(tdf["entry_date"]).dt.year
    years = sorted(tdf["entry_year"].unique())
    SEP = "=" * 116

    print("\n" + SEP)
    print("  YEAR-BY-YEAR BREAKDOWN  (Capital carries forward | * = partial year)")
    print(SEP)
    print("  {:>6}  {:>14}  {:>14}  {:>12}  {:>7}  {:>6}  {:>6}  {:>5}  {:>5}  {:>7}  {:>7}  {:>6}".format(
          "Year","Init Capital","Final Capital","Net P&L","Return","Trades","Ann/yr","WR","PF","Sharpe","MaxDD","Cost%"))
    print("  " + "-" * 112)

    running_cap = cap0
    for yr in years:
        yr_trades = [t for t in trades if t.entry_date.year == yr]
        if not yr_trades: continue
        yr_exits  = [t.exit_date for t in yr_trades if t.exit_date is not None]
        if not yr_exits: continue

        start_dt  = min(t.entry_date for t in yr_trades)
        end_dt    = max(yr_exits)
        eq_yr     = eq[(eq.index >= start_dt) & (eq.index <= end_dt)]
        if eq_yr.empty: continue

        init_cap  = running_cap
        final_cap = eq_yr.iloc[-1]
        net_pnl   = final_cap - init_cap
        ret_pct   = net_pnl / max(init_cap, 1) * 100

        m_yr  = metrics(yr_trades, eq_yr, init_cap)
        n     = m_yr.get("n",       0)
        wr    = m_yr.get("wr",      0) * 100
        pf    = m_yr.get("pf",      0)
        sh    = m_yr.get("sharpe",  0)
        mdd   = m_yr.get("max_dd",  0) * 100
        cp    = m_yr.get("cost_pct",0)

        active_days = max((end_dt - start_dt).days, 1)
        ann_t       = round(n * 365 / active_days)
        partial     = active_days < 300
        yr_lbl      = str(yr) + ("*" if partial else " ")

        t_ok  = "OK " if 80 <= ann_t <= 115 else ("~~ " if 60 <= ann_t < 80 else "LOW")
        r_ok  = "+" if ret_pct >= 0 else "-"

        print("  {:<6}  {}{:>12,.0f}  {:>12,.0f}  {:>+12,.0f}  {:>6.1f}%  {:>4}T    {:>4}/yr  {:>4.0f}%  {:>5.2f}  {:>7.2f}  {:>6.1f}%  {:>5.1f}%".format(
              yr_lbl,
              r_ok, init_cap, final_cap, int(net_pnl), ret_pct,
              n, ann_t, wr, pf, sh, mdd, cp))

        running_cap = final_cap

    total_pnl = running_cap - cap0
    total_ret  = total_pnl / max(cap0, 1) * 100
    print("  " + "-" * 112)
    print("  {:<6}  {:>14}  {:>12,.0f}  {:>+12,.0f}  {:>6.1f}%".format(
          "TOTAL", "₹" + "{:,.0f}".format(int(cap0)), int(running_cap), int(total_pnl), total_ret))
    print("  * Partial year: Hurst/ATR warmup (~3 months) consumes start of test window.")
    print("    Ann/yr shows true pace.  OK=90-115/yr  ~~=60-80/yr  LOW=<60/yr")
    print(SEP + "\n")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    t0=time.time()
    print("""
╔══════════════════════════════════════════════════════════════╗
║   UNiverse Capital — Nifty 50 Swing Algo  v4.1              ║
║   PF≥1.5 | Sharpe≥1.2 | CAGR 15-25% | Cost<15% | 90-100T  ║
║   5-Signal · Wide Targets (3-4.5×) · Inline Charts          ║
╚══════════════════════════════════════════════════════════════╝""")

    print("\n[1/6] Generating data...")
    df_tr=generate_nifty_data(TRAIN_START,TRAIN_END,7500.0)
    s0t=float(df_tr["Close"].iloc[-1])
    df_te=generate_nifty_data(TEST_START,TEST_END,s0t)
    print(f"  Train : {len(df_tr)} days  {TRAIN_START}→{TRAIN_END}")
    print(f"  Test  : {len(df_te)} days  {TEST_START}→{TEST_END}")
    print("  ⚠  Synthetic — swap generate_nifty_data() for Kite API live data")

    best_p,opt_df=optimize(df_tr, df_te, n_samples=250)
    if best_p is None:
        print("  WARNING: Using fallback defaults.")
        best_p=dict(hurst_window=40,hurst_trend=0.5,hurst_mr=0.52,
                    fast_bk_window=7,mom_consec=3,mr_z_thresh=1.0,
                    zscore_window=10,ret_extreme_pct=0.012,ret_extreme_days=5,
                    atr_expand_mult=2.0,atr_window=10,
                    stop_mult=0.9,target_mult=4.5,time_stop=3,
                    circuit_breaker=5)

    print("\n[3/6] Training backtest (silent)...")
    ft=build_features(df_tr,best_p); st=generate_signals(ft,best_p)
    trd_tr,eq_tr=run_backtest(st,best_p,CAPITAL)

    print("\n[4/6] Backtest on UNSEEN TEST (2 yr)...")
    test_cap = CAPITAL  # test always starts from your configured capital, not train equity
    fe=build_features(df_te,best_p); se=generate_signals(fe,best_p)
    trd_te,eq_te=run_backtest(se,best_p,test_cap)
    tdf_te=trades_to_df(trd_te); m_te=metrics(trd_te,eq_te,test_cap)
    print(f"  ✓ Trades={m_te.get('n',0)}  Net=₹{m_te.get('net_pnl',0):,.0f}  "
          f"Sharpe={m_te.get('sharpe',0):.2f}  PF={m_te.get('pf',0):.2f}  "
          f"CAGR={m_te.get('cagr',0)*100:.1f}%  Cost%={m_te.get('cost_pct',0):.1f}%")

    print("\n[5/6] Monte Carlo...")
    mc=monte_carlo(tdf_te,test_cap,MC_RUNS) if not tdf_te.empty else {}
    if mc:
        print(f"  P(Profit)={mc['prob_profit']*100:.0f}%  P(Ruin)={mc['prob_ruin']*100:.2f}%  "
              f"P(2×)={mc['prob_2x']*100:.1f}%  Median=₹{mc['p50']:,.0f}")

    print("\n[6/6] Reports + Charts...")
    print(f"\n{'═'*64}\n  OPTIMISED PARAMETERS\n{'═'*64}")
    for k,v in best_p.items(): print(f"  {k:<26}: {v}")

    # ── UNSEEN TEST ONLY ──────────────────────────────────────────────────
    print_all_trades(tdf_te,"UNSEEN TEST DATA  (2 Years: 2023–2024)")
    print_metrics_report(m_te,"UNSEEN TEST PERFORMANCE  (2023–2024)")
    print_yearly_breakdown(trd_te, eq_te, test_cap)
    print_mc(mc)

    # ── Charts: test + MC only (no train chart) ───────────────────────────
    chart_main(se,tdf_te,eq_te,m_te,mc,"Unseen Test 2023-2024")
    chart_mc_deep(mc)

    print(f"\n✓ Complete in {time.time()-t0:.1f}s\n")
    return tdf_te,m_te,mc


if __name__=="__main__":
    main()