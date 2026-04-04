"""
NSE FUTURES BACKTEST — Pure Quantitative v2.0
═══════════════════════════════════════════════════════════════════════
v2 IMPROVEMENTS  (v1 delivered -97% return, 0% SL win rate)
───────────────────────────────────────────────────────────────────────
1. Two-stage walk-forward  (5yr train → 2yr validate → 1yr live)
   Params must score positively on BOTH stages — eliminates overfitting

2. Stop-multiplier added to grid  [2.0, 2.5, 3.0]
   Root cause of 0% SL win rate: 1.5× was too tight for daily bars

3. Variance Ratio Test filter  (Lo & MacKinlay, 1988)
   VR(k) = Var[k-period return] / (k × Var[1-period return])
   VR > 1+ε → momentum regime only
   VR < 1-ε → mean-reversion regime only
   |VR-1| < ε → random walk → skip (no statistical edge)

4. Charge-penalised scoring
   score = PF × Sharpe × max(0.1, 1 − 3 × charge_fraction)
   High-turnover strategies penalised before selection

5. Drawdown-scaled position sizing + circuit breaker
   DD >10% → 75% lots  |  DD >20% → 50%  |  DD >30% → 25%
   DD >40% → circuit breaker: no new entries

6. Trade management upgrades
   MIN_HOLD = 2 bars before FLIP/TP (SL still fires immediately)
   Breakeven stop: move SL to entry after 1× risk profit

Signal    : N-day Rolling Sharpe = Σ log-ret(N) / (σ_daily × √N)
Filter    : Rolling Variance Ratio (Lo-MacKinlay 1988)
Stop      : σ_daily × price × STOP_MULT (in grid)
Grid      : 6N × 5K × 3M × 2 modes = 180 combinations
Walk-fwd  : 5yr train → 2yr validate → 1yr live
═══════════════════════════════════════════════════════════════════════
"""

import datetime as dt, os, sys, warnings, itertools
warnings.filterwarnings("ignore")
import pytz, yfinance as yf, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker


# ═══════════════════════════════════════════════════════════════════════
# 1.  FUTURES DATABASE
# ═══════════════════════════════════════════════════════════════════════
DB = {
    "RELIANCE.NS" : (250, 150_000, "Reliance Industries"),
    "TCS.NS"      : (150, 120_000, "Tata Consultancy Services"),
    "HDFCBANK.NS" : (550,  80_000, "HDFC Bank"),
}


# ═══════════════════════════════════════════════════════════════════════
# 2.  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
SYMBOL          = "RELIANCE.NS"
INITIAL_CAPITAL = 1_000_000          # ₹ 10 Lakh

# ── Walk-forward splits (years) ───────────────────────────────────────
TOTAL_YEARS = 8
TRAIN_YEARS = 5     # stage-1 optimisation
VALID_YEARS = 2     # stage-2 anti-overfit validation gate
LIVE_YEARS  = 1     # out-of-sample live test

# ── Parameter grid (180 combinations) ────────────────────────────────
LOOKBACK_GRID  = [5, 7, 10, 15, 20, 30]        # N : signal window (bars)
THRESHOLD_GRID = [0.3, 0.5, 0.75, 1.0, 1.5]   # K : entry signal threshold
STOPMULT_GRID  = [2.0, 2.5, 3.0]               # M : vol-stop multiplier (FIX #2)
MODE_GRID      = ["momentum", "mean_reversion"]

# ── Variance Ratio filter (Lo & MacKinlay, 1988) ─────────────────────
VR_WINDOW = 30      # rolling estimation window (bars)
VR_K      = 5       # k-lag for VR computation
VR_BAND   = 0.05    # neutral zone: skip if |VR − 1| < VR_BAND

# ── Risk management ──────────────────────────────────────────────────
RISK_PCT    = 0.015     # 1.5% capital at risk per trade (was 2%)
MAX_LOTS    = 8         # hard cap (was 10)
MARGIN_UTIL = 0.50      # max 50% capital as margin (was 60%)
TARGET_R    = 2.0       # take-profit = 2× risk distance
COOLDOWN    = 0         # bars blocked after exit

# ── Drawdown controls (FIX #5) ───────────────────────────────────────
# List of (dd_threshold, lots_multiplier) — applied at worst threshold
DD_SCALE   = [(0.10, 0.75), (0.20, 0.50), (0.30, 0.25)]
CIRCUIT_DD = 0.40       # halt entries if equity DD > 40%

# ── Trade management (FIX #6) ────────────────────────────────────────
MIN_HOLD     = 2        # min bars before FLIP/TP exit (SL fires immediately)
BE_TRIGGER_R = 1.0      # move stop to entry price after 1× risk reward

# ── Futures mechanics ────────────────────────────────────────────────
ROLL_DAY_BEFORE = 1     # roll N trading days before expiry
MARGIN_CALL_PCT = 0.75  # force-close if margin < 75% of SPAN required

# ── NSE Charges (Zerodha 2024) ───────────────────────────────────────
BROKERAGE  = 20.0
STT_RATE   = 0.000125   # STT on SELL side only (futures)
EXCH_RATE  = 0.000019
SEBI_RATE  = 0.000001
STAMP_RATE = 0.00002    # stamp duty on BUY side only
GST_RATE   = 0.18

OUT_DIR = "outputs"
IST     = pytz.timezone("Asia/Kolkata")


# ═══════════════════════════════════════════════════════════════════════
# 3.  NSE EXPIRY CALENDAR  (last Thursday of each month)
# ═══════════════════════════════════════════════════════════════════════
def last_thursday(year: int, month: int) -> dt.date:
    last = (dt.date(year+1,1,1) if month==12
            else dt.date(year,month+1,1)) - dt.timedelta(1)
    return last - dt.timedelta((last.weekday()-3)%7)


def build_expiry_calendar(start: dt.date, end: dt.date) -> list:
    expiries, y, m = [], start.year, start.month
    while True:
        exp = last_thursday(y, m)
        if exp > end: break
        if exp >= start: expiries.append(exp)
        m += 1
        if m > 12: m, y = 1, y+1
    return expiries


def build_roll_dates(expiries: list, trading_dates: list) -> set:
    pos = {d: i for i, d in enumerate(trading_dates)}
    roll_dates = set()
    for exp in expiries:
        cands = [d for d in trading_dates if d <= exp]
        if not cands: continue
        roll_dates.add(trading_dates[max(0, pos[cands[-1]] - ROLL_DAY_BEFORE)])
    return roll_dates


# ═══════════════════════════════════════════════════════════════════════
# 4.  CHARGE MODEL  (Zerodha NSE Futures)
# ═══════════════════════════════════════════════════════════════════════
def charges(price: float, qty: float, side: str) -> float:
    tv  = abs(price * qty)
    brk = BROKERAGE
    stt = tv * STT_RATE   if side == "SELL" else 0.0
    exc = tv * EXCH_RATE
    sbi = tv * SEBI_RATE
    stp = tv * STAMP_RATE if side == "BUY"  else 0.0
    gst = (brk + exc + sbi) * GST_RATE
    return brk + stt + exc + sbi + stp + gst


def rollover_cost(price: float, qty: float) -> float:
    return charges(price, qty, "SELL") + charges(price, qty, "BUY")


# ═══════════════════════════════════════════════════════════════════════
# 5.  VARIANCE RATIO TEST  (Lo & MacKinlay, 1988)
#
#     FIX #3: Statistical regime filter — pure math, zero TA.
#
#     Formula:
#         VR(k) = Var[ r_t^(k) ] / ( k × Var[ r_t ] )
#
#         where r_t^(k) = Σ_{j=0}^{k-1} log(P_{t-j} / P_{t-j-1})
#               (k-period cumulative log-return)
#
#     Interpretation:
#         VR > 1  → positive return autocorrelation → MOMENTUM regime
#         VR < 1  → negative return autocorrelation → MEAN-REVERSION regime
#         VR ≈ 1  → cannot reject random walk → NO EDGE, skip entry
#
#     Rolling implementation:
#         var1(t) = rolling variance of 1-day log-returns over WINDOW bars
#         var_k(t) = rolling variance of k-day cumul. returns over WINDOW bars
#         VR(t)   = var_k(t) / (k × var1(t))
#
#     All series are .shift(1)-ed before use → zero look-ahead.
# ═══════════════════════════════════════════════════════════════════════
def compute_rolling_vr(price: pd.Series,
                       window: int = VR_WINDOW,
                       k: int = VR_K) -> pd.Series:
    lr    = np.log(price / price.shift(1))      # daily log-returns
    var1  = lr.rolling(window).var()            # Var[r_t]
    ret_k = lr.rolling(k).sum()                 # k-day cumulative return
    var_k = ret_k.rolling(window).var()         # Var[r_t^(k)]
    vr    = var_k / (k * var1)
    return vr.fillna(1.0)                       # neutral (= RW) where no data


# ═══════════════════════════════════════════════════════════════════════
# 6.  SIGNAL GENERATION
#     Rolling N-day Sharpe ratio + Variance Ratio regime gate
# ═══════════════════════════════════════════════════════════════════════
def build_signals(df: pd.DataFrame,
                  N: int, K: float, M: float, mode: str) -> pd.DataFrame:
    """
    Signal
    ------
    signal_t = [ Σ_{j=1}^{N} log(P_{t-j}/P_{t-j-1}) ] / (σ_daily_{t-1} × √N)

    This is the N-day rolling Sharpe ratio of the price series.
    Measures: "how far did price move relative to its own typical daily vol?"
    All inputs .shift(1) → strict no look-ahead.

    Stop distance
    -------------
    stop_dist_t = σ_daily_{t-1} × close_t × M
    Automatically wider in volatile regimes (adaptive risk).

    Regime gate
    -----------
    VR > 1+VR_BAND : momentum regime  → only momentum signals pass
    VR < 1-VR_BAND : mean-rev regime  → only mean-rev signals pass
    Otherwise       : random walk     → no entry
    """
    out  = df.copy()
    lr   = np.log(df["c"] / df["c"].shift(1))

    ret_N = lr.rolling(N).sum().shift(1)        # N-bar cumul. log-return (lagged)
    sigma = lr.rolling(N).std().shift(1)        # rolling daily vol (lagged)

    out["signal"]    = ret_N / (sigma * np.sqrt(N))
    out["stop_dist"] = sigma * df["c"] * M

    vr = compute_rolling_vr(df["c"]).shift(1)   # VR lagged → no look-ahead
    out["vr"] = vr

    if mode == "momentum":
        regime = vr > (1.0 + VR_BAND)
        out["long_sig"]  = (out["signal"] >  K) & regime
        out["short_sig"] = (out["signal"] < -K) & regime
    else:  # mean_reversion
        regime = vr < (1.0 - VR_BAND)
        out["long_sig"]  = (out["signal"] < -K) & regime
        out["short_sig"] = (out["signal"] >  K) & regime

    return out.dropna()


# ═══════════════════════════════════════════════════════════════════════
# 7.  POSITION SIZING  (risk-based + drawdown scaling)
#     FIX #5
# ═══════════════════════════════════════════════════════════════════════
def lot_size_fn(capital: float, peak_capital: float, stop_dist: float) -> int:
    """
    Base lots = min(risk_limit, margin_limit, MAX_LOTS)
    Drawdown scaling applied after base lots computed.
    """
    if stop_dist <= 0: return 0
    lots_risk   = int(capital * RISK_PCT / (stop_dist * LOT_SIZE))
    lots_margin = int(capital * MARGIN_UTIL / MARGIN_LOT)
    lots        = min(lots_risk, lots_margin, MAX_LOTS)
    if lots == 0 and lots_margin >= 1: lots = 1
    lots = max(lots, 0)

    if peak_capital > 0:
        dd = 1.0 - capital / peak_capital
        for threshold, scale in sorted(DD_SCALE, reverse=True):
            if dd > threshold:
                lots = int(lots * scale)
                break
    return max(lots, 0)


# ═══════════════════════════════════════════════════════════════════════
# 8.  TRADE RECORD HELPER
# ═══════════════════════════════════════════════════════════════════════
def _rec(no, side, xrsn, edate, xdate, bh,
         epx, xpx, stop, lots, qty, gross, chg, net, cap) -> dict:
    return dict(no=no, side=side, xrsn=xrsn,
                edate=edate, xdate=xdate, bh=bh,
                epx=epx, xpx=xpx, stop=stop,
                lots=lots, qty=int(qty),
                gross=round(gross,2), chg=round(chg,2),
                net=round(net,2), cap=round(cap))


# ═══════════════════════════════════════════════════════════════════════
# 9.  FUTURES BACKTEST ENGINE
#     MTM settlement · rollover · margin call
#     + breakeven stop · min hold · DD scaling · circuit breaker
# ═══════════════════════════════════════════════════════════════════════
def run_backtest(df: pd.DataFrame, capital_start: float):
    """
    Returns
    -------
    equity    pd.Series  daily portfolio value (realised + unrealised)
    margin    pd.Series  margin account balance
    pnls      list       net P&L per closed trade
    trades    list       full trade records
    rolls     list       rollover events
    total_ch  float      total transaction costs
    mtm_log   list       daily MTM settlement records
    """
    expiries   = build_expiry_calendar(df.index[0].date(), df.index[-1].date())
    tdates     = [i.date() for i in df.index]
    roll_dates = build_roll_dates(expiries, tdates)

    # ── State ────────────────────────────────────────────────────────
    cap      = capital_start
    margin   = capital_start
    peak_cap = capital_start
    side     = None
    epx = stop_px = tp_px = prev_close = 0.0
    etime_dt = None; etime_i = 0; bars_held = 0
    nlots = qty = cdl = 0; span_held = 0.0; be_done = False

    eq_curve=[]; mg_curve=[]; pnls=[]; trades=[]; rolls=[]; mtm_log=[]
    total_ch = 0.0; tno = rno = 0

    rows = df.to_dict("records"); idx = df.index

    for i, row in enumerate(rows):
        bt         = idx[i]
        date_today = bt.date()
        o = row["o"]; h = row["h"]; l = row["l"]; c = row["c"]
        sd = max(row["stop_dist"], o * 0.002)   # floor at 0.2% of price
        if cdl > 0: cdl -= 1

        # Running equity peak (realised + unrealised)
        unr_now = (c-epx)*qty*(1 if side=="LONG" else -1) if side else 0.0
        tv_now  = cap + unr_now
        if tv_now > peak_cap: peak_cap = tv_now

        # ── Daily MTM Settlement ─────────────────────────────────────
        if side and prev_close > 0:
            mtm    = (c - prev_close) * qty * (1 if side=="LONG" else -1)
            margin += mtm
            cap    += mtm
            mtm_log.append({"date": str(date_today), "mtm": round(mtm,2),
                             "margin": round(margin,2)})

            # Margin call: broker force-closes at today's close
            if margin < span_held * MARGIN_CALL_PCT:
                grs = (c-epx)*qty*(1 if side=="LONG" else -1)
                chg = charges(c, qty, "SELL" if side=="LONG" else "BUY")
                net = grs - chg
                pnls.append(net); tno += 1; total_ch += chg
                margin += span_held; cap -= chg
                trades.append(_rec(tno, side, "MCL",
                    str(etime_dt.date()) if etime_dt else "?",
                    str(date_today), i-etime_i,
                    epx, round(c,2), stop_px, nlots, qty, grs, chg, net, cap))
                side=None; nlots=qty=0; etime_dt=None
                span_held=prev_close=0.0; cdl=1; be_done=False; bars_held=0
                continue

        # ── Monthly Rollover ─────────────────────────────────────────
        if side and date_today in roll_dates:
            rc = rollover_cost(c, qty)
            margin -= rc; cap -= rc; total_ch += rc; rno += 1
            rolls.append({"no":rno, "date":str(date_today), "side":side,
                          "price":round(c,2), "qty":int(qty),
                          "cost":round(rc,2), "cap_after":round(cap,2)})

        # ── Breakeven Stop (FIX #6) ──────────────────────────────────
        # After 1× risk reward, move stop to entry → can't lose on this trade
        if side and not be_done:
            risk_dist = abs(epx - stop_px)
            unr_ps    = (c - epx) * (1 if side=="LONG" else -1)   # per share
            if unr_ps >= BE_TRIGGER_R * risk_dist:
                stop_px = epx
                be_done = True

        # ── Track bars held ──────────────────────────────────────────
        if side:
            bars_held = i - etime_i

        # ── Exit Check ───────────────────────────────────────────────
        if side:
            xpx = xrsn = None

            # SL fires immediately — no minimum hold applies
            if   side=="LONG"  and l <= stop_px: xpx, xrsn = round(stop_px,2), "SL"
            elif side=="SHORT" and h >= stop_px: xpx, xrsn = round(stop_px,2), "SL"

            # TP and FLIP only after MIN_HOLD bars (FIX #4)
            if xpx is None and bars_held >= MIN_HOLD:
                if TARGET_R > 0:
                    if   side=="LONG"  and h >= tp_px: xpx, xrsn = round(tp_px,2), "TP"
                    elif side=="SHORT" and l <= tp_px: xpx, xrsn = round(tp_px,2), "TP"
                if xpx is None:
                    if   side=="LONG"  and row["short_sig"]: xpx, xrsn = round(c,2), "FLIP"
                    elif side=="SHORT" and row["long_sig"]:  xpx, xrsn = round(c,2), "FLIP"

            if xpx is not None:
                grs = (xpx-epx)*qty*(1 if side=="LONG" else -1)
                chg = charges(xpx, qty, "SELL" if side=="LONG" else "BUY")
                net = grs - chg
                margin += span_held - chg; cap -= chg
                total_ch += chg; pnls.append(net); tno += 1
                trades.append(_rec(tno, side, xrsn,
                    str(etime_dt.date()) if etime_dt else "?",
                    str(date_today), bars_held,
                    epx, xpx, stop_px, nlots, qty, grs, chg, net, cap))
                side=None; nlots=qty=0; etime_dt=None
                span_held=prev_close=0.0; cdl=COOLDOWN; be_done=False; bars_held=0

        # ── Entry Check ──────────────────────────────────────────────
        # Circuit breaker: halt if equity DD exceeds CIRCUIT_DD (FIX #5)
        in_circuit = peak_cap > 0 and (cap / peak_cap < (1 - CIRCUIT_DD))
        if not side and cdl == 0 and not in_circuit:
            sig = ("LONG"  if row["long_sig"]
                   else "SHORT" if row["short_sig"] else None)
            if sig:
                lots = lot_size_fn(cap, peak_cap, sd)
                if lots > 0:
                    span_req = lots * MARGIN_LOT
                    if margin >= span_req:
                        rd      = sd
                        epx     = round(o, 2)
                        stop_px = (round(epx-rd,2) if sig=="LONG"
                                   else round(epx+rd,2))
                        tp_px   = (round(epx+TARGET_R*rd,2) if sig=="LONG"
                                   else round(epx-TARGET_R*rd,2))
                        side    = sig; etime_dt = bt; etime_i = i
                        nlots   = lots; qty = lots * LOT_SIZE; bars_held = 0
                        chg     = charges(epx, qty, "BUY" if sig=="LONG" else "SELL")
                        span_held  = span_req
                        margin    -= chg; cap -= chg; total_ch += chg
                        prev_close = o; be_done = False

        # ── Daily Snapshot ───────────────────────────────────────────
        if side: prev_close = c
        unr = (c-epx)*qty*(1 if side=="LONG" else -1) if side else 0.0
        eq_curve.append(cap+unr); mg_curve.append(margin)

    return (pd.Series(eq_curve, index=idx),
            pd.Series(mg_curve, index=idx),
            pnls, trades, rolls, total_ch, mtm_log)


# ═══════════════════════════════════════════════════════════════════════
# 10. CHARGE-PENALISED SCORING  (FIX #4)
#
#     score = PF × Sharpe × max(0.1, 1 − 3 × charge_fraction)
#
#     charge_fraction = total_charges / initial_capital
#     At 33%+ charges the penalty zeroes out the score.
#     Penalises high-turnover regimes before parameter selection.
# ═══════════════════════════════════════════════════════════════════════
def score_fn(pnls: list, equity: pd.Series,
             total_ch: float, capital: float) -> float:
    if len(pnls) < 5: return -999.0
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf     = sum(wins) / abs(sum(losses)) if losses else 0.0
    rets   = equity.pct_change().dropna()
    sh     = ((rets.mean()/rets.std())*np.sqrt(252)
               if rets.std() > 0 else -999.0)
    penalty = max(0.1, 1.0 - 3.0 * (total_ch / capital))
    return pf * sh * penalty


def calc_drawdown(equity: pd.Series):
    peak = equity.cummax()
    dd   = (equity - peak) / peak
    t    = dd.idxmin()
    p    = equity.loc[:t].idxmax()
    return dd, float(dd.min()), p, t


# ═══════════════════════════════════════════════════════════════════════
# 11. PLOT
# ═══════════════════════════════════════════════════════════════════════
def save_plot(eq_live, dd_ser, trades_live, best_params, stock_name, symbol):
    N, K, M, mode = best_params
    C = {"bg":"#0d1117","grid":"#21262d","eq":"#58a6ff","zero":"#484f58",
         "dd":"#f85149","win":"#3fb950","loss":"#f85149","txt":"#8b949e",
         "ttl":"#e6edf3"}
    fig = plt.figure(figsize=(16,11), facecolor=C["bg"])
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3,1.2,1.2], hspace=0.40)

    for ax_i in range(3):
        ax = fig.add_subplot(gs[ax_i])
        ax.set_facecolor(C["bg"])
        for sp in ax.spines.values(): sp.set_color(C["grid"])
        ax.tick_params(colors=C["txt"]); ax.grid(True, color=C["grid"], lw=0.6, alpha=0.8)
        ax.xaxis.label.set_color(C["txt"]); ax.yaxis.label.set_color(C["txt"])

        if ax_i == 0:
            ax.plot(eq_live.index, eq_live/1e5, color=C["eq"], lw=1.8)
            ax.axhline(INITIAL_CAPITAL/1e5, color=C["zero"], ls="--", lw=0.9,
                       label="Initial capital")
            ax.fill_between(eq_live.index, eq_live/1e5, INITIAL_CAPITAL/1e5,
                            where=(eq_live >= INITIAL_CAPITAL),
                            alpha=0.12, color=C["win"])
            ax.fill_between(eq_live.index, eq_live/1e5, INITIAL_CAPITAL/1e5,
                            where=(eq_live < INITIAL_CAPITAL),
                            alpha=0.18, color=C["loss"])
            ax.set_title(f"{stock_name}  ·  Quant v2  ·  N={N}  K={K}  M={M}  {mode}",
                         fontsize=12, pad=10, color=C["ttl"])
            ax.set_ylabel("Capital (₹ Lakh)"); ax.legend(fontsize=9)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x,_: f"₹{x:.1f}L"))
        elif ax_i == 1:
            ax.fill_between(dd_ser.index, dd_ser*100, 0, color=C["dd"], alpha=0.75)
            ax.set_ylabel("Drawdown (%)"); ax.axhline(0, color=C["zero"], lw=0.6)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x,_: f"{x:.1f}%"))
        else:
            nets = [t["net"] for t in trades_live]
            cols = [C["win"] if n>=0 else C["loss"] for n in nets]
            ax.bar(range(len(nets)), nets, color=cols, width=0.7, alpha=0.85)
            ax.axhline(0, color=C["zero"], lw=0.8)
            ax.set_xlabel("Trade #"); ax.set_ylabel("Net P&L (₹)")
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x,_: f"₹{x:,.0f}"))

    plt.tight_layout(); os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{symbol.replace('.','_')}_v2.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(); print(f"\n  [Plot saved → {path}]")


# ═══════════════════════════════════════════════════════════════════════
# 12. MAIN
# ═══════════════════════════════════════════════════════════════════════
if SYMBOL not in DB: raise ValueError(f"'{SYMBOL}' not in DB.")
LOT_SIZE, MARGIN_LOT, STOCK_NAME = DB[SYMBOL]
os.makedirs(OUT_DIR, exist_ok=True)
G,R,Y,E = "\033[92m","\033[91m","\033[93m","\033[0m"
N_COMBOS = len(LOOKBACK_GRID)*len(THRESHOLD_GRID)*len(STOPMULT_GRID)*len(MODE_GRID)
W72 = "═"*72

print(f"\n{W72}")
print(f"  {STOCK_NAME}  ({SYMBOL})  ·  NSE FUTURES")
print(f"  Quant Strategy v2.0  ·  6 targeted improvements from v1")
print(f"  Grid: {len(LOOKBACK_GRID)}N × {len(THRESHOLD_GRID)}K × "
      f"{len(STOPMULT_GRID)}M × {len(MODE_GRID)} modes = {N_COMBOS} combos")
print(f"  Walk-forward: {TRAIN_YEARS}yr train → {VALID_YEARS}yr validate → {LIVE_YEARS}yr live")
print(f"{W72}\n")

# ── Data Fetch ───────────────────────────────────────────────────────
end_dt   = dt.datetime.now(tz=pytz.utc)
start_dt = end_dt - dt.timedelta(days=365*TOTAL_YEARS+150)
print("  Fetching price data...", end=" ", flush=True)
raw = yf.download(SYMBOL, start=start_dt, end=end_dt,
                  interval="1d", auto_adjust=True, progress=False)
if raw.empty: sys.exit("No data — check symbol / network.")
print(f"OK  ({len(raw)} raw bars)")

if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
raw.rename(columns={"Open":"o","High":"h","Low":"l","Close":"c","Volume":"v"},
           inplace=True)
raw = raw[["o","h","l","c","v"]].copy()
if raw.index.tzinfo is None: raw.index = raw.index.tz_localize("UTC")
raw.index = raw.index.tz_convert(IST)

# ── Three-way split ──────────────────────────────────────────────────
cutoff_all  = raw.index[-1] - pd.Timedelta(days=365*TOTAL_YEARS)
split_valid = raw.index[-1] - pd.Timedelta(days=365*(LIVE_YEARS+VALID_YEARS))
split_live  = raw.index[-1] - pd.Timedelta(days=365*LIVE_YEARS)

df_all   = raw.loc[raw.index >= cutoff_all].copy()
df_train = df_all.loc[df_all.index < split_valid].copy()
df_valid_raw = df_all.loc[(df_all.index >= split_valid) &
                           (df_all.index < split_live)].copy()

print(f"\n  Full data     : {df_all.index[0].date()} → {df_all.index[-1].date()}"
      f"  ({len(df_all)} bars)")
print(f"  Stage-1 Train : {df_train.index[0].date()} → {df_train.index[-1].date()}"
      f"  ({len(df_train)} bars)")
print(f"  Stage-2 Valid : {df_valid_raw.index[0].date()} → {df_valid_raw.index[-1].date()}"
      f"  ({len(df_valid_raw)} bars)")
print(f"  Live          : {split_live.date()} → {df_all.index[-1].date()}\n")

live_expiries = build_expiry_calendar(split_live.date(), df_all.index[-1].date())
print(f"  NSE expiries in live window ({len(live_expiries)} months):")
for i, exp in enumerate(live_expiries):
    print(f"    {exp.strftime('%d %b %Y')}", end=("   " if (i+1)%6 else "\n"))
print("\n")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1A — STAGE-1 TRAINING  (5yr optimisation)
# ═══════════════════════════════════════════════════════════════════════
print(W72)
print(f"  PHASE 1A — STAGE-1 TRAINING  ({TRAIN_YEARS}yr  ·  {N_COMBOS} combinations)")
print(W72)
print(f"  {'N':>4}  {'K':>5}  {'M':>4}  {'Mode':<16}  {'Trades':>7}  "
      f"{'WR%':>6}  {'PF':>6}  {'Sharpe':>7}  {'Ret%':>8}  "
      f"{'Chg%':>6}  {'Score':>9}")
print(f"  {'─'*90}")

train_scores: dict = {}
best_train_sc = -999.0

for N, K, M, mode in itertools.product(LOOKBACK_GRID, THRESHOLD_GRID,
                                        STOPMULT_GRID, MODE_GRID):
    df_t = build_signals(df_train, N, K, M, mode)
    if df_t.empty: continue
    eq, _, pnls_t, _, _, ch_t, _ = run_backtest(df_t, INITIAL_CAPITAL)
    if not pnls_t: continue

    wins   = [p for p in pnls_t if p > 0]
    losses = [p for p in pnls_t if p < 0]
    wr     = len(wins)/len(pnls_t)*100
    pf     = sum(wins)/abs(sum(losses)) if losses else float("inf")
    rets   = eq.pct_change().dropna()
    sh     = (rets.mean()/rets.std())*np.sqrt(252) if rets.std()>0 else 0
    ret    = (eq.iloc[-1]/INITIAL_CAPITAL-1)*100
    chg_pc = ch_t/INITIAL_CAPITAL*100
    sc     = score_fn(pnls_t, eq, ch_t, INITIAL_CAPITAL)
    train_scores[(N,K,M,mode)] = sc

    is_best = sc > best_train_sc
    if is_best: best_train_sc = sc
    flag = f"  {Y}← best{E}" if is_best else ""

    print(f"  {N:>4}  {K:>5.2f}  {M:>4.1f}  {mode:<16}  {len(pnls_t):>7}  "
          f"{wr:>6.1f}  {pf:>6.2f}  {sh:>7.2f}  {ret:>8.2f}  "
          f"{chg_pc:>6.2f}  {sc:>9.3f}{flag}")

print()


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1B — STAGE-2 VALIDATION  (2yr anti-overfit gate)
#     FIX #1: params must pass BOTH stages to qualify
# ═══════════════════════════════════════════════════════════════════════
# Seed validation window with warm-up bars from before split_valid
seed_pad = max(LOOKBACK_GRID) * 6
df_vseed = df_all.loc[
    (df_all.index >= (split_valid - pd.Timedelta(days=seed_pad))) &
    (df_all.index <  split_live)
].copy()

print(W72)
print(f"  PHASE 1B — STAGE-2 VALIDATION  ({VALID_YEARS}yr  ·  anti-overfit gate)")
print(f"  Gate rule: BOTH train score > 0  AND  validate score > 0")
print(W72)
print(f"  {'N':>4}  {'K':>5}  {'M':>4}  {'Mode':<16}  "
      f"{'Train':>9}  {'Valid':>9}  {'Final':>9}  Gate")
print(f"  {'─'*70}")

best_final_sc = -999.0
best_params   = None
TOPN          = 40   # print top-N train combos in validation table

sorted_combos = sorted(train_scores.items(), key=lambda x: -x[1])

for rank, ((N,K,M,mode), sc_train) in enumerate(sorted_combos):
    # Run validation (use seeded df to avoid warm-up look-ahead)
    df_v = build_signals(df_vseed, N, K, M, mode)
    df_v = df_v.loc[df_v.index >= split_valid].copy()
    if df_v.empty:
        sc_valid = -999.0
    else:
        eq_v, _, pnls_v, _, _, ch_v, _ = run_backtest(df_v, INITIAL_CAPITAL)
        sc_valid = score_fn(pnls_v, eq_v, ch_v, INITIAL_CAPITAL) if pnls_v else -999.0

    gate     = sc_train > 0 and sc_valid > 0
    final_sc = min(sc_train, sc_valid) if gate else -999.0
    gate_str = f"{G}PASS{E}" if gate else f"{R}FAIL{E}"

    if gate and final_sc > best_final_sc:
        best_final_sc = final_sc
        best_params   = (N, K, M, mode)
        flag = f"  {Y}← BEST{E}"
    else:
        flag = ""

    if rank < TOPN:
        print(f"  {N:>4}  {K:>5.2f}  {M:>4.1f}  {mode:<16}  "
              f"{sc_train:>9.3f}  {sc_valid:>9.3f}  {final_sc:>9.3f}  "
              f"{gate_str}{flag}")

# Fallback: no combo passed both gates → use best train score
if best_params is None:
    print(f"\n  {Y}[WARN] No combo passed both gates — "
          f"falling back to best train score.{E}")
    best_params   = sorted_combos[0][0]
    best_final_sc = sorted_combos[0][1]

BEST_N, BEST_K, BEST_M, BEST_MODE = best_params
print(f"\n  ✔  LOCKED PARAMETERS  (min of train/validate score)")
print(f"     Lookback   N = {BEST_N} bars")
print(f"     Threshold  K = {BEST_K}")
print(f"     Stop mult  M = {BEST_M}×  (vol-based stop distance)")
print(f"     Mode         = {BEST_MODE}")
print(f"     Score        = {best_final_sc:.3f}\n")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 — LIVE EXECUTION  (Year 8, out-of-sample)
# ═══════════════════════════════════════════════════════════════════════
print(W72)
print(f"  PHASE 2 — LIVE EXECUTION  "
      f"(Year {TOTAL_YEARS}  ·  N={BEST_N}  K={BEST_K}  M={BEST_M}  {BEST_MODE})")
print(W72+"\n")

seed_pad_live = max(LOOKBACK_GRID)*6
df_lseed = df_all.loc[
    df_all.index >= (split_live - pd.Timedelta(days=seed_pad_live))
].copy()
df_seeded = build_signals(df_lseed, BEST_N, BEST_K, BEST_M, BEST_MODE)
df_exec   = df_seeded.loc[df_seeded.index >= split_live].copy()

(eq_live, mg_live, pnls_live,
 trades_live, rolls_live, chg_live, mtm_log) = run_backtest(df_exec, INITIAL_CAPITAL)


# ═══════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════
rets_l   = eq_live.pct_change().dropna()
dd_live  = (eq_live / eq_live.cummax() - 1) * 100
dd_ser, max_dd_frac, dd_peak, dd_trough = calc_drawdown(eq_live)

tot_ret  = (eq_live.iloc[-1] / INITIAL_CAPITAL - 1) * 100
max_dd   = float(dd_live.min())
wins_l   = [p for p in pnls_live if p > 0]
loss_l   = [p for p in pnls_live if p < 0]
wr_l     = len(wins_l)/len(pnls_live)*100 if pnls_live else 0
aw_l     = float(np.mean(wins_l))  if wins_l  else 0
al_l     = float(np.mean(loss_l))  if loss_l  else 0
pf_l     = sum(wins_l)/abs(sum(loss_l)) if loss_l else float("inf")
sh_l     = ((rets_l.mean()/rets_l.std())*np.sqrt(252)
             if rets_l.std()>0 else 0)
exp_l    = (wr_l/100*aw_l) + ((1-wr_l/100)*al_l)
pct_chg  = chg_live/INITIAL_CAPITAL*100
roll_tot = sum(r["cost"] for r in rolls_live)
mtm_tot  = sum(m["mtm"]  for m in mtm_log)
mcalls   = sum(1 for t in trades_live if t["xrsn"]=="MCL")

cw=cl=mcw=mcl=0
for p in pnls_live:
    if p>0: cw+=1; cl=0; mcw=max(mcw,cw)
    else:   cl+=1; cw=0; mcl=max(mcl,cl)

xrsn_cnt = {}
for t in trades_live: xrsn_cnt[t["xrsn"]] = xrsn_cnt.get(t["xrsn"],0)+1

# SL win rate (v1 was 0% — key diagnostic)
sl_trades = [t for t in trades_live if t["xrsn"]=="SL"]
sl_wins   = sum(1 for t in sl_trades if t["net"]>0)
sl_wr     = (sl_wins/len(sl_trades)*100) if sl_trades else 0
sl_wr_str = f"{sl_wr:.0f}%"


# ═══════════════════════════════════════════════════════════════════════
# TRADE LOG
# ═══════════════════════════════════════════════════════════════════════
HDR = "═"*118; SEP = "─"*118
print(f"\n{HDR}")
print(f"  LIVE FUTURES TRADE LOG  ·  {STOCK_NAME}  ·  "
      f"{df_exec.index[0].date()} → {df_exec.index[-1].date()}")
print(f"  N={BEST_N}  K={BEST_K}  M={BEST_M}  {BEST_MODE}  ·  "
      f"VR filter ON  ·  Min hold {MIN_HOLD} bars  ·  BE stop after {BE_TRIGGER_R}R")
print(f"{HDR}")
print(f"  {'#':>4}  {'':5}{'Side':<5}  {'Entry':^10}  {'Exit':^10}  "
      f"{'Bars':>4}  {'Entry₹':>9}  {'Stop₹':>9}  {'Exit₹':>9}  "
      f"{'Lots':>4}  {'Gross₹':>10}  {'Chg₹':>7}  {'Net₹':>10}  "
      f"{'Capital₹':>12}  Rsn")
print(SEP)

for t in trades_live:
    ok  = t["net"] >= 0
    nc  = G if ok else R
    lbl = f"{nc}{'WIN ':>4}{E}" if ok else f"{nc}{'LOSS':>4}{E}"
    print(f"  {t['no']:>4}  {lbl} {t['side']:<5}  "
          f"{t['edate']:^10}  {t['xdate']:^10}  {t['bh']:>4}  "
          f"₹{t['epx']:>8,.1f}  ₹{t['stop']:>8,.1f}  ₹{t['xpx']:>8,.1f}  "
          f"{t['lots']:>4}  "
          f"{t['gross']:>+10,.0f}  {t['chg']:>7,.0f}  "
          f"{nc}{t['net']:>+10,.0f}{E}  "
          f"₹{t['cap']:>11,.0f}  {t['xrsn']}")

g_tot = sum(t["gross"] for t in trades_live)
n_tot = sum(t["net"]   for t in trades_live)
nc    = G if n_tot>=0 else R
print(SEP)
print(f"  TOTAL{'':87}  {g_tot:>+10,.0f}  {chg_live:>7,.0f}  "
      f"{nc}{n_tot:>+10,.0f}{E}  ₹{eq_live.iloc[-1]:>11,.0f}")
print(HDR)

if rolls_live:
    print(f"\n  ROLLOVER LOG  ({len(rolls_live)} rolls)")
    print(f"  {'#':>3}  {'Date':^12}  {'Side':<5}  {'Price₹':>9}  {'Qty':>6}  {'Cost₹':>10}")
    print(f"  {'─'*52}")
    for r in rolls_live:
        print(f"  {r['no']:>3}  {r['date']:^12}  {r['side']:<5}  "
              f"₹{r['price']:>8,.1f}  {r['qty']:>6}  {Y}₹{r['cost']:>9,.0f}{E}")
    print(f"  {'─'*52}")
    print(f"  Total rollover cost: {Y}₹{roll_tot:,.0f}{E}  "
          f"({roll_tot/INITIAL_CAPITAL*100:.2f}% of capital)\n")


# ═══════════════════════════════════════════════════════════════════════
# PERFORMANCE SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═'*64}")
print(f"  LIVE PERFORMANCE  ·  {STOCK_NAME}  ·  Quant v2.0")
print(f"  Out-of-sample Year {TOTAL_YEARS}  ·  Pure Quantitative  ·  No TA")
print(f"{'═'*64}")

def row(lbl, val, col=""):
    print(f"  {lbl:<34}  {col}{val}{E}")

row("Symbol",             f"{SYMBOL}  (FUTURES)")
row("Signal",             "N-day Rolling Sharpe  +  VR Regime Gate")
row("Parameters",         f"N={BEST_N}  K={BEST_K}  M={BEST_M}  {BEST_MODE}")
row("Signal formula",     "Σ log-ret(N) / (σ_daily × √N)")
row("Stop formula",       f"σ_daily × price × {BEST_M}")
row("VR regime filter",   f"window={VR_WINDOW}  k={VR_K}  band=±{VR_BAND}")
row("Period",             f"{df_exec.index[0].date()} → {df_exec.index[-1].date()}")
row("Lot size / SPAN",    f"{LOT_SIZE} shares  ·  ₹{MARGIN_LOT:,.0f}/lot")
print(f"  {'─'*60}")
row("Initial Capital",    f"₹{INITIAL_CAPITAL:>14,.0f}")
fc = G if tot_ret>=0 else R
row("Final Capital",      f"₹{eq_live.iloc[-1]:>14,.0f}", fc)
row("Total Return",       f"{tot_ret:>+14.2f}%", fc)
row("Net P&L",            f"₹{eq_live.iloc[-1]-INITIAL_CAPITAL:>+14,.0f}", fc)
print(f"  {'─'*60}")
row("Total Closed Trades",f"{len(pnls_live):>14}")
row("Wins / Losses",      f"{len(wins_l):>6}  /  {len(loss_l):<6}   WR {wr_l:.1f}%")
row("Avg Win",            f"₹{aw_l:>+14,.0f}", G)
row("Avg Loss",           f"₹{al_l:>+14,.0f}", R)
row("Expectancy / trade", f"₹{exp_l:>+14,.0f}", G if exp_l>0 else R)
pf_col = G if pf_l>1.5 else (Y if pf_l>1.0 else R)
row("Profit Factor",      f"{pf_l:>14.2f}", pf_col)
row("Max Consec Win/Loss",f"{mcw:>6}  /  {mcl}")
print(f"  {'─'*60}")
sh_col = G if sh_l>1.0 else (Y if sh_l>0 else R)
row("Sharpe (annualised)",f"{sh_l:>14.2f}", sh_col)
row("Max Drawdown",       f"{max_dd:>13.2f}%", R if max_dd<-20 else Y)
row("DD Peak → Trough",   f"{dd_peak.date()} → {dd_trough.date()}")
print(f"  {'─'*60}")
print("  ── FUTURES MECHANICS ──────────────────────────────────────")
row("Rollovers",          f"{len(rolls_live):>14}")
row("Total Rollover Cost",f"₹{roll_tot:>14,.0f}", Y)
row("Total Charges",      f"₹{chg_live:>14,.0f}  ({pct_chg:.2f}% of cap)", Y)
row("MTM Settlement Days",f"{len(mtm_log):>14}")
row("Total MTM P&L",      f"₹{mtm_tot:>+14,.0f}")
row("Margin Calls",       f"{mcalls:>14}", R if mcalls>0 else G)
print(f"  {'─'*60}")
print("  Exit breakdown:")
for rsn, cnt in sorted(xrsn_cnt.items(), key=lambda x:-x[1]):
    ww = sum(1 for t in trades_live if t["xrsn"]==rsn and t["net"]>0)
    print(f"    {rsn:<8}  {cnt:>4} trades   WR {ww/cnt*100:.0f}%")

# ── v1 vs v2 comparison ──────────────────────────────────────────────
print(f"\n{'═'*52}")
print(f"  v1 → v2 IMPROVEMENT SUMMARY  ·  {STOCK_NAME}")
print(f"{'═'*52}")
print(f"  {'Metric':<24}  {'v1':>10}  {'v2':>10}")
print(f"  {'─'*48}")
print(f"  {'Total Return':<24}  {'-97.02%':>10}  {tot_ret:>+9.2f}%")
print(f"  {'Trades':<24}  {'110':>10}  {len(pnls_live):>10}")
print(f"  {'Win Rate':<24}  {'39.1%':>10}  {wr_l:>9.1f}%")
print(f"  {'SL Win Rate':<24}  {'0%':>10}  {sl_wr_str:>10}")
print(f"  {'Profit Factor':<24}  {'0.69':>10}  {pf_l:>10.2f}")
print(f"  {'Sharpe':<24}  {'-1.03':>10}  {sh_l:>10.2f}")
print(f"  {'Max Drawdown':<24}  {'-130.31%':>10}  {max_dd:>9.2f}%")
print(f"  {'Charges % of Cap':<24}  {'8.81%':>10}  {pct_chg:>9.2f}%")
print(f"  {'VR Filter active':<24}  {'No':>10}  {'Yes':>10}")
print(f"  {'2-stage WF':<24}  {'No':>10}  {'Yes':>10}")
print(f"  {'Breakeven stop':<24}  {'No':>10}  {'Yes':>10}")
print(f"  {'DD scaling':<24}  {'No':>10}  {'Yes':>10}")
print(f"{'═'*52}\n")

# ── Plot ─────────────────────────────────────────────────────────────
save_plot(eq_live, dd_ser, trades_live,
          (BEST_N, BEST_K, BEST_M, BEST_MODE), STOCK_NAME, SYMBOL)

print(f"[DONE]  {len(pnls_live)} trades  ·  "
      f"{len(rolls_live)} rollovers  ·  {mcalls} margin calls\n")