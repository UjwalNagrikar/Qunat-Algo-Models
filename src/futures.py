
import datetime as dt
import os, sys, warnings
warnings.filterwarnings("ignore")
import pytz, yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

# ═══════════════════════════════════════════════════════════════════════
# 1.  FUTURES DATABASE
# ═══════════════════════════════════════════════════════════════════════
DB = {
    "RELIANCE.NS" : (250,  150_000, "Reliance Industries"),
}

# ═══════════════════════════════════════════════════════════════════════
# 2.  CONFIG
# ═══════════════════════════════════════════════════════════════════════
SYMBOL          = "RELIANCE.NS"
INITIAL_CAPITAL = 1_000_000        # ₹ 10 Lakh

# ── Walk-Forward ─────────────────────────────────────────────────────
TOTAL_YEARS     = 8
TRAIN_YEARS     = 7
LIVE_YEARS      = 1

# ── N_BAR candidates (training picks the best) ───────────────────────
N_BAR_CANDIDATES = [10, 15, 20, 30, 40, 55, 80]

# ── Risk Management ──────────────────────────────────────────────────
RISK_PCT        = 0.02             # 2% capital per trade
MAX_LOTS        = 10
MARGIN_UTIL     = 0.60             # use 60% of capital for margin
TARGET_R        = 2.0              # TP = 2× risk distance
COOLDOWN        = 2

# ── Futures-specific ─────────────────────────────────────────────────
ROLL_DAY_BEFORE = 1                # roll N days before expiry (1 = day before)
MARGIN_CALL_PCT = 0.75             # force-close if margin_balance < 75% of SPAN

# ── NSE Charges (Zerodha 2024) ───────────────────────────────────────
BROKERAGE  = 20.0
STT_RATE   = 0.000125              # STT on SELL side only (futures)
EXCH_RATE  = 0.000019
SEBI_RATE  = 0.000001
STAMP_RATE = 0.00002               # stamp on BUY side
GST_RATE   = 0.18

OUT_DIR    = "outputs"   # Local outputs directory
IST        = pytz.timezone("Asia/Kolkata")
SAVE_PLOTS = True  # Set True to save PNG files

# ═══════════════════════════════════════════════════════════════════════
# 3.  NSE EXPIRY CALENDAR
#     Last Thursday of every month.
#     If Thursday is a market holiday → previous Wednesday.
#     (We approximate: use last Thursday, flag in output)
# ═══════════════════════════════════════════════════════════════════════
def last_thursday(year: int, month: int) -> dt.date:
    """Return the last Thursday of a given year/month."""
    # Start from last day of month, walk back to Thursday
    if month == 12:
        last_day = dt.date(year+1, 1, 1) - dt.timedelta(days=1)
    else:
        last_day = dt.date(year, month+1, 1) - dt.timedelta(days=1)
    offset = (last_day.weekday() - 3) % 7   # 3 = Thursday
    return last_day - dt.timedelta(days=offset)


def build_expiry_calendar(start: dt.date, end: dt.date) -> list:
    """All monthly expiry dates between start and end."""
    expiries = []
    y, m = start.year, start.month
    while True:
        exp = last_thursday(y, m)
        if exp > end: break
        if exp >= start:
            expiries.append(exp)
        m += 1
        if m > 12: m = 1; y += 1
    return expiries


# ═══════════════════════════════════════════════════════════════════════
# 4.  HELPERS
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
    """
    Rolling = closing old contract + opening new contract.
    Charge both legs fully.
    Futures prices of near/far month are very close at roll time
    so we use same price for simplicity (conservative — real roll
    spread is typically ₹0.5–2 per share for liquid stocks).
    """
    exit_chg  = charges(price, qty, "SELL")   # close near-month
    entry_chg = charges(price, qty, "BUY")    # open far-month
    return exit_chg + entry_chg


def lot_size_fn(capital: float, entry_px: float, stop_px: float) -> int:
    risk_per_share = abs(entry_px - stop_px)
    if risk_per_share <= 0: return 0
    lots_risk   = int(capital * RISK_PCT / (risk_per_share * LOT_SIZE))
    lots_margin = int(capital * MARGIN_UTIL / MARGIN_LOT)
    lots = min(lots_risk, lots_margin, MAX_LOTS)
    if lots == 0 and lots_margin >= 1: lots = 1
    return max(lots, 0)


def build_donchian(df: pd.DataFrame, n: int) -> pd.DataFrame:
    out = df.copy()
    out["dc_high"]   = df["h"].shift(1).rolling(n).max()
    out["dc_low"]    = df["l"].shift(1).rolling(n).min()
    out["dc_mid"]    = (out["dc_high"] + out["dc_low"]) / 2
    out["long_sig"]  = df["c"] > out["dc_high"]
    out["short_sig"] = df["c"] < out["dc_low"]
    out.dropna(inplace=True)
    return out


def calc_drawdown(equity: pd.Series):
    """Return drawdown series, max drawdown fraction, peak and trough dates."""
    peak = equity.cummax()
    dd = (equity - peak) / peak
    trough = dd.idxmin()
    peak_date = equity.loc[:trough].idxmax()
    return dd, dd.min(), peak_date, trough


# ═══════════════════════════════════════════════════════════════════════
# 5.  FUTURES BACKTEST ENGINE
#     Adds on top of basic engine:
#       (a) Daily MTM settlement
#       (b) Rollover on ROLL_DAY_BEFORE expiry
#       (c) Margin call check after MTM
# ═══════════════════════════════════════════════════════════════════════
def run_futures_backtest(df: pd.DataFrame, capital_start: float):
    """
    Returns:
        equity     : pd.Series  — daily portfolio value (capital + unrealised)
        margin_acc : pd.Series  — margin account balance (cash available)
        pnls       : list       — realised net P&L per closed trade
        trades     : list       — trade records
        rolls      : list       — rollover events
        total_chg  : float      — total charges paid
        mtm_log    : list       — daily MTM settlements
    """
    # Build expiry calendar for this df's date range
    start_d  = df.index[0].date()
    end_d    = df.index[-1].date()
    expiries = build_expiry_calendar(start_d, end_d)
    exp_set  = set(expiries)

    # Roll dates = ROLL_DAY_BEFORE trading days before each expiry
    # We find roll dates by looking at actual trading dates in df
    trading_dates = [idx.date() for idx in df.index]
    roll_dates = set()
    for exp in expiries:
        # Find index of expiry in trading dates (or closest before)
        candidates = [d for d in trading_dates if d <= exp]
        if not candidates: continue
        exp_idx = trading_dates.index(candidates[-1])
        roll_idx = max(0, exp_idx - ROLL_DAY_BEFORE)
        roll_dates.add(trading_dates[roll_idx])

    # State
    cap       = capital_start    # total capital (realised)
    margin_acc= capital_start    # margin account = cash for margin
    side      = None
    epx       = stop_px = tp_px = prev_close = 0.0
    etime     = None
    nlots     = qty = 0
    cdl       = 0
    span_margin_held = 0.0       # margin blocked = nlots × MARGIN_LOT

    eq_curve  = []
    margin_curve = []
    pnls      = []
    trades    = []
    rolls     = []
    mtm_log   = []
    total_chg = 0.0
    tno = rno = 0

    for i in range(len(df)):
        row = df.iloc[i]; bt = df.index[i]
        date_today = bt.date()
        o = float(row["o"]); h = float(row["h"])
        l = float(row["l"]); c = float(row["c"])
        dc_h = float(row["dc_high"])
        dc_l = float(row["dc_low"])
        dc_m = float(row["dc_mid"])
        if cdl > 0: cdl -= 1

        # ── DAILY MTM SETTLEMENT ─────────────────────────────────────
        # NSE settles futures daily at closing price.
        # Gain/loss is credited/debited to margin account same day.
        if side and prev_close > 0:
            mtm_pnl = (c - prev_close) * qty * (1 if side=="LONG" else -1)
            margin_acc += mtm_pnl          # settled to cash
            cap        += mtm_pnl          # total capital moves too
            mtm_log.append(dict(
                date=str(date_today), mtm=round(mtm_pnl,2),
                margin_acc=round(margin_acc,2)
            ))

            # ── MARGIN CALL CHECK ────────────────────────────────────
            # If margin account falls below MARGIN_CALL_PCT of SPAN margin
            # broker force-closes position at open of next bar (we use today's close)
            if margin_acc < span_margin_held * MARGIN_CALL_PCT:
                xpx  = round(c, 2); xrsn = "MCL"   # Margin Call Liquidation
                grs  = (xpx - epx) * qty * (1 if side=="LONG" else -1)
                chg  = charges(xpx, qty, "SELL" if side=="LONG" else "BUY")
                net  = grs - chg
                # Note: MTM already updated cap, so we adjust back for gross
                # The net here is the TOTAL trade P&L (all MTM + final close)
                pnls.append(net); tno += 1; total_chg += chg
                margin_acc += span_margin_held    # release margin
                cap        -= chg
                trades.append(dict(
                    no=tno, side=side, sig="MCL",
                    edate=str(etime)[:10], xdate=str(date_today),
                    bh=i - list(df.index).index(etime) if etime in df.index else 0,
                    epx=epx, xpx=xpx, stop=stop_px,
                    lots=nlots, qty=int(qty),
                    gross=round(grs,2), chg=round(chg,2), net=round(net,2),
                    cap=round(cap), xrsn=xrsn
                ))
                side=None; nlots=qty=0; etime=None
                span_margin_held=0.0; prev_close=0.0; cdl=COOLDOWN

        # ── ROLLOVER CHECK ───────────────────────────────────────────
        # If holding a position and today is a roll date → roll to next month
        if side and date_today in roll_dates:
            roll_chg = rollover_cost(c, qty)
            margin_acc -= roll_chg
            cap        -= roll_chg
            total_chg  += roll_chg
            rno += 1
            rolls.append(dict(
                no=rno, date=str(date_today),
                side=side, price=round(c,2),
                qty=int(qty), cost=round(roll_chg,2),
                cap_after=round(cap,2)
            ))
            # Position continues — same side, same lots, new contract
            # Stop and TP remain as-is (price-based, not contract-based)

        # ── EXIT CHECK ───────────────────────────────────────────────
        if side:
            xpx = xrsn = None

            # 1. Stop loss
            if side=="LONG"  and l <= stop_px: xpx=round(stop_px,2); xrsn="SL"
            elif side=="SHORT" and h >= stop_px: xpx=round(stop_px,2); xrsn="SL"

            # 2. Take profit
            if xpx is None and TARGET_R > 0:
                if side=="LONG"  and h >= tp_px: xpx=round(tp_px,2); xrsn="TP"
                elif side=="SHORT" and l<=tp_px:  xpx=round(tp_px,2); xrsn="TP"

            # 3. Channel midpoint
            if xpx is None:
                if side=="LONG"  and c<=dc_m: xpx=round(c,2); xrsn="MID"
                elif side=="SHORT" and c>=dc_m: xpx=round(c,2); xrsn="MID"

            # 4. Opposite signal
            if xpx is None:
                if side=="LONG"  and row["short_sig"]: xpx=round(c,2); xrsn="FLIP"
                elif side=="SHORT" and row["long_sig"]:  xpx=round(c,2); xrsn="FLIP"

            if xpx is not None:
                grs = (xpx - epx) * qty * (1 if side=="LONG" else -1)
                chg = charges(xpx, qty, "SELL" if side=="LONG" else "BUY")
                net = grs - chg
                margin_acc += span_margin_held    # release blocked margin
                margin_acc -= chg
                cap        -= chg
                total_chg  += chg; pnls.append(net); tno += 1
                try:   bh = i - list(df.index).index(etime)
                except: bh = 0
                trades.append(dict(
                    no=tno, side=side, sig="SIGNAL",
                    edate=str(etime)[:10], xdate=str(date_today), bh=bh,
                    epx=epx, xpx=xpx, stop=stop_px,
                    lots=nlots, qty=int(qty),
                    gross=round(grs,2), chg=round(chg,2), net=round(net,2),
                    cap=round(cap), xrsn=xrsn
                ))
                side=None; nlots=qty=0; etime=None
                span_margin_held=0.0; prev_close=0.0; cdl=COOLDOWN

        # ── ENTRY CHECK ──────────────────────────────────────────────
        if not side and cdl == 0:
            sig = "LONG" if row["long_sig"] else ("SHORT" if row["short_sig"] else None)
            if sig:
                s_px  = dc_l if sig=="LONG" else dc_h
                lots  = lot_size_fn(margin_acc, o, s_px)  # size against cash available
                if lots > 0:
                    rd       = abs(o - s_px) or o * 0.005
                    span_req = lots * MARGIN_LOT           # margin blocked by broker
                    if margin_acc >= span_req:             # can we afford margin?
                        epx          = round(o, 2)
                        stop_px      = round(s_px, 2)
                        tp_px        = round(epx + TARGET_R*rd, 2) if sig=="LONG" \
                                       else round(epx - TARGET_R*rd, 2)
                        side         = sig
                        etime        = bt
                        nlots        = lots
                        qty          = lots * LOT_SIZE
                        prev_close   = o                   # seed MTM with entry price
                        chg          = charges(epx, qty, "BUY" if side=="LONG" else "SELL")
                        span_margin_held = span_req
                        margin_acc  -= chg                 # pay entry charge
                        cap         -= chg
                        total_chg   += chg

        # ── SNAPSHOT ─────────────────────────────────────────────────
        # Update prev_close for next day's MTM
        if side:
            prev_close = c

        unr = (c - epx) * qty * (1 if side=="LONG" else -1) if side else 0.0
        eq_curve.append(cap + unr)
        margin_curve.append(margin_acc)

    equity     = pd.Series(eq_curve,   index=df.index)
    margin_ser = pd.Series(margin_curve, index=df.index)
    return equity, margin_ser, pnls, trades, rolls, total_chg, mtm_log


def score(pnls, equity):
    if len(pnls) < 5: return -999
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf     = sum(wins)/abs(sum(losses)) if losses else 0
    rets   = equity.pct_change().dropna()
    sh     = (rets.mean()/rets.std())*np.sqrt(252) if rets.std()>0 else 0
    return pf * sh


# ═══════════════════════════════════════════════════════════════════════
# 6.  DATA FETCH
# ═══════════════════════════════════════════════════════════════════════
if SYMBOL not in DB:
    raise ValueError(f"'{SYMBOL}' not in DB.")
LOT_SIZE, MARGIN_LOT, STOCK_NAME = DB[SYMBOL]

print(f"\n{'═'*70}")
print(f"  {STOCK_NAME}  ({SYMBOL})")
print(f"  NSE FUTURES ONLY  ·  {N_BAR_CANDIDATES[0]}-{N_BAR_CANDIDATES[-1]} Bar Donchian")
print(f"  Walk-Forward: Train {TRAIN_YEARS}yr → Live {LIVE_YEARS}yr")
print(f"  Futures mechanics: MTM + Rollover + Margin Call")
print(f"{'═'*70}\n")

end   = dt.datetime.now(tz=pytz.utc)
start = end - dt.timedelta(days=365*TOTAL_YEARS + 120)

raw = yf.download(SYMBOL, start=start, end=end,
                  interval="1d", auto_adjust=True, progress=False)
if raw.empty: sys.exit("No data — check symbol / network.")

if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
raw.rename(columns={"Open":"o","High":"h","Low":"l","Close":"c","Volume":"v"},
           inplace=True)
raw = raw[["o","h","l","c","v"]].copy()
if raw.index.tzinfo is None:
    raw.index = raw.index.tz_localize("UTC")
raw.index = raw.index.tz_convert(IST)

cutoff_all = raw.index[-1] - pd.Timedelta(days=365*TOTAL_YEARS)
split_date = raw.index[-1] - pd.Timedelta(days=365*LIVE_YEARS)

df_all   = raw.loc[raw.index >= cutoff_all].copy()
df_train = df_all.loc[df_all.index <  split_date].copy()

print(f"  Full data   : {df_all.index[0].date()} → {df_all.index[-1].date()} "
      f"({len(df_all)} bars)")
print(f"  Train       : {df_train.index[0].date()} → {df_train.index[-1].date()} "
      f"({len(df_train)} bars)")
print(f"  Live        : {split_date.date()} → {df_all.index[-1].date()}\n")

# Show expiry calendar for live year
live_start = split_date.date()
live_end   = df_all.index[-1].date()
live_expiries = build_expiry_calendar(live_start, live_end)
print(f"  NSE Expiry dates in live window ({len(live_expiries)} expiries):")
for i, exp in enumerate(live_expiries):
    print(f"    {exp.strftime('%d %b %Y')}  (last Thu)", end="")
    if (i+1) % 4 == 0: print()
print("\n")

# ═══════════════════════════════════════════════════════════════════════
# 7.  PHASE 1 — TRAINING
# ═══════════════════════════════════════════════════════════════════════
print(f"{'═'*70}")
print("  PHASE 1  —  TRAINING  (7yr, futures engine)")
print(f"  Candidates: {N_BAR_CANDIDATES}")
print(f"{'═'*70}")
print(f"  {'N':>4}  {'Trades':>6}  {'Rolls':>5}  {'WR%':>6}  {'PF':>6}  "
      f"{'Sharpe':>7}  {'Ret%':>8}  {'MDD%':>8}  {'Score':>8}")
print(f"  {'─'*66}")

train_results = {}

for n in N_BAR_CANDIDATES:
    df_t = build_donchian(df_train, n)
    eq, _, pnls_t, _, rolls_t, _, _ = run_futures_backtest(df_t, INITIAL_CAPITAL)

    if not pnls_t:
        print(f"  {n:>4}  — no trades"); continue

    wins   = [p for p in pnls_t if p > 0]
    losses = [p for p in pnls_t if p < 0]
    wr     = len(wins)/len(pnls_t)*100
    pf     = sum(wins)/abs(sum(losses)) if losses else float("inf")
    rets   = eq.pct_change().dropna()
    sh     = (rets.mean()/rets.std())*np.sqrt(252) if rets.std()>0 else 0
    ret    = (eq.iloc[-1]/INITIAL_CAPITAL - 1)*100
    dd     = (eq/eq.cummax()-1).min()*100
    sc     = score(pnls_t, eq)
    train_results[n] = sc

    flag = "  ← best" if sc == max(train_results.values()) else ""
    print(f"  {n:>4}  {len(pnls_t):>6}  {len(rolls_t):>5}  {wr:>6.1f}  {pf:>6.2f}  "
          f"{sh:>7.2f}  {ret:>8.2f}  {dd:>8.2f}  {sc:>8.3f}{flag}")

BEST_N = max(train_results, key=train_results.get)
print(f"\n  ✔  LOCKED: N_BAR = {BEST_N}  "
      f"(score = {train_results[BEST_N]:.3f})\n")

# ═══════════════════════════════════════════════════════════════════════
# 8.  PHASE 2 — LIVE EXECUTION  (Year 8, futures)
# ═══════════════════════════════════════════════════════════════════════
print(f"{'═'*70}")
print(f"  PHASE 2  —  LIVE EXECUTION  (Year 8, N_BAR={BEST_N}, FUTURES)")
print(f"{'═'*70}\n")

df_seed   = df_all.loc[
    df_all.index >= (split_date - pd.Timedelta(days=BEST_N * 3))
].copy()
df_seeded = build_donchian(df_seed, BEST_N)
df_exec   = df_seeded.loc[df_seeded.index >= split_date].copy()

(eq_live, margin_live, pnls_live,
 trades_live, rolls_live, chg_live, mtm_log) = run_futures_backtest(
    df_exec, INITIAL_CAPITAL
)

# ═══════════════════════════════════════════════════════════════════════
# 9.  METRICS
# ═══════════════════════════════════════════════════════════════════════
rets_l   = eq_live.pct_change().dropna()
dd_live  = (eq_live / eq_live.cummax() - 1) * 100

fin_dd, fin_dd_frac, fin_dd_peak, fin_dd_trough = calc_drawdown(eq_live)
print(f"\n[INFO] (Futures) Equity bars: {len(eq_live)}, max drawdown = {fin_dd_frac*100:.2f}%")
print(f"[INFO] (Futures) Drawdown peak→trough: {fin_dd_peak.date()} → {fin_dd_trough.date()}")

tot_ret  = (eq_live.iloc[-1] / INITIAL_CAPITAL - 1) * 100
max_dd   = dd_live.min()
wins_l   = [p for p in pnls_live if p > 0]
loss_l   = [p for p in pnls_live if p < 0]
wr_l     = len(wins_l)/len(pnls_live)*100 if pnls_live else 0
aw_l     = np.mean(wins_l)  if wins_l  else 0
al_l     = np.mean(loss_l)  if loss_l  else 0
pf_l     = sum(wins_l)/abs(sum(loss_l)) if loss_l else float("inf")
sh_l     = (rets_l.mean()/rets_l.std())*np.sqrt(252) if rets_l.std()>0 else 0
exp_l    = (wr_l/100*aw_l) + ((1-wr_l/100)*al_l)
pct_chg  = chg_live / INITIAL_CAPITAL * 100

total_roll_cost = sum(r["cost"] for r in rolls_live)
total_mtm       = sum(m["mtm"]  for m in mtm_log)

cw=cl=mcw=mcl=0
for p in pnls_live:
    if p>0: cw+=1; cl=0; mcw=max(mcw,cw)
    else:   cl+=1; cw=0; mcl=max(mcl,cl)

xrsn_cnt = {}
for t in trades_live:
    xrsn_cnt[t["xrsn"]] = xrsn_cnt.get(t["xrsn"],0)+1

# ═══════════════════════════════════════════════════════════════════════
# 10.  PRINT ALL LIVE TRADES
# ═══════════════════════════════════════════════════════════════════════
G="\033[92m"; R="\033[91m"; Y="\033[93m"; E="\033[0m"
SEP="─"*118

print(f"\n{'═'*118}")
print(f"  LIVE FUTURES TRADES  ·  {STOCK_NAME}  ·  {BEST_N}-Bar Donchian  "
      f"·  {df_exec.index[0].date()} → {df_exec.index[-1].date()}")
print(f"{'═'*118}")
print(f"  {'#':>4}  {'':4}  {'Side':<5}  {'Entry':^10}  {'Exit':^10}  "
      f"{'Bars':>4}  {'Entry₹':>9}  {'Stop₹':>9}  {'Exit₹':>9}  "
      f"{'Lots':>4}  {'Gross₹':>10}  {'Chg₹':>7}  {'Net₹':>10}  "
      f"{'Capital₹':>12}  Rsn")
print(SEP)

for t in trades_live:
    ok  = t["net"] >= 0
    nc  = G if ok else R
    lbl = f"{nc}{'WIN ':>4}{E}" if ok else f"{nc}{'LOSS':>4}{E}"
    print(
        f"  {t['no']:>4}  {lbl}  {t['side']:<5}  "
        f"{t['edate']:^10}  {t['xdate']:^10}  {t['bh']:>4}  "
        f"₹{t['epx']:>8,.1f}  ₹{t['stop']:>8,.1f}  ₹{t['xpx']:>8,.1f}  "
        f"{t['lots']:>4}  "
        f"{t['gross']:>+10,.0f}  {t['chg']:>7,.0f}  "
        f"{nc}{t['net']:>+10,.0f}{E}  "
        f"₹{t['cap']:>11,.0f}  {t['xrsn']}"
    )

g_tot = sum(t["gross"] for t in trades_live)
n_tot = sum(t["net"]   for t in trades_live)
nc = G if n_tot>=0 else R
print(SEP)
print(f"  TOTAL  {'':86}  {g_tot:>+10,.0f}  {chg_live:>7,.0f}  "
      f"{nc}{n_tot:>+10,.0f}{E}  ₹{eq_live.iloc[-1]:>11,.0f}")
print(f"{'═'*118}")

# ── Rollover log ─────────────────────────────────────────────────────
if rolls_live:
    print(f"\n  ROLLOVER LOG  ({len(rolls_live)} rolls in live window)")
    print(f"  {'#':>3}  {'Date':^12}  {'Side':<5}  {'Price₹':>9}  "
          f"{'Qty':>6}  {'Roll Cost₹':>11}")
    print(f"  {'─'*54}")
    for r in rolls_live:
        print(f"  {r['no']:>3}  {r['date']:^12}  {r['side']:<5}  "
              f"₹{r['price']:>8,.1f}  {r['qty']:>6}  "
              f"{Y}₹{r['cost']:>10,.0f}{E}")
    print(f"  {'─'*54}")
    print(f"  Total rollover cost: {Y}₹{total_roll_cost:,.0f}{E}  "
          f"({total_roll_cost/INITIAL_CAPITAL*100:.2f}% of capital)\n")

# ═══════════════════════════════════════════════════════════════════════
# 11.  PERFORMANCE SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═'*64}")
print(f"  LIVE FUTURES PERFORMANCE  ·  {STOCK_NAME}")
print(f"  Out-of-sample Year 8  ·  Futures Engine  ·  MTM + Rollover")
print(f"{'═'*64}")

def row(lbl, val, col=""):
    print(f"  {lbl:<32}  {col}{val}{E}")

row("Symbol / Instrument",   f"{SYMBOL}  FUTURES")
row("Strategy",              f"{BEST_N}-Bar Donchian  ·  Pure Price Action")
row("Period",                f"{df_exec.index[0].date()} → {df_exec.index[-1].date()}")
row("Lot Size / SPAN Margin",f"{LOT_SIZE} shares/lot  ·  ₹{MARGIN_LOT:,.0f}/lot")
print(f"  {'─'*60}")
row("Initial Capital",       f"₹{INITIAL_CAPITAL:>14,.0f}")
fc = G if tot_ret>=0 else R
row("Final Capital",         f"₹{eq_live.iloc[-1]:>14,.0f}", fc)
row("Total Return (1yr)",    f"{tot_ret:>+14.2f}%", fc)
row("Net P&L",               f"₹{eq_live.iloc[-1]-INITIAL_CAPITAL:>+14,.0f}", fc)
print(f"  {'─'*60}")
row("Total Closed Trades",   f"{len(pnls_live):>14}")
row("Wins / Losses",         f"{len(wins_l):>6}  /  {len(loss_l):<6}   WR {wr_l:.1f}%")
row("Avg Win",               f"₹{aw_l:>+14,.0f}", G)
row("Avg Loss",              f"₹{al_l:>+14,.0f}", R)
row("Expectancy / trade",    f"₹{exp_l:>+14,.0f}", G if exp_l>0 else R)
row("Profit Factor",         f"{pf_l:>14.2f}", G if pf_l>1.5 else Y if pf_l>1 else R)
row("Max Consec W / L",      f"{mcw:>6}  /  {mcl}")
print(f"  {'─'*60}")
row("Sharpe (ann.)",         f"{sh_l:>14.2f}", G if sh_l>1 else Y if sh_l>0 else R)
row("Max Drawdown",          f"{max_dd:>13.2f}%", R if max_dd<-20 else Y)
print(f"  {'─'*60}")
print("  ── FUTURES-SPECIFIC ────────────────────────────────────────")
row("Rollovers executed",    f"{len(rolls_live):>14}")
row("Total Rollover Cost",   f"₹{total_roll_cost:>14,.0f}", Y)
row("Total Brokerage/Chg",   f"₹{chg_live:>14,.0f}  ({pct_chg:.2f}% of cap)", Y)
row("MTM Settlements",       f"{len(mtm_log):>14}  days")
row("Total MTM P&L",         f"₹{total_mtm:>+14,.0f}")
margin_calls = sum(1 for t in trades_live if t["xrsn"]=="MCL")
row("Margin Calls",          f"{margin_calls:>14}", R if margin_calls>0 else G)
print(f"  {'─'*60}")
print("  Exit breakdown:")
for rsn, cnt in sorted(xrsn_cnt.items(), key=lambda x:-x[1]):
    ww = sum(1 for t in trades_live if t["xrsn"]==rsn and t["net"]>0)
    print(f"    {rsn:<8}  {cnt:>4} trades   WR {ww/cnt*100:.0f}%")
print(f"{'═'*64}\n")


# print equty cruv 
plt.figure(figsize=(12,6))

plt.plot(eq_live.index, eq_live.values, linewidth=2)
plt.axhline(INITIAL_CAPITAL, linestyle="--")

plt.title("Equity Curve")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")

plt.grid()

# Save
eq_path = "outputs/equity_curve.png"
plt.savefig(eq_path, bbox_inches="tight")

# Show
from IPython.display import Image, display
display(Image(eq_path))

plt.close()


# Max drawdown curve
plt.figure(figsize=(12,4))

dd = (eq_live / eq_live.cummax() - 1) * 100

plt.fill_between(dd.index, dd.values, 0, alpha=0.3)
plt.plot(dd.index, dd.values, linewidth=1)

plt.title("Drawdown (%)")
plt.xlabel("Date")
plt.ylabel("Drawdown %")

plt.grid()

# Save
dd_path = "outputs/drawdown.png"
plt.savefig(dd_path, bbox_inches="tight")

# Show
display(Image(dd_path))

plt.close()

print(f"\n[OK]  Complete.  {len(pnls_live)} trades  |  "
      f"{len(rolls_live)} rollovers  |  {margin_calls} margin calls\n")
