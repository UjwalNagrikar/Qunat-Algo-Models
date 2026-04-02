"""
╔══════════════════════════════════════════════════════════════════════╗
║   NSE STOCK FUTURES  ·  Pure Price Action                           ║
║   N-Bar Donchian Breakout  ·  FUTURES ONLY (No Options, No Spot)    ║
║                                                                      ║
║   FUTURES-SPECIFIC MECHANICS:                                        ║
║   1. Monthly expiry  — last Thursday of every month (NSE)           ║
║   2. Daily MTM       — P&L settled to margin account each day       ║
║   3. Rollover        — position rolled 1 day before expiry          ║
║   4. Rollover cost   — full entry+exit charges on roll              ║
║   5. Margin call     — position force-closed if margin breached     ║
║   6. Walk-Forward    — 7yr train → 1yr live execution               ║
╚══════════════════════════════════════════════════════════════════════╝
"""

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
    "TCS.NS"      : (150,  175_000, "Tata Consultancy"),
    "HDFCBANK.NS" : (550,  100_000, "HDFC Bank"),
    "INFY.NS"     : (400,  100_000, "Infosys"),
    "ICICIBANK.NS": (700,   90_000, "ICICI Bank"),
    "SBIN.NS"     : (1500,  80_000, "SBI"),
    "AXISBANK.NS" : (625,   90_000, "Axis Bank"),
    "KOTAKBANK.NS": (400,  130_000, "Kotak Bank"),
    "LT.NS"       : (175,  200_000, "L&T"),
    "TATAMOTORS.NS":(1425,  75_000, "Tata Motors"),
    "TATASTEEL.NS": (5500,  50_000, "Tata Steel"),
    "MARUTI.NS"   : (100,  160_000, "Maruti Suzuki"),
    "WIPRO.NS"    : (1500,  65_000, "Wipro"),
    "BHARTIARTL.NS":(475,  115_000, "Bharti Airtel"),
    "BAJFINANCE.NS":(125,  180_000, "Bajaj Finance"),
    "SUNPHARMA.NS": (700,   95_000, "Sun Pharma"),
    "HCLTECH.NS"  : (700,   85_000, "HCL Tech"),
    "ONGC.NS"     : (3850,  45_000, "ONGC"),
    "NTPC.NS"     : (3000,  42_000, "NTPC"),
    "JSWSTEEL.NS" : (1350,  75_000, "JSW Steel"),
    "M&M.NS"      : (700,   90_000, "M&M"),
    "ADANIENT.NS" : (125,  200_000, "Adani Enterprises"),
    "^NSEI"       : (75,   130_000, "NIFTY 50"),
    "^NSEBANK"    : (15,   125_000, "BANKNIFTY"),
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

OUT_DIR    = "/mnt/user-data/outputs"   # Colab → "/content"
IST        = pytz.timezone("Asia/Kolkata")

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

# ═══════════════════════════════════════════════════════════════════════
# 12.  CHARTS
# ═══════════════════════════════════════════════════════════════════════
os.makedirs(OUT_DIR, exist_ok=True)
sym = SYMBOL.replace(".NS","").replace("^","")

BG   = "#07090f"; CARD = "#0d1120"; GRID = "#161d30"
TXT  = "#dde3f0"; DIM  = "#4a5570"
BLU  = "#4f8ef7"; GRN  = "#22c55e"; RED  = "#ef4444"
YLW  = "#f59e0b"; WHT  = "#f0f4ff"; ORG  = "#fb923c"

fig = plt.figure(figsize=(24, 18), facecolor=BG)
gs  = gridspec.GridSpec(4, 2, figure=fig,
                        height_ratios=[2.8, 1.0, 0.9, 0.9],
                        hspace=0.06, wspace=0.18,
                        left=0.07, right=0.97, top=0.91, bottom=0.06)

ax_eq  = fig.add_subplot(gs[0, :])
ax_dd  = fig.add_subplot(gs[1, :], sharex=ax_eq)
ax_mg  = fig.add_subplot(gs[2, :], sharex=ax_eq)
ax_bar = fig.add_subplot(gs[3, 0])
ax_kpi = fig.add_subplot(gs[3, 1])

for ax in (ax_eq, ax_dd, ax_mg, ax_bar, ax_kpi):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=DIM, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(GRID)

ei = eq_live.index; ev = eq_live.values

# Equity curve
ax_eq.plot(ei, ev, lw=2.0, color=BLU, zorder=4)
ax_eq.fill_between(ei, INITIAL_CAPITAL, ev,
                   where=ev>=INITIAL_CAPITAL,
                   color=GRN, alpha=0.18, interpolate=True)
ax_eq.fill_between(ei, INITIAL_CAPITAL, ev,
                   where=ev<INITIAL_CAPITAL,
                   color=RED, alpha=0.25, interpolate=True)
ax_eq.axhline(INITIAL_CAPITAL, color=WHT, lw=0.7, ls=(0,(5,4)), alpha=0.3)
ax_eq.plot(ei, eq_live.cummax().values, lw=0.7, color=WHT, alpha=0.12, ls="--")

# Mark rollover dates on equity
for r in rolls_live:
    try:
        rd = pd.Timestamp(r["date"]).tz_localize(IST)
        if rd in eq_live.index:
            ax_eq.axvline(rd, color=ORG, lw=0.8, alpha=0.45, ls="--")
    except: pass

# Mark trades
for t in trades_live:
    mask = eq_live.index.astype(str).str.startswith(t["xdate"])
    if mask.any():
        col = GRN if t["net"]>=0 else RED
        mk  = "^" if t["side"]=="LONG" else "v"
        ax_eq.scatter(eq_live.index[mask][0], eq_live[mask].iloc[0],
                      color=col, marker=mk, s=25, zorder=6, alpha=0.75,
                      linewidths=0)

fc2 = GRN if tot_ret>=0 else RED
ax_eq.text(0.99, 0.96, f"{tot_ret:+.2f}%", transform=ax_eq.transAxes,
           fontsize=18, fontweight="bold", color=fc2, ha="right", va="top",
           bbox=dict(boxstyle="round,pad=0.35",fc=BG,ec=fc2,lw=1.4,alpha=0.88))
ax_eq.set_ylabel("Portfolio Value (INR)", color=TXT, fontsize=9)
ax_eq.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x,_: f"Rs{x/1e5:.1f}L"))
ax_eq.grid(color=GRID, lw=0.45); ax_eq.tick_params(labelbottom=False)
ax_eq.set_xlim(ei[0], ei[-1])
ax_eq.legend(
    [plt.Line2D([0],[0],color=ORG,lw=1,ls="--"),
     plt.scatter([],[], marker="^", color=GRN, s=20),
     plt.scatter([],[], marker="v", color=RED, s=20)],
    ["Rollover", "Long exit", "Short exit"],
    fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=DIM,
    loc="upper left"
)

# Drawdown
dv = dd_live.values
ax_dd.fill_between(dd_live.index, 0, dv, color=RED, alpha=0.50)
ax_dd.plot(dd_live.index, dv, lw=0.85, color="#ff7070")
ax_dd.axhline(0, color=GRID, lw=0.8)
mi = dd_live.idxmin()
ax_dd.annotate(f"{max_dd:.1f}%", xy=(mi, dd_live.min()),
               xytext=(18,-14), textcoords="offset points",
               fontsize=8, color=RED, fontweight="bold",
               arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
ax_dd.set_ylabel("Drawdown", color=TXT, fontsize=8)
ax_dd.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x,_: f"{x:.0f}%"))
ax_dd.set_ylim(dv.min()*1.4, 0.5)
ax_dd.grid(color=GRID, lw=0.4); ax_dd.tick_params(labelbottom=False)

# Margin account balance
mv = margin_live.values
ax_mg.plot(margin_live.index, mv, lw=1.2, color=ORG, zorder=3)
ax_mg.fill_between(margin_live.index, 0, mv, color=ORG, alpha=0.15)
ax_mg.axhline(INITIAL_CAPITAL, color=WHT, lw=0.6, ls=":", alpha=0.3)
ax_mg.set_ylabel("Margin Acct (INR)", color=TXT, fontsize=8)
ax_mg.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x,_: f"Rs{x/1e5:.1f}L"))
ax_mg.grid(color=GRID, lw=0.3)
ax_mg.set_xlabel("Date", color=DIM, fontsize=8)

# Per-trade P&L bars
if trades_live:
    tnos  = [t["no"]  for t in trades_live]
    tnets = [t["net"] for t in trades_live]
    ax_bar.bar(tnos, tnets,
               color=[GRN if n>=0 else RED for n in tnets],
               alpha=0.80, width=0.7)
    ax_bar.axhline(0, color=GRID, lw=0.8)
ax_bar.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x,_: f"Rs{x/1e3:.0f}K"))
ax_bar.set_xlabel("Trade #", color=DIM, fontsize=8)
ax_bar.set_ylabel("Net P&L", color=TXT, fontsize=8)
ax_bar.set_title("Per-Trade P&L (Live)", color=TXT, fontsize=9, fontweight="bold")
ax_bar.grid(color=GRID, lw=0.35)

# KPI tiles
ax_kpi.axis("off")
kpis = [
    ("Return 1yr",     f"{tot_ret:+.2f}%",                  fc2),
    ("Win Rate",       f"{wr_l:.1f}%",                       GRN if wr_l>50 else YLW),
    ("Profit Factor",  f"{pf_l:.2f}",                        GRN if pf_l>1.5 else YLW),
    ("Sharpe",         f"{sh_l:.2f}",                        GRN if sh_l>1 else YLW),
    ("Max DD",         f"{max_dd:.1f}%",                     R if max_dd<-20 else YLW),
    ("Trades",         f"{len(pnls_live)}",                  TXT),
    ("Rolls",          f"{len(rolls_live)}",                 ORG),
    ("Roll Cost",      f"Rs{total_roll_cost/1e3:.1f}K",      YLW),
    ("Margin Calls",   f"{margin_calls}",                    R if margin_calls>0 else GRN),
    ("Expectancy",     f"Rs{exp_l/1e3:+.1f}K",              GRN if exp_l>0 else RED),
]
n = len(kpis); sp = 1.0/n
for k,(lbl,val,col) in enumerate(kpis):
    cx=k*sp+sp*0.04; cw=sp*0.92
    rect = plt.Rectangle((cx,0.06),cw,0.88,facecolor=GRID,
                          edgecolor=col,lw=0.9,alpha=0.65,
                          transform=ax_kpi.transAxes,clip_on=False)
    ax_kpi.add_patch(rect)
    ax_kpi.text(cx+cw/2,0.70,lbl,transform=ax_kpi.transAxes,
                ha="center",fontsize=6.2,color=DIM)
    ax_kpi.text(cx+cw/2,0.28,val,transform=ax_kpi.transAxes,
                ha="center",fontsize=9.0,color=col,fontweight="bold")

fig.text(0.07, 0.945,
         f"{STOCK_NAME}  ({SYMBOL})  FUTURES  ·  LIVE EXECUTION  (Year 8)",
         fontsize=15, fontweight="bold", color=WHT)
fig.text(0.07, 0.928,
         f"N_BAR={BEST_N} locked from 7-yr training  ·  "
         f"MTM daily settlement  ·  Monthly rollover  ·  "
         f"{df_exec.index[0].date()} → {df_exec.index[-1].date()}",
         fontsize=9, color=DIM)
fig.text(0.97, 0.01,
         "Hypothetical backtest — uses spot price as futures proxy. Not financial advice.",
         fontsize=7, color=DIM, ha="right", style="italic")

out1 = f"{OUT_DIR}/FUTURES_LIVE_{sym}_donchian{BEST_N}.png"
fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"[OK]  Chart saved  →  {out1}")
plt.close(fig)

print(f"\n[OK]  Complete.  {len(pnls_live)} trades  |  "
      f"{len(rolls_live)} rollovers  |  {margin_calls} margin calls\n")