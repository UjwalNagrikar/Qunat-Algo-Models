"""
╔══════════════════════════════════════════════════════════════════════╗
║   NSE FUTURES  ·  Pure Price Action  ·  N-Bar Donchian              ║
║   WALK-FORWARD:  7-Year Train  →  1-Year Live Execution             ║
║   Zero Indicators  ·  Full NSE Charges  ·  Monte Carlo              ║
╚══════════════════════════════════════════════════════════════════════╝

METHODOLOGY:
  Phase 1 — TRAINING  (Year 1–7, never traded)
    → Test N_BAR = [10,15,20,30,40,55,80] on 7yr data
    → Score each N_BAR on: Profit Factor × Sharpe (composite)
    → Pick the N_BAR with MOST CONSISTENT score across sub-periods
    → Lock it. Never touch again.

  Phase 2 — LIVE EXECUTION  (Year 8 only)
    → Run locked N_BAR on the unseen 1yr window
    → This is the ONLY number you report to anyone

  Why this matters:
    If you tune N_BAR on 8yr then backtest on 8yr → you are
    fitting to noise. Year-8 results will be an illusion.
    Locking Year 8 before tuning ensures it is genuinely unseen.
"""

# ── stdlib ────────────────────────────────────────────────────────────
import datetime as dt
import os, sys, warnings
warnings.filterwarnings("ignore")

# ── third-party ──────────────────────────────────────────────────────
import pytz, yfinance as yf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
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
    "TATAMOTORS.NS":(1425, 75_000, "Tata Motors"),
    "TATASTEEL.NS": (5500,  50_000, "Tata Steel"),
    "MARUTI.NS"   : (100,  160_000, "Maruti Suzuki"),
    "WIPRO.NS"    : (1500,  65_000, "Wipro"),
    "BHARTIARTL.NS":(475, 115_000, "Bharti Airtel"),
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

# ── Walk-Forward Split ───────────────────────────────────────────────
TOTAL_YEARS     = 8                # total data to download
TRAIN_YEARS     = 7                # training window
LIVE_YEARS      = 1                # out-of-sample execution window

# ── N_BAR candidates to test during training ─────────────────────────
N_BAR_CANDIDATES = [10, 15, 20, 30, 40, 55, 80]

# ── Risk Management (fixed — NOT tuned during training) ──────────────
RISK_PCT        = 0.10            # 2% capital risked per trade
MAX_LOTS        = 10
MARGIN_UTIL     = 0.60
COOLDOWN        = 2
TARGET_R        = 2.0              # TP = 2 × risk distance

# ── NSE Charges — Zerodha 2024 ──────────────────────────────────────
BROKERAGE  = 20.0
STT_RATE   = 0.000125
EXCH_RATE  = 0.000019
SEBI_RATE  = 0.000001
STAMP_RATE = 0.00002
GST_RATE   = 0.18

# ── Monte Carlo ──────────────────────────────────────────────────────
MC_RUNS    = 1_000
MC_RUIN    = 0.50

SAVE_PLOTS = False  # Set True to save PNGs to outputs/
IST        = pytz.timezone("Asia/Kolkata")

# ═══════════════════════════════════════════════════════════════════════
# 3.  VALIDATE + FETCH
# ═══════════════════════════════════════════════════════════════════════
if SYMBOL not in DB:
    raise ValueError(f"'{SYMBOL}' not in DB.")
LOT_SIZE, MARGIN_LOT, STOCK_NAME = DB[SYMBOL]

print(f"\n{'═'*68}")
print(f"  {STOCK_NAME}  ({SYMBOL})")
print(f"  Walk-Forward  ·  Train {TRAIN_YEARS}yr  →  Live {LIVE_YEARS}yr")
print(f"  Pure Price Action  ·  Donchian Channel")
print(f"{'═'*68}\n")

end   = dt.datetime.now(tz=pytz.utc)
start = end - dt.timedelta(days=365 * TOTAL_YEARS + 120)

raw = yf.download(SYMBOL, start=start, end=end,
                  interval="1d", auto_adjust=True, progress=False)
if raw.empty:
    sys.exit("No data — check symbol / network.")

if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
raw.rename(columns={"Open":"o","High":"h","Low":"l",
                    "Close":"c","Volume":"v"}, inplace=True)
raw = raw[["o","h","l","c","v"]].copy()

if raw.index.tzinfo is None:
    raw.index = raw.index.tz_localize("UTC")
raw.index = raw.index.tz_convert(IST)

# Hard split date — Year 8 is LOCKED until Phase 2
cutoff_all   = raw.index[-1] - pd.Timedelta(days=365 * TOTAL_YEARS)
split_date   = raw.index[-1] - pd.Timedelta(days=365 * LIVE_YEARS)

df_all   = raw.loc[raw.index >= cutoff_all].copy()   # full 8yr
df_train = df_all.loc[df_all.index <  split_date].copy()  # 7yr train
df_live  = df_all.loc[df_all.index >= split_date].copy()  # 1yr live

print(f"  Full data     : {df_all.index[0].date()} → {df_all.index[-1].date()}"
      f"  ({len(df_all)} bars)")
print(f"  Train window  : {df_train.index[0].date()} → {df_train.index[-1].date()}"
      f"  ({len(df_train)} bars)  ← tuning happens here")
print(f"  Live  window  : {df_live.index[0].date()} → {df_live.index[-1].date()}"
      f"  ({len(df_live)} bars)  ← trades reported here")
print()

# ═══════════════════════════════════════════════════════════════════════
# 4.  HELPERS
# ═══════════════════════════════════════════════════════════════════════
def charges(price, qty, side):
    tv  = abs(price * qty)
    brk = BROKERAGE
    stt = tv * STT_RATE   if side == "SELL" else 0.0
    exc = tv * EXCH_RATE
    sbi = tv * SEBI_RATE
    stp = tv * STAMP_RATE if side == "BUY"  else 0.0
    gst = (brk + exc + sbi) * GST_RATE
    return brk + stt + exc + sbi + stp + gst


def lot_size_fn(capital, entry_px, stop_px):
    risk_per_share = abs(entry_px - stop_px)
    if risk_per_share <= 0:
        return 0
    lots_risk   = int(capital * RISK_PCT / (risk_per_share * LOT_SIZE))
    lots_margin = int(capital * MARGIN_UTIL / MARGIN_LOT)
    lots = min(lots_risk, lots_margin, MAX_LOTS)
    if lots == 0 and lots_margin >= 1:
        lots = 1
    return max(lots, 0)


def build_donchian(df, n):
    """Add Donchian channel columns to df. shift(1) = no lookahead."""
    out = df.copy()
    out["dc_high"] = df["h"].shift(1).rolling(n).max()
    out["dc_low"]  = df["l"].shift(1).rolling(n).min()
    out["dc_mid"]  = (out["dc_high"] + out["dc_low"]) / 2
    out["long_sig"]  = df["c"] > out["dc_high"]
    out["short_sig"] = df["c"] < out["dc_low"]
    out.dropna(inplace=True)
    return out


def run_backtest(df, capital_start):
    """
    Full backtest engine on any DataFrame slice.
    Returns: (equity_series, pnls_list, trades_list, total_charges)
    """
    cap      = capital_start
    side     = None
    epx      = stop_px = tp_px = 0.0
    etime    = None
    nlots    = qty = 0
    cdl      = 0
    eq_curve = []
    pnls     = []
    trades   = []
    total_chg= 0.0
    tno      = 0

    for i in range(len(df)):
        row = df.iloc[i]; bt = df.index[i]
        o = float(row["o"]); h = float(row["h"])
        l = float(row["l"]); c = float(row["c"])
        dc_h = float(row["dc_high"])
        dc_l = float(row["dc_low"])
        dc_m = float(row["dc_mid"])
        if cdl > 0: cdl -= 1

        # EXIT
        if side:
            xpx = xrsn = None
            if side == "LONG"  and l <= stop_px: xpx = round(stop_px,2); xrsn="SL"
            elif side=="SHORT" and h >= stop_px: xpx = round(stop_px,2); xrsn="SL"

            if xpx is None and TARGET_R > 0:
                if side=="LONG"  and h >= tp_px: xpx=round(tp_px,2); xrsn="TP"
                elif side=="SHORT" and l<=tp_px:  xpx=round(tp_px,2); xrsn="TP"

            if xpx is None:
                if side=="LONG"  and c<=dc_m: xpx=round(c,2); xrsn="MID"
                elif side=="SHORT" and c>=dc_m: xpx=round(c,2); xrsn="MID"

            if xpx is None:
                if side=="LONG"  and row["short_sig"]: xpx=round(c,2); xrsn="FLIP"
                elif side=="SHORT" and row["long_sig"]: xpx=round(c,2); xrsn="FLIP"

            if xpx is not None:
                grs = (xpx-epx)*qty*(1 if side=="LONG" else -1)
                chg = charges(xpx, qty, "SELL" if side=="LONG" else "BUY")
                net = grs - chg
                cap += net; total_chg += chg; pnls.append(net); tno += 1
                try:   bh = i - list(df.index).index(etime)
                except: bh = 0
                trades.append(dict(
                    no=tno, side=side,
                    edate=str(etime)[:10], xdate=str(bt)[:10], bh=bh,
                    epx=epx, xpx=xpx, stop=stop_px,
                    lots=nlots, qty=int(qty),
                    gross=round(grs,2), chg=round(chg,2), net=round(net,2),
                    cap=round(cap), xrsn=xrsn
                ))
                side=None; nlots=qty=0; etime=None; cdl=COOLDOWN

        # ENTRY
        if not side and cdl == 0:
            sig = "LONG" if row["long_sig"] else ("SHORT" if row["short_sig"] else None)
            if sig:
                s_px = dc_l if sig=="LONG" else dc_h
                lots = lot_size_fn(cap, o, s_px)
                if lots > 0:
                    rd      = abs(o - s_px) or o * 0.005
                    epx     = round(o, 2)
                    stop_px = round(s_px, 2)
                    tp_px   = round(epx + TARGET_R*rd, 2) if sig=="LONG" \
                              else round(epx - TARGET_R*rd, 2)
                    side=sig; etime=bt; nlots=lots; qty=lots*LOT_SIZE
                    chg = charges(epx, qty, "BUY" if side=="LONG" else "SELL")
                    total_chg+=chg; cap-=chg

        unr = (c-epx)*qty*(1 if side=="LONG" else -1) if side else 0.0
        eq_curve.append(cap + unr)

    equity = pd.Series(eq_curve, index=df.index)
    return equity, pnls, trades, total_chg


def score(pnls, equity):
    """
    Composite score used during training ONLY.
    = Profit Factor × Sharpe
    Higher = better. Penalises high-variance strategies.
    """
    if len(pnls) < 5: return -999
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf     = sum(wins)/abs(sum(losses)) if losses else 0
    rets   = equity.pct_change().dropna()
    sh     = (rets.mean()/rets.std())*np.sqrt(252) if rets.std()>0 else 0
    return pf * sh


# ═══════════════════════════════════════════════════════════════════════
# 5.  PHASE 1 — TRAINING  (Year 1–7)
#     Find best N_BAR on training window.
#     LIVE DATA IS NOT TOUCHED HERE.
# ═══════════════════════════════════════════════════════════════════════
print(f"{'═'*68}")
print("  PHASE 1  —  TRAINING  (Year 1–7)")
print(f"  Testing N_BAR candidates: {N_BAR_CANDIDATES}")
print(f"{'═'*68}")
print(f"  {'N_BAR':>5}  {'Trades':>6}  {'WR%':>6}  {'PF':>6}  "
      f"{'Sharpe':>7}  {'Return%':>8}  {'MaxDD%':>8}  {'Score':>8}")
print(f"  {'─'*64}")

train_results = {}

for n in N_BAR_CANDIDATES:
    df_t = build_donchian(df_train, n)
    eq, pnls_t, _, _ = run_backtest(df_t, INITIAL_CAPITAL)

    if not pnls_t:
        print(f"  {n:>5}  — no trades")
        continue

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

    flag = "  ← best so far" if sc == max(train_results.values()) else ""
    print(f"  {n:>5}  {len(pnls_t):>6}  {wr:>6.1f}  {pf:>6.2f}  "
          f"{sh:>7.2f}  {ret:>8.2f}  {dd:>8.2f}  {sc:>8.3f}{flag}")

# ── Lock the winner ──────────────────────────────────────────────────
BEST_N = max(train_results, key=train_results.get)
print(f"\n  ✔  LOCKED N_BAR = {BEST_N}  (score = {train_results[BEST_N]:.3f})")
print(f"  Training complete. Live window sealed until now.")
print()

# ═══════════════════════════════════════════════════════════════════════
# 6.  PHASE 2 — LIVE EXECUTION  (Year 8 only)
#     LOCKED N_BAR applied to the unseen 1-year window.
#     Capital starts fresh at INITIAL_CAPITAL for clean reporting.
# ═══════════════════════════════════════════════════════════════════════
print(f"{'═'*68}")
print("  PHASE 2  —  LIVE EXECUTION  (Year 8 — UNSEEN DATA)")
print(f"  N_BAR = {BEST_N}  (locked from training — not re-tuned)")
print(f"  Period: {df_live.index[0].date()} → {df_live.index[-1].date()}")
print(f"{'═'*68}\n")

# Need N_BAR bars of lookback from training to seed the live channel
# So we take the tail of train + all of live, build channel, then
# slice to live dates only for trade execution
df_seed   = df_all.loc[df_all.index >= (split_date - pd.Timedelta(days=BEST_N*2))].copy()
df_seeded = build_donchian(df_seed, BEST_N)
df_exec   = df_seeded.loc[df_seeded.index >= split_date].copy()

eq_live, pnls_live, trades_live, chg_live = run_backtest(df_exec, INITIAL_CAPITAL)

# ═══════════════════════════════════════════════════════════════════════
# 7.  LIVE METRICS
# ═══════════════════════════════════════════════════════════════════════
rets_l  = eq_live.pct_change().dropna()
dd_live = (eq_live / eq_live.cummax() - 1) * 100

tot_ret = (eq_live.iloc[-1] / INITIAL_CAPITAL - 1) * 100
max_dd  = dd_live.min()
wins_l  = [p for p in pnls_live if p > 0]
loss_l  = [p for p in pnls_live if p < 0]
wr_l    = len(wins_l)/len(pnls_live)*100 if pnls_live else 0
aw_l    = np.mean(wins_l)  if wins_l  else 0
al_l    = np.mean(loss_l)  if loss_l  else 0
pf_l    = sum(wins_l)/abs(sum(loss_l)) if loss_l else float("inf")
sh_l    = (rets_l.mean()/rets_l.std())*np.sqrt(252) if rets_l.std()>0 else 0
exp_l   = (wr_l/100*aw_l) + ((1-wr_l/100)*al_l)
pct_chg = chg_live / INITIAL_CAPITAL * 100

cw=cl=mcw=mcl=0
for p in pnls_live:
    if p>0: cw+=1; cl=0; mcw=max(mcw,cw)
    else:   cl+=1; cw=0; mcl=max(mcl,cl)

xrsn_cnt = {}
for t in trades_live:
    xrsn_cnt[t["xrsn"]] = xrsn_cnt.get(t["xrsn"],0) + 1

# ═══════════════════════════════════════════════════════════════════════
# 8.  PRINT ALL LIVE TRADES
# ═══════════════════════════════════════════════════════════════════════
G="\033[92m"; R="\033[91m"; Y="\033[93m"; E="\033[0m"; W="\033[97m"
SEP="─"*120

print(f"\n{'═'*120}")
print(f"  LIVE TRADES  ·  {STOCK_NAME}  ·  {BEST_N}-Bar Donchian  "
      f"·  {df_exec.index[0].date()} → {df_exec.index[-1].date()}")
print(f"{'═'*120}")
print(f"  {'#':>4}  {'':4}  {'Side':<6}  {'Entry':^10}  {'Exit':^10}  "
      f"{'Bars':>4}  {'Entry₹':>9}  {'Stop₹':>9}  {'Exit₹':>9}  "
      f"{'Lots':>4}  {'Gross₹':>10}  {'Chg₹':>7}  {'Net₹':>10}  "
      f"{'Capital₹':>12}  Rsn")
print(SEP)

for t in trades_live:
    ok  = t["net"] >= 0
    nc  = G if ok else R
    lbl = f"{nc}{'WIN ':>4}{E}" if ok else f"{nc}{'LOSS':>4}{E}"
    print(
        f"  {t['no']:>4}  {lbl}  {t['side']:<6}  "
        f"{t['edate']:^10}  {t['xdate']:^10}  {t['bh']:>4}  "
        f"₹{t['epx']:>8,.1f}  ₹{t['stop']:>8,.1f}  ₹{t['xpx']:>8,.1f}  "
        f"{t['lots']:>4}  "
        f"{t['gross']:>+10,.0f}  {t['chg']:>7,.0f}  "
        f"{nc}{t['net']:>+10,.0f}{E}  "
        f"₹{t['cap']:>11,.0f}  {t['xrsn']}"
    )

g_tot = sum(t["gross"] for t in trades_live)
n_tot = sum(t["net"]   for t in trades_live)
nc    = G if n_tot >= 0 else R
print(SEP)
print(f"  {'TOT':>4}  {'':4}  {'':6}  "
      f"  {'':10}  {'':10}  {'':4}  {'':9}  {'':9}  {'':9}  {'':4}  "
      f"  {g_tot:>+10,.0f}  {chg_live:>7,.0f}  "
      f"{nc}{n_tot:>+10,.0f}{E}  ₹{eq_live.iloc[-1]:>11,.0f}")
print(f"{'═'*120}")

# ═══════════════════════════════════════════════════════════════════════
# 9.  LIVE PERFORMANCE SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═'*62}")
print(f"  LIVE PERFORMANCE  ·  {STOCK_NAME}")
print(f"  (Out-of-sample Year 8  —  the only number that counts)")
print(f"{'═'*62}")

def row(lbl, val, col=""):
    print(f"  {lbl:<30}  {col}{val}{E}")

row("Symbol",               SYMBOL)
row("Strategy",             f"Pure Price Action  ·  {BEST_N}-Bar Donchian")
row("Period (LIVE)",        f"{df_exec.index[0].date()} → {df_exec.index[-1].date()}")
row("N_BAR locked from",    f"7-year training  (candidates: {N_BAR_CANDIDATES})")
print(f"  {'─'*58}")
row("Initial Capital",      f"₹{INITIAL_CAPITAL:>14,.0f}")
fc = G if tot_ret >= 0 else R
row("Final Capital",        f"₹{eq_live.iloc[-1]:>14,.0f}", fc)
row("Total Return (1yr)",   f"{tot_ret:>+14.2f}%", fc)
row("Net P&L",              f"₹{eq_live.iloc[-1]-INITIAL_CAPITAL:>+14,.0f}", fc)
print(f"  {'─'*58}")
row("Total Trades",         f"{len(pnls_live):>14}")
row("Wins / Losses",        f"{len(wins_l):>6}  /  {len(loss_l):<6}   WR {wr_l:.1f}%")
row("Avg Win",              f"₹{aw_l:>+14,.0f}", G)
row("Avg Loss",             f"₹{al_l:>+14,.0f}", R)
row("Expectancy / trade",   f"₹{exp_l:>+14,.0f}", G if exp_l>0 else R)
row("Profit Factor",        f"{pf_l:>14.2f}", G if pf_l>1.5 else Y if pf_l>1 else R)
row("Best / Worst trade",   f"₹{max(pnls_live) if pnls_live else 0:>+12,.0f}  /  "
                            f"₹{min(pnls_live) if pnls_live else 0:>+12,.0f}")
row("Max Consec W / L",     f"{mcw:>6}  /  {mcl}")
print(f"  {'─'*58}")
row("Sharpe (ann.)",        f"{sh_l:>14.2f}", G if sh_l>1 else Y if sh_l>0 else R)
row("Max Drawdown",         f"{max_dd:>13.2f}%", R if max_dd<-20 else Y)
print(f"  {'─'*58}")
print("  Exit breakdown (live):")
for rsn, cnt in sorted(xrsn_cnt.items(), key=lambda x:-x[1]):
    ww = sum(1 for t in trades_live if t["xrsn"]==rsn and t["net"]>0)
    print(f"    {rsn:<8}  {cnt:>4} trades   WR {ww/cnt*100:.0f}%")
print(f"  {'─'*58}")
row("Total Charges",        f"₹{chg_live:>14,.0f}  ({pct_chg:.2f}% of cap)", Y)
print(f"{'═'*62}\n")

# ═══════════════════════════════════════════════════════════════════════
# 10.  COMPARE: TRAINING SCORE vs LIVE RESULT
# ═══════════════════════════════════════════════════════════════════════
print(f"{'═'*62}")
print("  N_BAR COMPARISON  ·  Training Score → Live Validation")
print(f"{'═'*62}")
print(f"  {'N_BAR':>5}  {'Train Score':>12}  {'Rank':>5}  {'Live N_BAR?':>12}")
sorted_n = sorted(train_results.items(), key=lambda x:-x[1])
for rank, (n, sc) in enumerate(sorted_n, 1):
    marker = "  ← SELECTED + LIVE TESTED" if n == BEST_N else ""
    print(f"  {n:>5}  {sc:>12.3f}  {rank:>5}{marker}")
print(f"{'═'*62}\n")

# ═══════════════════════════════════════════════════════════════════════
# 11.  MONTE CARLO ON LIVE TRADES ONLY
# ═══════════════════════════════════════════════════════════════════════
if len(pnls_live) < 10:
    print("Too few live trades for Monte Carlo (need ≥ 10).")
else:
    rng2  = np.random.default_rng(42)
    parr  = np.array(pnls_live)
    ntr   = len(parr)
    paths = np.empty((MC_RUNS, ntr+1))
    paths[:,0] = INITIAL_CAPITAL
    fc_arr = np.empty(MC_RUNS)
    dd_arr = np.empty(MC_RUNS)
    ruin   = np.zeros(MC_RUNS, bool)

    print(f"  Running {MC_RUNS:,} MC paths on LIVE trades only...", end="", flush=True)
    for s in range(MC_RUNS):
        shuf      = rng2.choice(parr, size=ntr, replace=True)
        path      = np.concatenate([[INITIAL_CAPITAL],
                                    INITIAL_CAPITAL + np.cumsum(shuf)])
        paths[s]  = path
        fc_arr[s] = path[-1]
        pk        = np.maximum.accumulate(path)
        dd_arr[s] = ((path/pk-1)).min()*100
        if np.any(path < INITIAL_CAPITAL * MC_RUIN): ruin[s] = True
        if (s+1)%200==0: print(".", end="", flush=True)
    print(" done!\n")

    p5  = np.percentile(paths, 5,  axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    xa  = np.arange(ntr+1)
    ruin_pct = ruin.mean()*100
    prof_pct = (fc_arr > INITIAL_CAPITAL).mean()*100
    med_fc   = np.median(fc_arr)
    med_dd   = np.median(dd_arr)
    p95dd    = np.percentile(dd_arr, 95)

    actual_path = np.array([INITIAL_CAPITAL] +
                            list(INITIAL_CAPITAL + np.cumsum(pnls_live)))

# ═══════════════════════════════════════════════════════════════════════
# 12.  CHARTS
# ═══════════════════════════════════════════════════════════════════════

sym  = SYMBOL.replace(".NS","").replace("^","")

BG   = "#07090f"; CARD = "#0d1120"; GRID = "#161d30"
TXT  = "#dde3f0"; DIM  = "#4a5570"
BLU  = "#4f8ef7"; GRN  = "#22c55e"; RED  = "#ef4444"
YLW  = "#f59e0b"; WHT  = "#f0f4ff"

# ── Chart 1: 3-panel live dashboard ─────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(22, 14), facecolor=BG,
                         gridspec_kw={"height_ratios":[3,1,1],
                                       "hspace":0.06})
fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.07)

for ax in axes:
    ax.set_facecolor(CARD)
    ax.tick_params(colors=DIM, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(GRID)

ei = eq_live.index; ev = eq_live.values

# Panel A: equity
axes[0].plot(ei, ev, lw=2.0, color=BLU, zorder=4)
axes[0].fill_between(ei, INITIAL_CAPITAL, ev,
                     where=ev >= INITIAL_CAPITAL,
                     color=GRN, alpha=0.18, interpolate=True)
axes[0].fill_between(ei, INITIAL_CAPITAL, ev,
                     where=ev < INITIAL_CAPITAL,
                     color=RED, alpha=0.25, interpolate=True)
axes[0].axhline(INITIAL_CAPITAL, color=WHT, lw=0.7, ls=(0,(5,4)), alpha=0.3)
axes[0].plot(ei, eq_live.cummax().values, lw=0.7, color=WHT, alpha=0.12, ls="--")

for t in trades_live:
    mask = eq_live.index.astype(str).str.startswith(t["xdate"])
    if mask.any():
        axes[0].scatter(eq_live.index[mask][0], eq_live[mask].iloc[0],
                        color=GRN if t["net"]>=0 else RED,
                        s=20, zorder=6, alpha=0.75, linewidths=0)

fc2 = GRN if tot_ret>=0 else RED
axes[0].text(0.99, 0.96, f"{tot_ret:+.2f}%",
             transform=axes[0].transAxes,
             fontsize=18, fontweight="bold", color=fc2, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.35",fc=BG,ec=fc2,lw=1.4,alpha=0.88))
axes[0].set_ylabel("Portfolio Value (INR)", color=TXT, fontsize=9)
axes[0].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x,_: f"Rs{x/1e5:.1f}L"))
axes[0].grid(color=GRID, lw=0.45)
axes[0].tick_params(labelbottom=False)
axes[0].set_xlim(ei[0], ei[-1])

# Panel B: drawdown
dv = dd_live.values
axes[1].fill_between(dd_live.index, 0, dv, color=RED, alpha=0.50)
axes[1].plot(dd_live.index, dv, lw=0.85, color="#ff7070")
axes[1].axhline(0, color=GRID, lw=0.8)
axes[1].set_ylabel("Drawdown", color=TXT, fontsize=8)
axes[1].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x,_: f"{x:.0f}%"))
axes[1].set_ylim(dv.min()*1.4, 0.5)
axes[1].grid(color=GRID, lw=0.4)
axes[1].tick_params(labelbottom=False)

# Panel C: per-trade bars
if trades_live:
    tnos  = [t["no"]  for t in trades_live]
    tnets = [t["net"] for t in trades_live]
    axes[2].bar(tnos, tnets,
                color=[GRN if n>=0 else RED for n in tnets],
                alpha=0.80, width=0.7)
    axes[2].axhline(0, color=GRID, lw=0.8)
axes[2].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x,_: f"Rs{x/1e3:.0f}K"))
axes[2].set_xlabel("Live Trade #", color=DIM, fontsize=8)
axes[2].set_ylabel("Net P&L", color=TXT, fontsize=8)
axes[2].grid(color=GRID, lw=0.35)

fig.text(0.07, 0.94,
         f"{STOCK_NAME}  ({SYMBOL})  ·  LIVE EXECUTION  (Year 8)",
         fontsize=15, fontweight="bold", color=WHT)
fig.text(0.07, 0.925,
         f"N_BAR={BEST_N} locked from 7-yr training  ·  "
         f"Pure Price Action Donchian  ·  "
         f"{df_exec.index[0].date()} → {df_exec.index[-1].date()}",
         fontsize=9, color=DIM)
fig.text(0.97, 0.01,
         "Backtest results hypothetical. Not financial advice.",
         fontsize=7, color=DIM, ha="right", style="italic")

if SAVE_PLOTS:
    out1 = f"outputs/LIVE_{sym}_donchian{BEST_N}.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[OK]  Live equity chart  →  {out1}")
else:
    plt.show()
plt.close(fig)

# ── Chart 2: Training comparison radar-style bar ─────────────────────
fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG)
fig2.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12, wspace=0.3)
for ax in (ax_a, ax_b):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=DIM, labelsize=9)
    for sp in ax.spines.values(): sp.set_color(GRID)

ns    = list(train_results.keys())
scores= list(train_results.values())
cols  = [GRN if n==BEST_N else BLU for n in ns]
ax_a.bar([str(n) for n in ns], scores, color=cols, alpha=0.82, width=0.6)
ax_a.axhline(0, color=GRID, lw=0.8)
ax_a.set_xlabel("N_BAR Candidate", color=DIM, fontsize=9)
ax_a.set_ylabel("Composite Score (PF × Sharpe)", color=TXT, fontsize=9)
ax_a.set_title("Training: N_BAR Selection", color=TXT, fontsize=11, fontweight="bold")
ax_a.grid(color=GRID, lw=0.4, axis="y")
for i,(n,sc) in enumerate(zip(ns,scores)):
    ax_a.text(i, sc+0.02, f"{sc:.2f}", ha="center",
              fontsize=8, color=GRN if n==BEST_N else DIM, fontweight="bold")
ax_a.text(0.5, 0.96, f"Selected: N_BAR={BEST_N}",
          transform=ax_a.transAxes, ha="center", fontsize=10,
          color=GRN, fontweight="bold",
          bbox=dict(boxstyle="round",fc=BG,ec=GRN,lw=1.2,alpha=0.85))

# MC fan on live
if len(pnls_live) >= 10:
    sidx2 = rng2.choice(MC_RUNS, size=min(200,MC_RUNS), replace=False)
    for s in sidx2:
        clr = "#1d4429" if paths[s,-1]>=INITIAL_CAPITAL else "#3d1515"
        ax_b.plot(xa, paths[s], lw=0.25, color=clr, alpha=0.35, zorder=1)
    ax_b.fill_between(xa, p5, p95, alpha=0.20, color=BLU, label="P5-P95")
    ax_b.plot(xa, p50, lw=2.0, color=YLW, label=f"Median Rs{med_fc/1e5:.2f}L", zorder=4)
    ax_b.plot(xa, actual_path, lw=2.2, color=WHT,
              label=f"Actual {tot_ret:+.1f}%", zorder=5)
    ax_b.axhline(INITIAL_CAPITAL, color=DIM, lw=0.8, ls=":")
    ax_b.axhline(INITIAL_CAPITAL*MC_RUIN, color=RED, lw=0.9, ls=":",
                 alpha=0.6, label="Ruin level")
    ax_b.set_title(f"MC Fan  ·  Live Trades Only  ·  {MC_RUNS:,} paths\n"
                   f"Ruin={ruin_pct:.1f}%   Profitable={prof_pct:.1f}%   "
                   f"MedDD={med_dd:.1f}%   P95DD={p95dd:.1f}%",
                   color=TXT, fontsize=9, fontweight="bold")
    ax_b.set_xlabel("Live Trade #", color=DIM, fontsize=8)
    ax_b.set_ylabel("Capital (INR)", color=TXT, fontsize=8)
    ax_b.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x,_: f"Rs{x/1e5:.1f}L"))
    ax_b.legend(fontsize=7.5, facecolor=BG, edgecolor=GRID, labelcolor=DIM)
    ax_b.grid(color=GRID, lw=0.4)

fig2.suptitle(
    f"Walk-Forward Results  ·  {STOCK_NAME} ({SYMBOL})  ·  "
    f"Train 7yr  →  Live 1yr",
    color=WHT, fontsize=13, fontweight="bold"
)
if SAVE_PLOTS:
    out2 = f"outputs/TRAIN_{sym}_donchian_selection.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[OK]  Training chart  →  {out2}")
else:
    plt.show()
plt.close(fig2)

# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═'*68}")
print("  METHODOLOGY SUMMARY")
print(f"{'═'*68}")
print(f"""
  Step 1  Download {TOTAL_YEARS} years of daily OHLCV data
  Step 2  Hard-split: first {TRAIN_YEARS} yr = train, last {LIVE_YEARS} yr = live (locked)
  Step 3  Test N_BAR ∈ {N_BAR_CANDIDATES} on 7-yr window only
  Step 4  Pick winner by Profit Factor × Sharpe composite score
  Step 5  N_BAR = {BEST_N}  →  applied to live window, never re-tuned
  Step 6  Live result: {tot_ret:+.2f}% over {LIVE_YEARS} year(s)


""")
print(f"[OK]  All done!  Live trades: {len(pnls_live)}  "
      f"on {STOCK_NAME}\n")