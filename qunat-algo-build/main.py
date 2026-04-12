# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: IMPORTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
import yfinance as yf
from tabulate import tabulate
from datetime import datetime, timedelta

np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────────
TICKER          = "^NSEI"
INITIAL_CAPITAL = 1_000_000      # INR 10 lakhs
TC              = 0.0005         # 0.05% per side
SLIPPAGE        = 0.0002         # 0.02% per side
RISK_FREE       = 0.065          # 6.5% Indian 10Y G-Sec
KELLY_CAP       = 1.5            # max position multiplier
HURST_TREND     = 0.55
HURST_MR        = 0.45
VOL_FILTER_PCT  = 92             # skip top 8% extreme-vol days
SCORE_THRESH    = 1.8            # minimum |composite score| to trade
CONFIRM_DAYS    = 2              # signal must persist N days before entry
MAX_HOLD        = 8              # max days to hold a position
MIN_HOLD        = 3              # minimum days before considering exit
TRADING_DAYS    = 252

# ── Visual theme ─────────────────────────────────────────────────────────────
BG       = "#0D1117"
PANEL_BG = "#161B22"
BORDER   = "#30363D"
TEXT     = "#E6EDF3"
MUTED    = "#8B949E"
ACCENT   = "#58A6FF"      # blue
GREEN    = "#3FB950"
RED      = "#F85149"
GOLD     = "#D29922"
PURPLE   = "#BC8CFF"
ORANGE   = "#FFA657"
CYAN     = "#39D353"

PDF_FILE   = "nifty50_quant_report.pdf"
TRADES_CSV = "trades_log.csv"
EQUITY_CSV = "equity_curve.csv"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_nifty(start: datetime, end: datetime) -> pd.DataFrame:
    """Synthetic Nifty 50 OHLCV via GBM + AR(1), calibrated to 2014-2024 historical params."""
    print("[DATA] Generating calibrated synthetic Nifty 50 data ...")
    annual_mu = 0.13; annual_sigma = 0.175; ar1 = 0.08
    jump_prob = 0.015; jump_mu = -0.02; jump_sigma = 0.04; base = 8000.0
    bdays = pd.bdate_range(start=start, end=end); n = len(bdays)
    dt = 1 / TRADING_DAYS; sig_d = annual_sigma * np.sqrt(dt)
    eps = np.random.normal(0, 1, n)
    ar_eps = np.zeros(n); ar_eps[0] = eps[0]
    for i in range(1, n):
        ar_eps[i] = ar1 * ar_eps[i-1] + eps[i] * np.sqrt(1 - ar1**2)
    jumps = np.where(np.random.rand(n) < jump_prob,
                     np.random.normal(jump_mu, jump_sigma, n), 0.0)
    lr = annual_mu * dt - 0.5 * annual_sigma**2 * dt + sig_d * ar_eps + jumps
    closes = base * np.exp(np.cumsum(lr))
    iv = annual_sigma * np.sqrt(dt) * np.random.uniform(0.5, 2.0, n)
    opens  = closes * np.exp(np.random.normal(0, iv * 0.3, n))
    highs  = np.maximum(opens, closes) * np.exp(np.abs(np.random.normal(0, iv * 0.5, n)))
    lows   = np.minimum(opens, closes) * np.exp(-np.abs(np.random.normal(0, iv * 0.5, n)))
    vols   = (200_000_000 * np.random.lognormal(0, 0.5, n)).astype(int)
    df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows,
                        "Close": closes, "Volume": vols}, index=bdays)
    df["High"] = df[["Open","High","Close"]].max(axis=1)
    df["Low"]  = df[["Open","Low","Close"]].min(axis=1)
    return df.round(2)


def download_data() -> pd.DataFrame:
    """Download 10y Nifty 50 OHLCV; fall back to calibrated synthetic if unavailable."""
    end   = datetime.today()
    start = end - timedelta(days=365 * 10 + 5)
    df    = pd.DataFrame()
    print(f"[DATA] Attempting live download: {TICKER} ({start.date()} → {end.date()}) ...")
    try:
        raw = yf.download(TICKER, start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"), interval="1d", progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw[["Open","High","Low","Close","Volume"]].dropna()
        if len(raw) > 500:
            df = raw.copy(); print(f"[DATA] Live data: {len(df)} rows ✓")
        else:
            raise ValueError("Insufficient rows")
    except Exception as e:
        print(f"[DATA] Live download unavailable ({e}). Using synthetic data.")
        df = _generate_synthetic_nifty(start, end)
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df[df["Volume"] > 0].sort_index()
    print(f"[DATA] Shape: {df.shape} | Range: {df.index[0].date()} → {df.index[-1].date()}")
    print(df.head().to_string()); print()
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def hurst_dfa(series: np.ndarray) -> float:
    """Hurst exponent via Detrended Fluctuation Analysis — robust at short windows."""
    n = len(series)
    if n < 20: return 0.5
    try:
        x   = np.cumsum(series - series.mean())
        scales = np.unique(np.logspace(np.log10(4), np.log10(n//4), 10).astype(int))
        flucts = []
        for s in scales:
            n_seg = n // s
            if n_seg < 2: continue
            f2 = []
            for k in range(n_seg):
                seg = x[k*s:(k+1)*s]
                t   = np.arange(s)
                coef= np.polyfit(t, seg, 1)
                trend = np.polyval(coef, t)
                f2.append(np.mean((seg - trend)**2))
            flucts.append(np.sqrt(np.mean(f2)))
        if len(flucts) < 4: return 0.5
        log_s = np.log(scales[:len(flucts)])
        log_f = np.log(np.array(flucts))
        h, *_ = np.polyfit(log_s, log_f, 1)
        return float(np.clip(h, 0.05, 0.95))
    except Exception:
        return 0.5


def parkinson_vol(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson single-day high-low volatility estimator."""
    return np.sqrt((1 / (4 * np.log(2))) * (np.log(high / low))**2)


def variance_ratio_stat(log_ret: pd.Series, k: int = 5) -> pd.Series:
    """Lo-MacKinlay variance ratio: VR > 1 → momentum, VR < 1 → mean-reversion."""
    var1  = log_ret.rolling(k).var()
    var_k = log_ret.rolling(k).sum().rolling(k).var()
    return var_k / (var1 * k + 1e-12)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all non-lagging features using only data available at time T."""
    f = df.copy()
    O, H, L, C, V = f["Open"], f["High"], f["Low"], f["Close"], f["Volume"]
    pC = C.shift(1)

    # ── A. Price Action ──────────────────────────────────────────────────────
    f["daily_ret"]    = (C - O) / (O + 1e-9)
    f["intraday_rng"] = (H - L) / (O + 1e-9)
    f["gap"]          = (O - pC) / (pC + 1e-9)
    f["body_ratio"]   = np.abs(C - O) / (H - L + 1e-9)
    f["upper_wick"]   = (H - np.maximum(O, C)) / (H - L + 1e-9)
    f["lower_wick"]   = (np.minimum(O, C) - L) / (H - L + 1e-9)
    f["gap_dir"]      = np.sign(f["gap"])

    # ── B. Statistical / Volatility ──────────────────────────────────────────
    f["realized_vol"]   = f["log_ret"].rolling(10).std() * np.sqrt(TRADING_DAYS)
    vol_m, vol_s        = V.rolling(10).mean(), V.rolling(10).std()
    f["vol_zscore"]     = (V - vol_m) / (vol_s + 1e-9)
    f["parkinson_vol"]  = parkinson_vol(H, L)
    f["vr_stat"]        = variance_ratio_stat(f["log_ret"], k=5)

    # Multi-period returns (non-lagging: computed from available OHLC)
    f["ret_3d"]   = C / C.shift(3) - 1
    f["ret_5d"]   = C / C.shift(5) - 1
    f["ret_10d"]  = C / C.shift(10) - 1

    # ── C. Regime Detection ───────────────────────────────────────────────────
    roll_min = C.rolling(20).min()
    roll_max = C.rolling(20).max()
    f["range_z"]  = (C - roll_min) / (roll_max - roll_min + 1e-9)

    # Rolling Hurst via DFA — 60-day window
    print("[FEATURE] Computing rolling Hurst exponent (DFA, 60d) ...")
    lr_arr = f["log_ret"].values
    hurst_arr = np.full(len(f), 0.5)
    for i in range(60, len(f)):
        hurst_arr[i] = hurst_dfa(lr_arr[i-60:i])
    f["hurst"] = hurst_arr

    f.dropna(inplace=True)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(f: pd.DataFrame) -> pd.DataFrame:
    """
    Generate +1/-1/0 signals via composite scoring with:
    - Regime-filtered routing (trend vs mean-reversion)
    - Extreme-volatility filter
    - Score threshold gate (min |score| = SCORE_THRESH)
    - 2-day signal confirmation (reduces whipsaw)
    """
    s = f.copy()
    pvol_thresh = np.nanpercentile(s["parkinson_vol"].values, VOL_FILTER_PCT)

    # ── Trend score ──────────────────────────────────────────────────────────
    trend = pd.Series(0.0, index=s.index)
    trend += np.where(s["ret_5d"]  > 0.015,  1.5, np.where(s["ret_5d"]  < -0.015, -1.5, 0.0))
    trend += np.where(s["ret_3d"]  > 0.008,  1.0, np.where(s["ret_3d"]  < -0.008, -1.0, 0.0))
    trend += np.where(s["gap_dir"] > 0,       0.5, np.where(s["gap_dir"] < 0,      -0.5, 0.0))
    trend += np.where(s["body_ratio"] > 0.65, np.sign(s["daily_ret"]) * 0.5, 0.0)
    trend += np.where(s["vol_zscore"] > 1.5,  np.sign(s["daily_ret"]) * 0.5, 0.0)
    trend += np.where(s["vr_stat"]   > 1.1,   np.sign(s["ret_5d"]) * 0.5, 0.0)

    # ── Mean-reversion score ──────────────────────────────────────────────────
    mr = pd.Series(0.0, index=s.index)
    mr += np.where(s["range_z"]  < 0.15,  2.0, np.where(s["range_z"]  > 0.85, -2.0, 0.0))
    mr += np.where(s["range_z"]  < 0.25,  0.8, np.where(s["range_z"]  > 0.75, -0.8, 0.0))
    mr += np.where(s["lower_wick"] > 0.45,  0.8, 0.0)
    mr += np.where(s["upper_wick"] > 0.45, -0.8, 0.0)
    mr += np.where(s["ret_5d"]  < -0.02,  1.0, np.where(s["ret_5d"]  > 0.02, -1.0, 0.0))
    mr += np.where(s["vr_stat"] < 0.85,  np.sign(0.5 - s["range_z"]) * 0.7, 0.0)

    # ── Regime routing ────────────────────────────────────────────────────────
    composite = np.where(s["hurst"] > HURST_TREND, trend,
                np.where(s["hurst"] < HURST_MR,    mr, 0.0))

    # ── Filters ───────────────────────────────────────────────────────────────
    composite = np.where(s["parkinson_vol"] > pvol_thresh, 0.0, composite)  # vol filter
    raw_sig   = np.where(np.abs(composite) >= SCORE_THRESH, np.sign(composite), 0.0)

    # ── Signal confirmation: require same signal for CONFIRM_DAYS ─────────────
    raw_s = pd.Series(raw_sig, index=s.index)
    confirmed = raw_s.copy() * 0
    for i in range(CONFIRM_DAYS - 1, len(raw_s)):
        window = raw_s.iloc[i - CONFIRM_DAYS + 1: i + 1]
        if (window == 1.0).all():
            confirmed.iloc[i] = 1.0
        elif (window == -1.0).all():
            confirmed.iloc[i] = -1.0

    s["raw_score"]  = composite
    s["signal"]     = confirmed

    # ── Kelly + volatility sizing ─────────────────────────────────────────────
    trade_rets = s["log_ret"]
    w_mask     = trade_rets > 0
    avg_win    = trade_rets[w_mask].mean()  if w_mask.sum()  > 0 else 0.01
    avg_loss   = trade_rets[~w_mask].abs().mean() if (~w_mask).sum() > 0 else 0.01
    wr         = w_mask.mean()
    kelly_f    = float(np.clip(
        wr / (avg_loss + 1e-9) - (1 - wr) / (avg_win + 1e-9), 0.2, KELLY_CAP))

    rv_norm    = s["realized_vol"] / (s["realized_vol"].mean() + 1e-9)
    pos_size   = (kelly_f / (rv_norm + 1e-9)).clip(upper=KELLY_CAP)
    s["pos_size"] = pos_size * np.abs(confirmed)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: BACKTESTING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(s: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Event-driven backtest with multi-day positions.
    - Entry: T+1 Open after confirmed signal on T
    - Exit: signal reversal OR MAX_HOLD days reached (min MIN_HOLD days held)
    - No look-ahead bias: signal uses only day-T data
    """
    n      = len(s)
    opens  = s["Open"].values
    closes = s["Close"].values
    sigs   = s["signal"].values
    sizes  = s["pos_size"].values
    dates  = s.index

    portfolio = np.full(n, np.nan)
    benchmark = np.full(n, np.nan)
    daily_pnl = np.zeros(n)
    portfolio[0] = benchmark[0] = INITIAL_CAPITAL

    capital    = INITIAL_CAPITAL
    bm_capital = INITIAL_CAPITAL
    trade_log  = []

    in_trade     = False
    pos_dir      = 0
    entry_px     = 0.0
    entry_date   = None
    entry_capital= 0.0
    hold_count   = 0

    for i in range(1, n):
        # Benchmark: buy-and-hold daily return
        bm_ret      = (closes[i] - closes[i-1]) / (closes[i-1] + 1e-9)
        bm_capital *= (1 + bm_ret)
        benchmark[i] = bm_capital

        prev_sig = sigs[i-1]

        # ── Check for exit ────────────────────────────────────────────────────
        if in_trade:
            hold_count += 1
            exit_now = False
            if hold_count >= MIN_HOLD:
                if prev_sig == -pos_dir or prev_sig == 0 or hold_count >= MAX_HOLD:
                    exit_now = True

            if exit_now:
                exit_px   = opens[i]    # exit at next open
                cost      = (TC + SLIPPAGE) * 2
                raw_ret   = pos_dir * (exit_px - entry_px) / (entry_px + 1e-9)
                net_ret   = raw_ret - cost
                pnl_inr   = entry_capital * net_ret
                capital  += pnl_inr
                capital   = max(capital, 1)

                trade_log.append({
                    "entry_date"  : str(entry_date)[:10],
                    "exit_date"   : str(dates[i])[:10],
                    "signal"      : int(pos_dir),
                    "entry_price" : round(float(entry_px), 2),
                    "exit_price"  : round(float(exit_px), 2),
                    "pnl_pct"     : round(float(net_ret * 100), 4),
                    "pnl_inr"     : round(float(pnl_inr), 2),
                    "holding_days": hold_count,
                })
                daily_pnl[i] += pnl_inr
                in_trade = False; pos_dir = 0; hold_count = 0

        # ── Check for entry ───────────────────────────────────────────────────
        if not in_trade and prev_sig != 0:
            sz           = float(sizes[i-1])
            entry_capital= min(capital * sz, capital * KELLY_CAP)
            entry_capital= min(entry_capital, capital)
            entry_px     = opens[i]
            entry_date   = dates[i]
            pos_dir      = int(prev_sig)
            in_trade     = True
            hold_count   = 0
            # Entry cost
            cost_inr     = entry_capital * (TC + SLIPPAGE)
            capital     -= cost_inr
            capital      = max(capital, 1)

        portfolio[i] = capital

    # Fill any leading NaNs
    portfolio = pd.Series(portfolio, index=s.index).ffill().fillna(INITIAL_CAPITAL)
    benchmark = pd.Series(benchmark, index=s.index).ffill().fillna(INITIAL_CAPITAL)

    result              = s.copy()
    result["portfolio_value"]   = portfolio.values
    result["benchmark_value"]   = benchmark.values
    result["daily_pnl"]         = daily_pnl
    result["strategy_returns"]  = portfolio.pct_change().fillna(0)
    result["benchmark_returns"] = benchmark.pct_change().fillna(0)

    trades_df = pd.DataFrame(trade_log)
    return result, trades_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Drawdown from rolling peak at each timestep."""
    peak = np.maximum.accumulate(equity)
    return (equity - peak) / (peak + 1e-9)


def max_dd_duration(dd: np.ndarray) -> int:
    """Max consecutive days underwater."""
    m = c = 0
    for v in dd:
        c = c + 1 if v < 0 else 0
        m = max(m, c)
    return m


def omega_ratio(returns: np.ndarray, thr: float = 0.0) -> float:
    """Omega ratio vs threshold."""
    e = returns - thr
    g = e[e > 0].sum(); l = -e[e < 0].sum()
    return g / (l + 1e-9)


def sortino(returns: np.ndarray) -> float:
    """Annualized Sortino ratio."""
    rf_d   = RISK_FREE / TRADING_DAYS
    excess = returns - rf_d
    down   = returns[returns < rf_d]
    semi   = down.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(down) > 1 else 1e-9
    return excess.mean() * TRADING_DAYS / (semi + 1e-9)


def max_consec(wins: np.ndarray) -> tuple[int, int]:
    """(max_consecutive_wins, max_consecutive_losses)."""
    mw = ml = cw = cl = 0
    for w in wins:
        if w: cw += 1; cl = 0; mw = max(mw, cw)
        else: cl += 1; cw = 0; ml = max(ml, cl)
    return mw, ml


def compute_metrics(result: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """Compute full strategy vs benchmark performance metrics."""
    sr  = result["strategy_returns"].values
    br  = result["benchmark_returns"].values
    pv  = result["portfolio_value"].values
    bv  = result["benchmark_value"].values
    ny  = len(sr) / TRADING_DAYS

    def cagr(f, i, y):   return (f / i)**(1/y) - 1 if y > 0 else 0
    def ann_vol(r):       return r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    def sharpe(r):
        v = ann_vol(r)
        return (r.mean() - RISK_FREE/TRADING_DAYS) * TRADING_DAYS / (v + 1e-9)
    def var95(r):         return np.percentile(r, 5)
    def cvar95(r):
        v = var95(r); sub = r[r <= v]
        return sub.mean() if len(sub) > 0 else v

    dd_s = drawdown_series(pv)
    dd_b = drawdown_series(bv)

    res2 = result.copy(); res2.index = pd.to_datetime(res2.index)
    ann_s = res2["strategy_returns"].resample("YE").apply(lambda x: (1+x).prod()-1)
    ann_b = res2["benchmark_returns"].resample("YE").apply(lambda x: (1+x).prod()-1)

    m = {}
    for tag, ret, eq, dd, ann in [
        ("strategy",  sr, pv, dd_s, ann_s),
        ("benchmark", br, bv, dd_b, ann_b),
    ]:
        cg = cagr(eq[-1], eq[0], ny)
        m[tag] = {
            "total_return"   : (eq[-1]/eq[0]-1)*100,
            "cagr"           : cg*100,
            "best_year"      : ann.max()*100,
            "worst_year"     : ann.min()*100,
            "avg_ann_ret"    : ann.mean()*100,
            "ann_vol"        : ann_vol(ret)*100,
            "max_drawdown"   : dd.min()*100,
            "max_dd_dur"     : max_dd_duration(dd),
            "avg_drawdown"   : dd[dd<0].mean()*100 if (dd<0).sum()>0 else 0,
            "var95"          : var95(ret)*100,
            "cvar95"         : cvar95(ret)*100,
            "downside_dev"   : ret[ret<0].std(ddof=1)*np.sqrt(TRADING_DAYS)*100
                               if (ret<0).sum()>1 else 0,
            "sharpe"         : sharpe(ret),
            "sortino"        : sortino(ret),
            "calmar"         : cg / (abs(dd.min()) + 1e-9),
            "omega"          : omega_ratio(ret),
            "skewness"       : stats.skew(ret),
            "kurtosis"       : stats.kurtosis(ret),
            "hurst"          : hurst_dfa(ret[~np.isnan(ret)]),
            "autocorr"       : pd.Series(ret).autocorr(lag=1),
        }
    active = sr - br
    te     = active.std(ddof=1) * np.sqrt(TRADING_DAYS)
    m["strategy"]["info_ratio"]  = active.mean() * TRADING_DAYS / (te + 1e-9)
    m["benchmark"]["info_ratio"] = 0.0

    if len(trades) > 0:
        w = trades["pnl_pct"] > 0
        mw, ml = max_consec(w.values)
        m["trades"] = {
            "n_trades"    : len(trades),
            "win_rate"    : w.mean()*100,
            "avg_win"     : trades.loc[w,  "pnl_pct"].mean() if w.sum()>0 else 0,
            "avg_loss"    : trades.loc[~w, "pnl_pct"].mean() if (~w).sum()>0 else 0,
            "profit_factor": trades.loc[w, "pnl_pct"].sum() /
                             (-trades.loc[~w,"pnl_pct"].sum()+1e-9),
            "expectancy"  : trades["pnl_pct"].mean(),
            "avg_hold"    : trades["holding_days"].mean(),
            "max_w"       : mw, "max_l": ml,
        }
    else:
        m["trades"] = {k:0 for k in ["n_trades","win_rate","avg_win","avg_loss",
                                      "profit_factor","expectancy","avg_hold","max_w","max_l"]}
    return m


def print_metrics(m: dict) -> None:
    """Print formatted performance report."""
    s, b, t = m["strategy"], m["benchmark"], m["trades"]
    rows = [
        ["── RETURN METRICS ──", "", ""],
        ["Total Return (%)",          f"{s['total_return']:>8.2f}",   f"{b['total_return']:>8.2f}"],
        ["CAGR (%)",                  f"{s['cagr']:>8.2f}",          f"{b['cagr']:>8.2f}"],
        ["Best Year (%)",             f"{s['best_year']:>8.2f}",      f"{b['best_year']:>8.2f}"],
        ["Worst Year (%)",            f"{s['worst_year']:>8.2f}",     f"{b['worst_year']:>8.2f}"],
        ["Avg Annual Return (%)",     f"{s['avg_ann_ret']:>8.2f}",    f"{b['avg_ann_ret']:>8.2f}"],
        ["── RISK METRICS ──", "", ""],
        ["Annualized Volatility (%)", f"{s['ann_vol']:>8.2f}",        f"{b['ann_vol']:>8.2f}"],
        ["Max Drawdown (%)",          f"{s['max_drawdown']:>8.2f}",   f"{b['max_drawdown']:>8.2f}"],
        ["Max DD Duration (days)",    f"{s['max_dd_dur']:>8}",        f"{b['max_dd_dur']:>8}"],
        ["Avg Drawdown (%)",          f"{s['avg_drawdown']:>8.2f}",   f"{b['avg_drawdown']:>8.2f}"],
        ["VaR 95% (daily %)",        f"{s['var95']:>8.4f}",          f"{b['var95']:>8.4f}"],
        ["CVaR 95% (daily %)",       f"{s['cvar95']:>8.4f}",         f"{b['cvar95']:>8.4f}"],
        ["Downside Deviation (%)",    f"{s['downside_dev']:>8.4f}",   f"{b['downside_dev']:>8.4f}"],
        ["── RISK-ADJUSTED ──", "", ""],
        ["Sharpe Ratio",              f"{s['sharpe']:>8.4f}",         f"{b['sharpe']:>8.4f}"],
        ["Sortino Ratio",             f"{s['sortino']:>8.4f}",        f"{b['sortino']:>8.4f}"],
        ["Calmar Ratio",              f"{s['calmar']:>8.4f}",         f"{b['calmar']:>8.4f}"],
        ["Omega Ratio",               f"{s['omega']:>8.4f}",          f"{b['omega']:>8.4f}"],
        ["Information Ratio",         f"{s['info_ratio']:>8.4f}",     "     N/A"],
        ["── TRADE METRICS ──", "", ""],
        ["Total Trades",              f"{t['n_trades']:>8}",          "     N/A"],
        ["Win Rate (%)",              f"{t['win_rate']:>8.2f}",       "     N/A"],
        ["Avg Win (%)",               f"{t['avg_win']:>8.4f}",        "     N/A"],
        ["Avg Loss (%)",              f"{t['avg_loss']:>8.4f}",       "     N/A"],
        ["Profit Factor",             f"{t['profit_factor']:>8.4f}",  "     N/A"],
        ["Expectancy (% / trade)",    f"{t['expectancy']:>8.4f}",     "     N/A"],
        ["Avg Holding Period (days)", f"{t['avg_hold']:>8.2f}",       "     N/A"],
        ["Max Consecutive Wins",      f"{t['max_w']:>8}",             "     N/A"],
        ["Max Consecutive Losses",    f"{t['max_l']:>8}",             "     N/A"],
        ["── STATISTICAL ──", "", ""],
        ["Skewness",                  f"{s['skewness']:>8.4f}",       f"{b['skewness']:>8.4f}"],
        ["Kurtosis",                  f"{s['kurtosis']:>8.4f}",       f"{b['kurtosis']:>8.4f}"],
        ["Hurst Exponent",            f"{s['hurst']:>8.4f}",          f"{b['hurst']:>8.4f}"],
        ["Autocorrelation (lag-1)",   f"{s['autocorr']:>8.4f}",       f"{b['autocorr']:>8.4f}"],
    ]
    print("\n" + "="*66)
    print("  NIFTY 50 QUANT SYSTEM v2.0 — PERFORMANCE REPORT")
    print("="*66)
    print(tabulate(rows, headers=["Metric","Strategy","Benchmark"],
                   tablefmt="simple", colalign=("left","right","right")))
    print("="*66 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _setup_ax(ax, title="", xlabel="", ylabel=""):
    """Apply consistent dark professional styling to an axes."""
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER); spine.set_linewidth(0.8)
    ax.grid(True, color=BORDER, alpha=0.5, linewidth=0.5)
    if title:   ax.set_title(title,   color=GOLD,  fontsize=10, fontweight="bold", pad=8)
    if xlabel:  ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    if ylabel:  ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    return ax


def _new_fig(nrows=1, ncols=1, figsize=(14,7), title=""):
    """Create a new figure with dark background."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(BG)
    if title:
        fig.suptitle(title, color=TEXT, fontsize=12, fontweight="bold", y=0.98)
    return fig, axes


# ─── Plot 1: Equity Curve ────────────────────────────────────────────────────
def plot1_equity_curve(result: pd.DataFrame, pdf: PdfPages) -> None:
    """Equity curve vs benchmark with drawdown shading and rolling Sharpe subplot."""
    try:
        fig = plt.figure(figsize=(15, 9), facecolor=BG)
        gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.08,
                                height_ratios=[3, 1, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        for ax in [ax1,ax2,ax3]: ax.set_facecolor(PANEL_BG)
        fig.suptitle("EQUITY CURVE  ·  Strategy vs Nifty 50 Benchmark",
                     color=TEXT, fontsize=13, fontweight="bold", y=0.99)

        dates = result.index
        norm_s = result["portfolio_value"]  / result["portfolio_value"].iloc[0]  * 100
        norm_b = result["benchmark_value"]  / result["benchmark_value"].iloc[0]  * 100
        dd     = drawdown_series(result["portfolio_value"].values)

        ax1.plot(dates, norm_s, color=ACCENT,  lw=1.8, label="Strategy",  zorder=3)
        ax1.plot(dates, norm_b, color=ORANGE,  lw=1.4, label="Benchmark (Nifty 50)", zorder=2, alpha=0.85)

        # Shade significant drawdowns > 5%
        in_dd = False; dd_start = None
        for i, (d, v) in enumerate(zip(dates, dd)):
            if v < -0.05 and not in_dd:
                in_dd = True; dd_start = d
            elif v >= -0.01 and in_dd:
                ax1.axvspan(dd_start, d, color=RED, alpha=0.10, zorder=1)
                in_dd = False
        if in_dd: ax1.axvspan(dd_start, dates[-1], color=RED, alpha=0.10)

        ax1.axhline(100, color=BORDER, lw=0.7, ls="--", alpha=0.6)
        _setup_ax(ax1, ylabel="Indexed (base=100)")
        ax1.legend(loc="upper left", framealpha=0.15, facecolor=PANEL_BG,
                   edgecolor=BORDER, labelcolor=TEXT, fontsize=9)

        # Drawdown fill
        ax2.fill_between(dates, dd*100, 0, color=RED, alpha=0.5)
        ax2.plot(dates, dd*100, color=RED, lw=0.6)
        ax2.axhline(0, color=BORDER, lw=0.7)
        _setup_ax(ax2, ylabel="Drawdown %")

        # Rolling Sharpe (252d)
        rs = result["strategy_returns"].rolling(TRADING_DAYS).apply(
            lambda x: (x.mean()-RISK_FREE/TRADING_DAYS)/(x.std()+1e-9)*np.sqrt(TRADING_DAYS), raw=True)
        ax3.plot(dates, rs, color=CYAN, lw=1.2)
        ax3.axhline(1.0, color=GOLD, ls="--", lw=0.8, alpha=0.7)
        ax3.axhline(0.0, color=MUTED, ls="-",  lw=0.5, alpha=0.4)
        ax3.fill_between(dates, rs, 0, where=rs>0, color=CYAN, alpha=0.15)
        ax3.fill_between(dates, rs, 0, where=rs<0, color=RED,  alpha=0.15)
        _setup_ax(ax3, ylabel="252d Sharpe", xlabel="Date")

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax3.tick_params(axis="x", colors=TEXT, labelsize=8)
        plt.tight_layout(rect=[0,0,1,0.97])
        pdf.savefig(fig, facecolor=BG); plt.close(fig)
        print("[PLOT 1] Equity Curve ✓")
    except Exception as e: print(f"[WARNING] Plot 1 failed: {e}")


# ─── Plot 2: Drawdown ────────────────────────────────────────────────────────
def plot2_drawdown(result: pd.DataFrame, pdf: PdfPages) -> None:
    """Drawdown curve with top-5 annotated worst periods."""
    try:
        fig, ax = _new_fig(figsize=(15, 6), title="DRAWDOWN ANALYSIS  ·  Rolling Underwater Equity")
        _setup_ax(ax, ylabel="Drawdown (%)", xlabel="Date")

        dates = result.index
        dd    = drawdown_series(result["portfolio_value"].values)
        dd_s  = pd.Series(dd * 100, index=dates)

        ax.fill_between(dates, dd_s, 0, color=RED, alpha=0.35, label="Strategy Drawdown")
        ax.plot(dates, dd_s, color=RED, lw=0.8)

        # Benchmark drawdown overlay
        dd_b = drawdown_series(result["benchmark_value"].values) * 100
        ax.plot(dates, dd_b, color=ORANGE, lw=1.0, alpha=0.6, ls="--", label="Benchmark Drawdown")
        ax.axhline(0, color=BORDER, lw=0.7)

        # Top-5 worst drawdowns
        troughs = []; in_dd = False; lmin = 0; lidx = 0
        for i, v in enumerate(dd):
            if v < 0:
                if not in_dd: in_dd=True; lmin=v; lidx=i
                elif v < lmin: lmin=v; lidx=i
            else:
                if in_dd: troughs.append((lmin, lidx)); in_dd=False
        if in_dd: troughs.append((lmin, lidx))
        top5 = sorted(troughs, key=lambda x: x[0])[:5]
        for depth, idx in top5:
            ax.annotate(f"{depth*100:.1f}%\n{str(dates[idx])[:10]}",
                        xy=(dates[idx], depth*100),
                        xytext=(15, 15), textcoords="offset points",
                        color=GOLD, fontsize=7.5, fontweight="bold",
                        arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=0.8),
                        bbox=dict(boxstyle="round,pad=0.3", fc=PANEL_BG, ec=BORDER, alpha=0.85))

        ax.legend(framealpha=0.15, facecolor=PANEL_BG, edgecolor=BORDER,
                  labelcolor=TEXT, fontsize=9)
        plt.tight_layout()
        pdf.savefig(fig, facecolor=BG); plt.close(fig)
        print("[PLOT 2] Drawdown Curve ✓")
    except Exception as e: print(f"[WARNING] Plot 2 failed: {e}")


# ─── Plot 3: Returns Distribution ────────────────────────────────────────────
def plot3_returns_dist(result: pd.DataFrame, pdf: PdfPages) -> None:
    """Return distribution with Normal/t-dist fits, KDE, VaR, CVaR lines."""
    try:
        fig, axes = _new_fig(1, 2, figsize=(15, 6),
                             title="RETURNS DISTRIBUTION  ·  Strategy vs Benchmark")
        for ax, tag, ret_col, col in [
            (axes[0], "Strategy",  "strategy_returns",  ACCENT),
            (axes[1], "Benchmark", "benchmark_returns",  ORANGE),
        ]:
            ret = result[ret_col].dropna().values
            ret = ret[np.abs(ret) < 0.12]   # trim extreme outliers for display
            _setup_ax(ax, title=tag, xlabel="Daily Return (%)", ylabel="Density")

            ax.hist(ret*100, bins=80, density=True, color=col, alpha=0.3,
                    edgecolor="none", label="Observed")

            x   = np.linspace(ret.min()*100, ret.max()*100, 400)
            mu  = ret.mean()*100; sig = ret.std()*100
            ax.plot(x, stats.norm.pdf(x, mu, sig), color=GREEN, lw=2, label="Normal Fit")
            try:
                df_t, loc_t, sc_t = stats.t.fit(ret*100)
                ax.plot(x, stats.t.pdf(x, df_t, loc_t, sc_t),
                        color=PURPLE, lw=2, ls="--", label=f"t-dist (df={df_t:.1f})")
            except Exception: pass
            try:
                kde = gaussian_kde(ret*100)
                ax.plot(x, kde(x), color=GOLD, lw=1.5, ls=":", label="KDE")
            except Exception: pass

            var95  = np.percentile(ret, 5)*100
            cvar95 = ret[ret <= np.percentile(ret, 5)].mean()*100
            ax.axvline(var95,  color=RED,    lw=1.5, ls="--", label=f"VaR 95%={var95:.2f}%")
            ax.axvline(cvar95, color=PURPLE, lw=1.5, ls=":",  label=f"CVaR 95%={cvar95:.2f}%")

            sk = stats.skew(ret); ku = stats.kurtosis(ret)
            ax.text(0.97, 0.97, f"Skew: {sk:.3f}\nKurt: {ku:.3f}",
                    transform=ax.transAxes, color=GOLD, fontsize=8.5,
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.4", fc=PANEL_BG, ec=BORDER, alpha=0.85))
            ax.legend(framealpha=0.15, facecolor=PANEL_BG, edgecolor=BORDER,
                      labelcolor=TEXT, fontsize=7.5)
        plt.tight_layout()
        pdf.savefig(fig, facecolor=BG); plt.close(fig)
        print("[PLOT 3] Returns Distribution ✓")
    except Exception as e: print(f"[WARNING] Plot 3 failed: {e}")


# ─── Plot 4: Trade-by-Trade P&L ──────────────────────────────────────────────
def plot4_trade_pnl(trades: pd.DataFrame, pdf: PdfPages) -> None:
    """Scatter of per-trade P&L colored by win/loss, cumulative overlay."""
    try:
        if len(trades) == 0:
            print("[WARNING] Plot 4 skipped: no trades."); return
        fig, ax1 = plt.subplots(figsize=(15, 6), facecolor=BG)
        _setup_ax(ax1, title="TRADE-BY-TRADE P&L  ·  Execution Quality",
                  xlabel="Entry Date", ylabel="Trade P&L (%)")
        ax2 = ax1.twinx()
        ax2.set_facecolor(PANEL_BG)
        ax2.tick_params(colors=GOLD, labelsize=8)

        t2   = trades.copy()
        t2["entry_date"] = pd.to_datetime(t2["entry_date"])
        wins = t2["pnl_pct"] > 0
        sz   = np.clip(t2["holding_days"] * 25, 15, 250)

        ax1.scatter(t2.loc[wins,  "entry_date"], t2.loc[wins,  "pnl_pct"],
                    c=GREEN, s=sz[wins],  alpha=0.75, zorder=3, label="Win",
                    edgecolors="none")
        ax1.scatter(t2.loc[~wins, "entry_date"], t2.loc[~wins, "pnl_pct"],
                    c=RED,   s=sz[~wins], alpha=0.75, zorder=3, label="Loss",
                    edgecolors="none")
        ax1.axhline(0, color=BORDER, lw=0.8)
        ax1.legend(loc="upper left", framealpha=0.15, facecolor=PANEL_BG,
                   edgecolor=BORDER, labelcolor=TEXT, fontsize=9)

        cum = t2["pnl_pct"].cumsum()
        ax2.plot(t2["entry_date"], cum, color=GOLD, lw=1.8, label="Cumulative P&L (%)")
        ax2.fill_between(t2["entry_date"], cum, 0, color=GOLD, alpha=0.08)
        ax2.set_ylabel("Cumulative P&L (%)", color=GOLD, fontsize=8)
        ax2.legend(loc="upper right", framealpha=0.15, facecolor=PANEL_BG,
                   edgecolor=BORDER, labelcolor=TEXT, fontsize=9)

        # Legend for dot sizes
        for d, lbl in [(1,"1d"),(5,"5d"),(10,"10d")]:
            ax1.scatter([], [], c="white", s=np.clip(d*25,15,250), alpha=0.5, label=f"Hold: {lbl}")
        plt.tight_layout()
        pdf.savefig(fig, facecolor=BG); plt.close(fig)
        print("[PLOT 4] Trade P&L ✓")
    except Exception as e: print(f"[WARNING] Plot 4 failed: {e}")


# ─── Plot 5: Rolling Metrics ─────────────────────────────────────────────────
def plot5_rolling(result: pd.DataFrame, pdf: PdfPages) -> None:
    """4-panel rolling metrics: Sharpe, Sortino, Volatility, Win Rate."""
    try:
        fig = plt.figure(figsize=(15, 12), facecolor=BG)
        fig.suptitle("ROLLING PERFORMANCE METRICS  ·  63-Day Window",
                     color=TEXT, fontsize=13, fontweight="bold", y=0.99)
        axes = [fig.add_subplot(4,1,i+1) for i in range(4)]
        for ax in axes: ax.set_facecolor(PANEL_BG)

        W   = 63
        ret = result["strategy_returns"]
        d   = result.index

        # Sharpe
        rsh = ret.rolling(W).apply(
            lambda x: (x.mean()-RISK_FREE/TRADING_DAYS)/(x.std()+1e-9)*np.sqrt(TRADING_DAYS), raw=True)
        axes[0].plot(d, rsh, color=ACCENT, lw=1.2)
        axes[0].axhline(1.0, color=GOLD, ls="--", lw=0.9, alpha=0.8)
        axes[0].axhline(0.0, color=MUTED, ls="-",  lw=0.5, alpha=0.4)
        axes[0].fill_between(d, rsh, 0, where=rsh>0, color=ACCENT, alpha=0.15)
        axes[0].fill_between(d, rsh, 0, where=rsh<0, color=RED,    alpha=0.15)
        _setup_ax(axes[0], ylabel="Sharpe")
        axes[0].annotate("Sharpe = 1.0", xy=(d[int(len(d)*0.02)], 1.05),
                         color=GOLD, fontsize=7.5)

        # Sortino
        def rs(x):
            down = x[x < RISK_FREE/TRADING_DAYS]
            s    = down.std() * np.sqrt(TRADING_DAYS) if len(down)>1 else 1e-9
            return (x.mean()-RISK_FREE/TRADING_DAYS)*TRADING_DAYS/(s+1e-9)
        rsort = ret.rolling(W).apply(rs, raw=False)
        axes[1].plot(d, rsort, color=GREEN, lw=1.2)
        axes[1].axhline(1.0, color=GOLD, ls="--", lw=0.9, alpha=0.8)
        axes[1].axhline(0.0, color=MUTED, ls="-",  lw=0.5, alpha=0.4)
        axes[1].fill_between(d, rsort, 0, where=rsort>0, color=GREEN, alpha=0.15)
        axes[1].fill_between(d, rsort, 0, where=rsort<0, color=RED,   alpha=0.15)
        _setup_ax(axes[1], ylabel="Sortino")

        # Volatility
        rvol = ret.rolling(W).std() * np.sqrt(TRADING_DAYS) * 100
        axes[2].plot(d, rvol, color=ORANGE, lw=1.2)
        axes[2].fill_between(d, rvol, rvol.mean(), color=ORANGE, alpha=0.12)
        axes[2].axhline(rvol.mean(), color=MUTED, ls="--", lw=0.8)
        _setup_ax(axes[2], ylabel="Ann. Vol (%)")

        # Win Rate
        rwr = ret.rolling(W).apply(lambda x: (x > 0).mean()*100, raw=True)
        axes[3].plot(d, rwr, color=PURPLE, lw=1.2)
        axes[3].axhline(50, color=GOLD, ls="--", lw=0.9, alpha=0.8)
        axes[3].fill_between(d, rwr, 50, where=rwr>50, color=GREEN, alpha=0.12)
        axes[3].fill_between(d, rwr, 50, where=rwr<50, color=RED,   alpha=0.12)
        _setup_ax(axes[3], ylabel="Win Rate (%)", xlabel="Date")

        for ax in axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        plt.tight_layout(rect=[0,0,1,0.97], h_pad=0.3)
        pdf.savefig(fig, facecolor=BG); plt.close(fig)
        print("[PLOT 5] Rolling Metrics ✓")
    except Exception as e: print(f"[WARNING] Plot 5 failed: {e}")


# ─── Plot 6: Heatmaps ────────────────────────────────────────────────────────
def plot6_heatmaps(result: pd.DataFrame, pdf: PdfPages) -> None:
    """Monthly returns heatmap + day-of-week bar chart."""
    try:
        fig = plt.figure(figsize=(15, 8), facecolor=BG)
        fig.suptitle("SEASONALITY ANALYSIS  ·  Monthly Returns & Day-of-Week Effect",
                     color=TEXT, fontsize=13, fontweight="bold", y=0.99)
        gs   = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 1.2], wspace=0.3)
        ax1  = fig.add_subplot(gs[0])
        ax2  = fig.add_subplot(gs[1])
        ax1.set_facecolor(PANEL_BG); ax2.set_facecolor(PANEL_BG)

        ret = result["strategy_returns"].copy()
        ret.index = pd.to_datetime(ret.index)

        monthly = ret.resample("ME").apply(lambda x: (1+x).prod()-1)*100
        piv = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.values,
        }).pivot(index="year", columns="month", values="ret")
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        piv.columns = [month_names[c-1] for c in piv.columns]

        cmap = sns.diverging_palette(10, 133, as_cmap=True)
        sns.heatmap(piv, ax=ax1, cmap=cmap, center=0, annot=True, fmt=".1f",
                    linewidths=0.4, linecolor=BORDER,
                    annot_kws={"size": 7, "color": TEXT},
                    cbar_kws={"shrink": 0.8, "label": "Return (%)"},
                    robust=True)
        ax1.set_title("Monthly Returns (%)", color=GOLD, fontsize=10, fontweight="bold", pad=8)
        ax1.tick_params(colors=TEXT, labelsize=8)
        ax1.set_xlabel("Month", color=MUTED, fontsize=8)
        ax1.set_ylabel("Year", color=MUTED, fontsize=8)
        ax1.collections[0].colorbar.ax.yaxis.label.set_color(TEXT)
        ax1.collections[0].colorbar.ax.tick_params(colors=TEXT)

        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
        ret_dow = ret.to_frame("ret")
        ret_dow["dow"] = ret_dow.index.day_name()
        avg_dow = ret_dow.groupby("dow")["ret"].mean()*100
        avg_dow = avg_dow.reindex(dow_order).fillna(0)
        cols_bar = [GREEN if v >= 0 else RED for v in avg_dow.values]
        bars = ax2.barh(avg_dow.index, avg_dow.values, color=cols_bar, alpha=0.85, height=0.6)
        ax2.axvline(0, color=MUTED, lw=0.8)
        for bar, val in zip(bars, avg_dow.values):
            ax2.text(val + (0.001 if val >= 0 else -0.001),
                     bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}%", va="center",
                     ha="left" if val >= 0 else "right",
                     color=TEXT, fontsize=8)
        ax2.set_title("Avg Return by\nDay of Week (%)", color=GOLD, fontsize=10,
                      fontweight="bold", pad=8)
        _setup_ax(ax2, xlabel="Avg Return (%)")
        ax2.invert_yaxis()

        plt.tight_layout(rect=[0,0,1,0.96])
        pdf.savefig(fig, facecolor=BG); plt.close(fig)
        print("[PLOT 6] Heatmaps ✓")
    except Exception as e: print(f"[WARNING] Plot 6 failed: {e}")


# ─── Plot 7: Risk-Return Scatter ─────────────────────────────────────────────
def plot7_risk_return(result: pd.DataFrame, pdf: PdfPages) -> None:
    """Rolling 252d risk-return scatter with iso-Sharpe lines and time gradient."""
    try:
        fig, ax = _new_fig(figsize=(12, 8),
                           title="RISK-RETURN LANDSCAPE  ·  Rolling 252-Day Windows")
        _setup_ax(ax, xlabel="Annualized Volatility (%)", ylabel="Annualized Return (%)")

        W = TRADING_DAYS
        for col, label, marker, sz in [
            ("strategy_returns",  "Strategy",  "o", 20),
            ("benchmark_returns", "Benchmark", "s", 20),
        ]:
            rv = result[col].rolling(W).std()  * np.sqrt(W) * 100
            rr = result[col].rolling(W).mean() * W * 100
            mask = rv.notna() & rr.notna()
            vols = rv[mask].values; rets = rr[mask].values; n = len(vols)
            if n == 0: continue
            cmap = plt.get_cmap("plasma" if col=="strategy_returns" else "viridis")
            sc = ax.scatter(vols, rets, c=np.arange(n), cmap=cmap, s=sz,
                            marker=marker, alpha=0.5, zorder=3)
            # Mark avg
            ax.scatter(vols.mean(), rets.mean(), color=GOLD if col=="strategy_returns" else ORANGE,
                       s=200, marker="D" if col=="strategy_returns" else "*",
                       zorder=5, label=f"{label} (mean)", edgecolors="white", linewidths=0.8)

        # Iso-Sharpe lines
        all_vols = result["strategy_returns"].rolling(W).std().dropna() * np.sqrt(W) * 100
        vr = np.linspace(max(all_vols.min()-2, 1), all_vols.max()+5, 100)
        for sh, ls, lw in [(0.5,":",0.9),(1.0,"--",1.0),(1.5,"-.",0.9)]:
            rl = RISK_FREE*100 + sh * vr
            ax.plot(vr, rl, color=CYAN, lw=lw, ls=ls, alpha=0.5, label=f"Sharpe={sh}")

        ax.axhline(RISK_FREE*100, color=MUTED, ls="--", lw=0.7, alpha=0.5,
                   label=f"Risk-Free ({RISK_FREE*100:.1f}%)")
        ax.legend(framealpha=0.15, facecolor=PANEL_BG, edgecolor=BORDER,
                  labelcolor=TEXT, fontsize=8)
        plt.tight_layout()
        pdf.savefig(fig, facecolor=BG); plt.close(fig)
        print("[PLOT 7] Risk-Return Scatter ✓")
    except Exception as e: print(f"[WARNING] Plot 7 failed: {e}")


# ─── Plot 8: Monte Carlo ─────────────────────────────────────────────────────
def plot8_monte_carlo(result: pd.DataFrame, pdf: PdfPages) -> None:
    """Bootstrap Monte Carlo with percentile paths and final-value histogram."""
    try:
        N_SIM = 1000
        ret   = result["strategy_returns"].dropna().values
        n     = len(ret)
        sims  = np.zeros((N_SIM, n))
        for i in range(N_SIM):
            samp    = np.random.choice(ret, size=n, replace=True)
            sims[i] = INITIAL_CAPITAL * np.cumprod(1 + samp)

        fig = plt.figure(figsize=(15, 8), facecolor=BG)
        fig.suptitle("MONTE CARLO SIMULATION  ·  1000 Bootstrap Paths",
                     color=TEXT, fontsize=13, fontweight="bold", y=0.99)
        gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 1.5], wspace=0.25)
        ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
        ax1.set_facecolor(PANEL_BG); ax2.set_facecolor(PANEL_BG)

        days = np.arange(n)
        # All paths (very transparent)
        for i in range(N_SIM):
            ax1.plot(days, sims[i]/1e5, color=ACCENT, alpha=0.025, lw=0.3)

        pcts   = [5, 25, 50, 75, 95]
        labels = ["P5","P25","P50","P75","P95"]
        pcolors= [RED, ORANGE, "white", GREEN, CYAN]
        for pct, lbl, col in zip(pcts, labels, pcolors):
            path = np.percentile(sims, pct, axis=0)
            ax1.plot(days, path/1e5, color=col, lw=2.0, label=f"{lbl}: ₹{path[-1]/1e5:.1f}L")

        actual = result["portfolio_value"].values
        ax1.plot(days, actual/1e5, color=GOLD, lw=2.5, ls="--", zorder=5,
                 label=f"Actual: ₹{actual[-1]/1e5:.1f}L")
        _setup_ax(ax1, ylabel="Portfolio Value (₹ Lakhs)", xlabel="Trading Days")
        ax1.legend(framealpha=0.15, facecolor=PANEL_BG, edgecolor=BORDER,
                   labelcolor=TEXT, fontsize=8)

        finals = sims[:, -1] / 1e5
        ax2.hist(finals, bins=50, color=ACCENT, alpha=0.6, orientation="horizontal",
                 edgecolor="none")
        for pct, lbl, col in zip(pcts, labels, pcolors):
            val = np.percentile(finals, pct)
            ax2.axhline(val, color=col, lw=1.5, ls="--", label=f"{lbl}: ₹{val:.1f}L")
        ax2.axhline(actual[-1]/1e5, color=GOLD, lw=2.0, label=f"Actual: ₹{actual[-1]/1e5:.1f}L")
        _setup_ax(ax2, ylabel="Final Value (₹ Lakhs)", xlabel="Count")
        ax2.legend(framealpha=0.15, facecolor=PANEL_BG, edgecolor=BORDER,
                   labelcolor=TEXT, fontsize=7.5)

        print(f"\n[MC] P5  Final : ₹{np.percentile(sims[:,-1],5):>12,.0f}")
        print(f"[MC] P50 Final : ₹{np.percentile(sims[:,-1],50):>12,.0f}")
        print(f"[MC] P95 Final : ₹{np.percentile(sims[:,-1],95):>12,.0f}")

        plt.tight_layout(rect=[0,0,1,0.97])
        pdf.savefig(fig, facecolor=BG); plt.close(fig)
        print("[PLOT 8] Monte Carlo ✓")
    except Exception as e: print(f"[WARNING] Plot 8 failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(result: pd.DataFrame, trades: pd.DataFrame) -> None:
    """Persist trades log and equity curve as CSV files."""
    try:
        trades.to_csv(TRADES_CSV, index=False)
        print(f"[OUTPUT] {TRADES_CSV} saved ({len(trades)} rows)")
    except Exception as e:
        print(f"[WARNING] Could not save trades CSV: {e}")
    try:
        dd = drawdown_series(result["portfolio_value"].values)
        rs = result["strategy_returns"].rolling(TRADING_DAYS).apply(
            lambda x: (x.mean()-RISK_FREE/TRADING_DAYS)/(x.std()+1e-9)*np.sqrt(TRADING_DAYS), raw=True)
        eq = pd.DataFrame({
            "date"           : result.index,
            "portfolio_value": result["portfolio_value"].values,
            "benchmark_value": result["benchmark_value"].values,
            "drawdown_pct"   : dd * 100,
            "rolling_sharpe" : rs.values,
            "strategy_ret"   : result["strategy_returns"].values,
            "benchmark_ret"  : result["benchmark_returns"].values,
        })
        eq.to_csv(EQUITY_CSV, index=False)
        print(f"[OUTPUT] {EQUITY_CSV} saved ({len(eq)} rows)")
    except Exception as e:
        print(f"[WARNING] Could not save equity CSV: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Execute the full Nifty 50 quantitative trading pipeline end-to-end."""
    print("\n" + "═"*66)
    print("  NIFTY 50 QUANT SYSTEM v2.0  ·  Initializing Pipeline")
    print("═"*66 + "\n")

    df       = download_data()
    features = compute_features(df)
    print(f"[PIPELINE] Features computed: {features.shape}")

    signals  = generate_signals(features)
    sc       = signals["signal"].value_counts()
    print(f"[PIPELINE] Signal counts → Long:{sc.get(1.0,0)}  "
          f"Short:{sc.get(-1.0,0)}  Flat:{sc.get(0.0,0)}\n")

    result, trades = run_backtest(signals)
    print(f"[PIPELINE] Trades executed  : {len(trades)}")
    if len(trades) > 0:
        print(f"[PIPELINE] Avg hold (days)  : {trades['holding_days'].mean():.2f}")
    print(f"[PIPELINE] Final portfolio  : ₹{result['portfolio_value'].iloc[-1]:>12,.0f}")
    print(f"[PIPELINE] Final benchmark  : ₹{result['benchmark_value'].iloc[-1]:>12,.0f}\n")

    metrics = compute_metrics(result, trades)
    print_metrics(metrics)

    print("[PIPELINE] Generating 8-plot PDF report ...")
    with PdfPages(PDF_FILE) as pdf:
        # PDF metadata
        d = pdf.infodict()
        d["Title"]   = "Nifty 50 Quant System v2.0 — Performance Report"
        d["Author"]  = "UNiverse Capital"
        d["Subject"] = "Quantitative Trading Analysis"
        plot1_equity_curve(result, pdf)
        plot2_drawdown(result, pdf)
        plot3_returns_dist(result, pdf)
        plot4_trade_pnl(trades, pdf)
        plot5_rolling(result, pdf)
        plot6_heatmaps(result, pdf)
        plot7_risk_return(result, pdf)
        plot8_monte_carlo(result, pdf)
    print(f"[OUTPUT] PDF saved → {PDF_FILE}")

    save_outputs(result, trades)

    print("\n" + "═"*66)
    print("  PIPELINE COMPLETE ✓")
    print("═"*66 + "\n")

if __name__ == "__main__":
    main()