"""
Institutional-Grade Nifty 50 Index Futures Quantitative Trading System
=======================================================================
Non-lagging, signal-based multi-factor model with full performance analytics.
Run: python model.py
"""

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
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from scipy import stats
from scipy.special import gammaln
import yfinance as yf
from tabulate import tabulate
from datetime import datetime, timedelta
import math
import itertools

np.random.seed(42)

# Config
TICKER         = "^NSEI"
INITIAL_CAPITAL = 1_000_000      # INR 10 lakhs
TC              = 0.0005         # 0.05% per side
SLIPPAGE        = 0.0002         # 0.02% per side
RISK_FREE       = 0.065          # 6.5% Indian 10Y G-Sec
KELLY_CAP       = 2.0
HURST_TREND     = 0.55
HURST_MR        = 0.45
VOL_FILTER_PCT  = 90             # skip top 10% vol days
PLOT_STYLE      = "dark_background"
PDF_FILE        = "nifty50_quant_report.pdf"
TRADES_CSV      = "trades_log.csv"
EQUITY_CSV      = "equity_curve.csv"
TRADING_DAYS    = 252


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_nifty(start: datetime, end: datetime) -> pd.DataFrame:
    """
    Generate synthetic Nifty 50 OHLCV data calibrated to historical parameters.
    Uses GBM for price path + AR(1) factor for autocorrelation + realistic intraday structure.
    Parameters calibrated to Nifty 50 observed statistics (2014-2024).
    """
    print("[DATA] Generating synthetic Nifty 50 data (GBM + AR(1), calibrated to historical params) ...")
    # Calibrated Nifty 50 parameters
    annual_mu    = 0.13     # ~13% CAGR
    annual_sigma = 0.175    # ~17.5% annualized vol
    ar1_coef     = 0.08     # mild positive autocorrelation
    jump_prob    = 0.015    # 1.5% daily jump probability
    jump_mu      = -0.02    # avg jump size (negative bias for fat tails)
    jump_sigma   = 0.04     # jump vol
    base_price   = 8000.0   # Nifty level circa 2016

    # Business days
    bdays = pd.bdate_range(start=start, end=end, freq="B")
    n     = len(bdays)
    np.random.seed(42)

    dt   = 1 / TRADING_DAYS
    mu_d = annual_mu * dt
    sig_d= annual_sigma * np.sqrt(dt)

    # AR(1) residual
    eps   = np.random.normal(0, 1, n)
    ar_eps= np.zeros(n)
    ar_eps[0] = eps[0]
    for i in range(1, n):
        ar_eps[i] = ar1_coef * ar_eps[i-1] + eps[i] * np.sqrt(1 - ar1_coef**2)

    # Jump component
    jumps  = np.where(np.random.rand(n) < jump_prob,
                      np.random.normal(jump_mu, jump_sigma, n), 0.0)

    # Daily log returns
    log_rets = mu_d - 0.5 * annual_sigma**2 * dt + sig_d * ar_eps + jumps

    # Price path
    closes = base_price * np.exp(np.cumsum(log_rets))

    # Intraday structure: realistic OHLC
    intraday_vol = annual_sigma * np.sqrt(dt) * np.random.uniform(0.5, 2.0, n)
    opens  = closes * np.exp(np.random.normal(0, intraday_vol * 0.3, n))   # gap
    highs  = np.maximum(opens, closes) * np.exp(np.abs(np.random.normal(0, intraday_vol * 0.5, n)))
    lows   = np.minimum(opens, closes) * np.exp(-np.abs(np.random.normal(0, intraday_vol * 0.5, n)))

    # Volume: log-normal with autocorrelation
    vol_base   = 200_000_000
    vol_noise  = np.random.lognormal(0, 0.5, n)
    volumes    = (vol_base * vol_noise).astype(int)

    df = pd.DataFrame({
        "Open"  : np.round(opens, 2),
        "High"  : np.round(highs, 2),
        "Low"   : np.round(lows,  2),
        "Close" : np.round(closes, 2),
        "Volume": volumes,
    }, index=bdays)

    # Ensure OHLC consistency
    df["High"]  = df[["Open","High","Close"]].max(axis=1)
    df["Low"]   = df[["Open","Low","Close"]].min(axis=1)
    return df


def download_data() -> pd.DataFrame:
    """Download 10 years of Nifty 50 daily OHLCV data; fall back to synthetic if unavailable."""
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
        if len(raw) > 200:
            df = raw.copy()
            print(f"[DATA] Live data retrieved: {len(df)} rows.")
        else:
            raise ValueError("Insufficient rows from live feed.")
    except Exception as e:
        print(f"[DATA] Live download failed ({e}). Using calibrated synthetic data.")
        df = _generate_synthetic_nifty(start, end)

    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    df       = df[df["Volume"] > 0].sort_index()
    print(f"[DATA] Shape       : {df.shape}")
    print(f"[DATA] Date range  : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"[DATA] First 5 rows:\n{df.head()}\n")
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def compute_hurst(series: np.ndarray) -> float:
    """Compute Hurst Exponent via R/S analysis on a 1-D array."""
    n = len(series)
    if n < 20:
        return 0.5
    try:
        lags   = range(2, min(n // 2, 20))
        rs_vals = []
        for lag in lags:
            sub   = series[:lag]
            sub   = sub - sub.mean()
            cs    = np.cumsum(sub)
            r     = cs.max() - cs.min()
            s     = sub.std(ddof=1)
            if s > 0:
                rs_vals.append(np.log(r / s))
            else:
                rs_vals.append(np.nan)
        log_lags = np.log(list(lags))
        rs_vals  = np.array(rs_vals)
        mask     = ~np.isnan(rs_vals)
        if mask.sum() < 4:
            return 0.5
        slope, *_ = np.polyfit(log_lags[mask], rs_vals[mask], 1)
        return float(np.clip(slope, 0.0, 1.0))
    except Exception:
        return 0.5


def parkinson_vol(high: pd.Series, low: pd.Series) -> pd.Series:
    """Compute Parkinson single-day realized volatility estimate."""
    return np.sqrt((1 / (4 * np.log(2))) * (np.log(high / low)) ** 2)


def variance_ratio(log_ret: pd.Series, k: int = 5) -> pd.Series:
    """Rolling Lo-MacKinlay variance ratio (k-day vs 1-day)."""
    var1 = log_ret.rolling(k).var()
    vark = log_ret.rolling(k).apply(
        lambda x: np.sum(x) ** 2 / k if len(x) == k else np.nan, raw=True
    )
    vr = vark / (var1 * k + 1e-12)
    return vr


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all non-lagging features for signal generation."""
    f = df.copy()
    O, H, L, C, V = f["Open"], f["High"], f["Low"], f["Close"], f["Volume"]
    prev_C = C.shift(1)

    # ── A. Price Action ──
    f["daily_ret"]   = (C - O) / (O + 1e-9)
    f["intraday_rng"]= (H - L) / (O + 1e-9)
    f["gap"]         = (O - prev_C) / (prev_C + 1e-9)
    f["body_ratio"]  = np.abs(C - O) / (H - L + 1e-9)
    f["upper_wick"]  = (H - np.maximum(O, C)) / (H - L + 1e-9)
    f["lower_wick"]  = (np.minimum(O, C) - L) / (H - L + 1e-9)
    f["gap_dir"]     = np.sign(f["gap"])

    # ── B. Statistical / Volatility ──
    f["realized_vol"]  = f["log_ret"].rolling(5).std() * np.sqrt(TRADING_DAYS)
    vol_mean           = V.rolling(5).mean()
    vol_std            = V.rolling(5).std()
    f["vol_zscore"]    = (V - vol_mean) / (vol_std + 1e-9)
    f["parkinson_vol"] = parkinson_vol(H, L)
    f["variance_ratio"]= variance_ratio(f["log_ret"], k=5)

    # ── C. Regime Detection ──
    roll_min = C.rolling(20).min()
    roll_max = C.rolling(20).max()
    f["range_zscore"]  = (C - roll_min) / (roll_max - roll_min + 1e-9)

    # Rolling Hurst (60-day window)
    hurst_vals = np.full(len(f), 0.5)
    log_ret_arr = f["log_ret"].values
    for i in range(60, len(f)):
        hurst_vals[i] = compute_hurst(log_ret_arr[i-60:i])
    f["hurst"] = hurst_vals

    f.dropna(inplace=True)
    return f


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(f: pd.DataFrame) -> pd.DataFrame:
    """Generate +1 / -1 / 0 signals using composite scoring with regime & vol filters."""
    s = f.copy()

    # ── Parkinson vol filter threshold ──
    pvol_thresh = np.nanpercentile(s["parkinson_vol"].values, VOL_FILTER_PCT)

    # ── Composite score (threshold-based) ──
    score = pd.Series(0.0, index=s.index)

    # Trend score components
    score += np.where(s["daily_ret"]  > 0.002,  1.0, np.where(s["daily_ret"]  < -0.002, -1.0, 0.0))
    score += np.where(s["gap_dir"]    > 0,       0.5, np.where(s["gap_dir"]    < 0,      -0.5, 0.0))
    score += np.where(s["body_ratio"] > 0.6,    np.sign(s["daily_ret"]) * 0.5, 0.0)
    score += np.where(s["vol_zscore"] > 1.5,    np.sign(s["daily_ret"]) * 0.5, 0.0)

    # Mean-reversion score components
    mr_score = pd.Series(0.0, index=s.index)
    mr_score += np.where(s["range_zscore"] < 0.2,  1.0, np.where(s["range_zscore"] > 0.8, -1.0, 0.0))
    mr_score += np.where(s["lower_wick"]   > 0.4,  0.5, 0.0)
    mr_score += np.where(s["upper_wick"]   > 0.4, -0.5, 0.0)
    mr_score += np.where(s["variance_ratio"] < 0.8, 0.5 * np.sign(0.5 - s["range_zscore"]), 0.0)

    # ── Regime filter ──
    trend_signal = np.sign(score)
    mr_signal    = np.sign(mr_score)

    raw_signal = np.where(
        s["hurst"] > HURST_TREND, trend_signal,
        np.where(s["hurst"] < HURST_MR, mr_signal, 0.0)
    )

    # ── Volatility filter (skip extreme vol) ──
    raw_signal = np.where(s["parkinson_vol"] > pvol_thresh, 0.0, raw_signal)

    s["raw_signal"] = raw_signal

    # ── Kelly sizing ──
    win_mask  = s["log_ret"] > 0
    avg_win   = s.loc[win_mask,  "log_ret"].mean() if win_mask.sum() > 0 else 0.01
    avg_loss  = s.loc[~win_mask, "log_ret"].abs().mean() if (~win_mask).sum() > 0 else 0.01
    win_rate  = win_mask.mean()
    kelly_f   = (win_rate / (avg_loss + 1e-9)) - ((1 - win_rate) / (avg_win + 1e-9))
    kelly_f   = float(np.clip(kelly_f, 0.1, KELLY_CAP))

    # Volatility scaling
    rv_norm   = s["realized_vol"] / (s["realized_vol"].mean() + 1e-9)
    pos_size  = kelly_f / (rv_norm + 1e-9)
    pos_size  = pos_size.clip(upper=KELLY_CAP)

    s["signal"]   = raw_signal
    s["pos_size"] = pos_size * np.abs(raw_signal)
    s["pos_size"] = s["pos_size"].clip(upper=KELLY_CAP)

    return s


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: BACKTESTING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(s: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vectorized backtest engine. Signal on T → enter T+1 Open, exit T+1 Close."""
    n      = len(s)
    dates  = s.index.to_numpy()
    opens  = s["Open"].values
    closes = s["Close"].values
    sigs   = s["signal"].values
    sizes  = s["pos_size"].values

    portfolio = np.zeros(n)
    benchmark = np.zeros(n)
    pnl       = np.zeros(n)
    portfolio[0] = INITIAL_CAPITAL
    benchmark[0] = INITIAL_CAPITAL

    trade_log = []
    capital   = INITIAL_CAPITAL
    bm_capital= INITIAL_CAPITAL

    for i in range(1, n):
        sig  = sigs[i-1]     # signal from previous day
        sz   = sizes[i-1]
        entry_px = opens[i]
        exit_px  = closes[i]

        # Benchmark: buy-and-hold
        bm_ret     = (closes[i] - closes[i-1]) / (closes[i-1] + 1e-9)
        bm_capital *= (1 + bm_ret)
        benchmark[i] = bm_capital

        if sig == 0:
            portfolio[i] = capital
            pnl[i]       = 0
            continue

        # Position value capped at capital
        pos_val  = min(capital * sz, capital)
        # Gross return of the trade
        raw_ret  = sig * (exit_px - entry_px) / (entry_px + 1e-9)
        # Costs
        cost     = (TC + SLIPPAGE) * 2   # both sides
        net_ret  = raw_ret - cost

        trade_pnl   = pos_val * net_ret
        capital    += trade_pnl
        capital     = max(capital, 0)    # floor at 0
        portfolio[i]= capital
        pnl[i]      = trade_pnl

        # Record trade
        trade_log.append({
            "entry_date"  : str(dates[i])[:10],
            "exit_date"   : str(dates[i])[:10],
            "signal"      : int(sig),
            "entry_price" : round(float(entry_px), 2),
            "exit_price"  : round(float(exit_px), 2),
            "pnl_pct"     : round(float(net_ret * 100), 4),
            "pnl_inr"     : round(float(trade_pnl), 2),
            "holding_days": 1,
        })

    s = s.copy()
    s["portfolio_value"]    = portfolio
    s["benchmark_value"]    = benchmark
    s["daily_pnl"]          = pnl
    s["strategy_returns"]   = pd.Series(portfolio, index=s.index).pct_change().fillna(0)
    s["benchmark_returns"]  = pd.Series(benchmark, index=s.index).pct_change().fillna(0)

    trades_df = pd.DataFrame(trade_log)
    return s, trades_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Compute drawdown from peak at each point."""
    peak = np.maximum.accumulate(equity)
    return (equity - peak) / (peak + 1e-9)


def max_drawdown_duration(dd: np.ndarray) -> int:
    """Return max number of consecutive days underwater."""
    max_dur, cur_dur = 0, 0
    for v in dd:
        if v < 0:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    return max_dur


def hurst_exponent(series: np.ndarray) -> float:
    """Compute Hurst exponent on the full return series."""
    return compute_hurst(series)


def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """Compute Omega ratio relative to threshold."""
    excess = returns - threshold
    gains  = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    return gains / (losses + 1e-9)


def calmar_ratio(cagr: float, max_dd: float) -> float:
    """Compute Calmar ratio."""
    return cagr / (abs(max_dd) + 1e-9)


def sortino_ratio(returns: np.ndarray, rf: float = RISK_FREE) -> float:
    """Compute annualized Sortino ratio."""
    daily_rf  = rf / TRADING_DAYS
    excess    = returns - daily_rf
    downside  = returns[returns < daily_rf]
    semi_dev  = np.std(downside, ddof=1) * np.sqrt(TRADING_DAYS) if len(downside) > 1 else 1e-9
    ann_exc   = excess.mean() * TRADING_DAYS
    return ann_exc / (semi_dev + 1e-9)


def information_ratio(strat_ret: np.ndarray, bench_ret: np.ndarray) -> float:
    """Compute Information Ratio vs benchmark."""
    active  = strat_ret - bench_ret
    te      = np.std(active, ddof=1) * np.sqrt(TRADING_DAYS)
    return (active.mean() * TRADING_DAYS) / (te + 1e-9)


def max_consecutive(wins: np.ndarray) -> tuple[int, int]:
    """Return (max_consecutive_wins, max_consecutive_losses)."""
    max_w = max_l = cw = cl = 0
    for w in wins:
        if w:
            cw += 1; cl = 0; max_w = max(max_w, cw)
        else:
            cl += 1; cw = 0; max_l = max(max_l, cl)
    return max_w, max_l


def compute_metrics(result: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """Compute full performance metrics for strategy and benchmark."""
    strat_ret = result["strategy_returns"].values
    bench_ret = result["benchmark_returns"].values
    port_val  = result["portfolio_value"].values
    bench_val = result["benchmark_value"].values
    n_years   = len(strat_ret) / TRADING_DAYS

    def _cagr(final, initial, years):
        return (final / initial) ** (1 / years) - 1 if years > 0 else 0

    def _ann_vol(r):
        return np.std(r, ddof=1) * np.sqrt(TRADING_DAYS)

    def _sharpe(r, rf=RISK_FREE):
        excess = r - rf / TRADING_DAYS
        vol    = _ann_vol(r)
        return (excess.mean() * TRADING_DAYS) / (vol + 1e-9)

    def _var(r, pct=5):
        return np.percentile(r, pct)

    def _cvar(r, pct=5):
        v = _var(r, pct)
        return r[r <= v].mean() if (r <= v).sum() > 0 else v

    dd_s = drawdown_series(port_val)
    dd_b = drawdown_series(bench_val)

    # Annual returns
    result2 = result.copy()
    result2.index = pd.to_datetime(result2.index)
    ann_strat = result2["strategy_returns"].resample("YE").apply(
        lambda x: (1 + x).prod() - 1)
    ann_bench = result2["benchmark_returns"].resample("YE").apply(
        lambda x: (1 + x).prod() - 1)

    m = {}
    for tag, ret, equity, dd, ann in [
        ("strategy",  strat_ret, port_val,  dd_s, ann_strat),
        ("benchmark", bench_ret, bench_val, dd_b, ann_bench),
    ]:
        cagr    = _cagr(equity[-1], equity[0], n_years)
        max_dd  = dd.min()
        m[tag]  = {
            "total_return"    : (equity[-1] / equity[0] - 1) * 100,
            "cagr"            : cagr * 100,
            "best_year"       : ann.max() * 100,
            "worst_year"      : ann.min() * 100,
            "avg_ann_ret"     : ann.mean() * 100,
            "ann_vol"         : _ann_vol(ret) * 100,
            "max_drawdown"    : max_dd * 100,
            "max_dd_duration" : max_drawdown_duration(dd),
            "avg_drawdown"    : dd[dd < 0].mean() * 100 if (dd < 0).sum() > 0 else 0,
            "var95"           : _var(ret, 5) * 100,
            "cvar95"          : _cvar(ret, 5) * 100,
            "downside_dev"    : np.std(ret[ret < 0], ddof=1) * np.sqrt(TRADING_DAYS) * 100,
            "sharpe"          : _sharpe(ret),
            "sortino"         : sortino_ratio(ret),
            "calmar"          : calmar_ratio(cagr, max_dd),
            "omega"           : omega_ratio(ret),
            "skewness"        : stats.skew(ret),
            "kurtosis"        : stats.kurtosis(ret),
            "hurst"           : hurst_exponent(ret),
            "autocorr_lag1"   : pd.Series(ret).autocorr(lag=1),
        }
    m["strategy"]["info_ratio"] = information_ratio(strat_ret, bench_ret)
    m["benchmark"]["info_ratio"] = 0.0

    # Trade metrics
    if len(trades) > 0:
        wins       = trades["pnl_pct"] > 0
        m["trades"] = {
            "n_trades"   : len(trades),
            "win_rate"   : wins.mean() * 100,
            "avg_win"    : trades.loc[wins,  "pnl_pct"].mean() if wins.sum() > 0 else 0,
            "avg_loss"   : trades.loc[~wins, "pnl_pct"].mean() if (~wins).sum() > 0 else 0,
            "profit_factor": (trades.loc[wins,  "pnl_pct"].sum() /
                              (-trades.loc[~wins, "pnl_pct"].sum() + 1e-9)),
            "expectancy" : trades["pnl_pct"].mean(),
            "avg_holding": trades["holding_days"].mean(),
            "max_consec_w": max_consecutive(wins.values)[0],
            "max_consec_l": max_consecutive(wins.values)[1],
        }
    else:
        m["trades"] = {k: 0 for k in [
            "n_trades","win_rate","avg_win","avg_loss","profit_factor",
            "expectancy","avg_holding","max_consec_w","max_consec_l"]}
    return m


def print_metrics(m: dict) -> None:
    """Print formatted performance metrics table."""
    s, b = m["strategy"], m["benchmark"]
    t    = m["trades"]
    rows = [
        ["── RETURN METRICS ──",             "",         ""],
        ["Total Return (%)",                 f"{s['total_return']:.2f}",    f"{b['total_return']:.2f}"],
        ["CAGR (%)",                         f"{s['cagr']:.2f}",            f"{b['cagr']:.2f}"],
        ["Best Year (%)",                    f"{s['best_year']:.2f}",       f"{b['best_year']:.2f}"],
        ["Worst Year (%)",                   f"{s['worst_year']:.2f}",      f"{b['worst_year']:.2f}"],
        ["Avg Annual Return (%)",            f"{s['avg_ann_ret']:.2f}",     f"{b['avg_ann_ret']:.2f}"],
        ["── RISK METRICS ──",               "",         ""],
        ["Annualized Volatility (%)",        f"{s['ann_vol']:.2f}",         f"{b['ann_vol']:.2f}"],
        ["Max Drawdown (%)",                 f"{s['max_drawdown']:.2f}",    f"{b['max_drawdown']:.2f}"],
        ["Max DD Duration (days)",           f"{s['max_dd_duration']}",     f"{b['max_dd_duration']}"],
        ["Avg Drawdown (%)",                 f"{s['avg_drawdown']:.2f}",    f"{b['avg_drawdown']:.2f}"],
        ["VaR 95% (daily %)",               f"{s['var95']:.4f}",           f"{b['var95']:.4f}"],
        ["CVaR 95% (daily %)",              f"{s['cvar95']:.4f}",          f"{b['cvar95']:.4f}"],
        ["Downside Deviation (%)",           f"{s['downside_dev']:.4f}",    f"{b['downside_dev']:.4f}"],
        ["── RISK-ADJUSTED ──",              "",         ""],
        ["Sharpe Ratio",                     f"{s['sharpe']:.4f}",          f"{b['sharpe']:.4f}"],
        ["Sortino Ratio",                    f"{s['sortino']:.4f}",         f"{b['sortino']:.4f}"],
        ["Calmar Ratio",                     f"{s['calmar']:.4f}",          f"{b['calmar']:.4f}"],
        ["Omega Ratio",                      f"{s['omega']:.4f}",           f"{b['omega']:.4f}"],
        ["Information Ratio",                f"{s['info_ratio']:.4f}",      "N/A"],
        ["── TRADE METRICS ──",              "",         ""],
        ["Total Trades",                     f"{t['n_trades']}",            "N/A"],
        ["Win Rate (%)",                     f"{t['win_rate']:.2f}",        "N/A"],
        ["Avg Win (%)",                      f"{t['avg_win']:.4f}",         "N/A"],
        ["Avg Loss (%)",                     f"{t['avg_loss']:.4f}",        "N/A"],
        ["Profit Factor",                    f"{t['profit_factor']:.4f}",   "N/A"],
        ["Expectancy (% per trade)",         f"{t['expectancy']:.4f}",      "N/A"],
        ["Avg Holding Period (days)",        f"{t['avg_holding']:.2f}",     "N/A"],
        ["Max Consecutive Wins",             f"{t['max_consec_w']}",        "N/A"],
        ["Max Consecutive Losses",           f"{t['max_consec_l']}",        "N/A"],
        ["── STATISTICAL ──",               "",         ""],
        ["Skewness",                         f"{s['skewness']:.4f}",        f"{b['skewness']:.4f}"],
        ["Kurtosis",                         f"{s['kurtosis']:.4f}",        f"{b['kurtosis']:.4f}"],
        ["Hurst Exponent",                   f"{s['hurst']:.4f}",           f"{b['hurst']:.4f}"],
        ["Autocorrelation (lag-1)",          f"{s['autocorr_lag1']:.4f}",   f"{b['autocorr_lag1']:.4f}"],
    ]
    print("\n" + "="*68)
    print("  NIFTY 50 QUANT SYSTEM — PERFORMANCE REPORT")
    print("="*68)
    print(tabulate(rows, headers=["Metric", "Strategy", "Benchmark"],
                   tablefmt="simple", colalign=("left","right","right")))
    print("="*68 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _style():
    """Apply dark professional matplotlib style."""
    plt.style.use(PLOT_STYLE)
    plt.rcParams.update({
        "font.family"     : "monospace",
        "axes.titlesize"  : 11,
        "axes.labelsize"  : 9,
        "xtick.labelsize" : 8,
        "ytick.labelsize" : 8,
        "figure.dpi"      : 120,
        "axes.spines.top" : False,
        "axes.spines.right": False,
    })

ACCENT   = "#00D4FF"
ACCENT2  = "#FF6B35"
GREEN_C  = "#00FF88"
RED_C    = "#FF3355"
GOLD_C   = "#FFD700"


def plot1_equity_curve(result: pd.DataFrame, pdf: PdfPages) -> None:
    """Plot equity curve vs benchmark with drawdown periods and rolling Sharpe."""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                        gridspec_kw={"height_ratios": [3, 1]},
                                        sharex=True)
        fig.patch.set_facecolor("#0A0A0A")
        ax1.set_facecolor("#111111"); ax2.set_facecolor("#111111")

        dates = result.index
        norm_strat = result["portfolio_value"] / result["portfolio_value"].iloc[0] * 100
        norm_bench = result["benchmark_value"] / result["benchmark_value"].iloc[0] * 100

        ax1.plot(dates, norm_strat, color=ACCENT,  lw=1.5, label="Strategy",  zorder=3)
        ax1.plot(dates, norm_bench, color=ACCENT2, lw=1.5, label="Benchmark", zorder=2, alpha=0.8)

        # Shade drawdown periods
        dd = drawdown_series(result["portfolio_value"].values)
        in_dd = False
        dd_start = None
        for i, (d, v) in enumerate(zip(dates, dd)):
            if v < -0.05 and not in_dd:
                in_dd = True; dd_start = d
            elif v >= -0.01 and in_dd:
                ax1.axvspan(dd_start, d, color=RED_C, alpha=0.12, zorder=1)
                in_dd = False
        if in_dd:
            ax1.axvspan(dd_start, dates[-1], color=RED_C, alpha=0.12, zorder=1)

        ax1.set_ylabel("Normalized Value (base=100)", color="white")
        ax1.set_title("EQUITY CURVE — Strategy vs Benchmark (Nifty 50)", color=GOLD_C, fontweight="bold")
        ax1.legend(framealpha=0.2, facecolor="#1A1A1A", edgecolor="#333333")
        ax1.tick_params(colors="white"); ax1.yaxis.label.set_color("white")

        # Rolling Sharpe
        sr = result["strategy_returns"]
        rolling_sharpe = sr.rolling(TRADING_DAYS).apply(
            lambda x: (x.mean() - RISK_FREE/TRADING_DAYS) / (x.std() + 1e-9) * np.sqrt(TRADING_DAYS),
            raw=True
        )
        ax2.plot(dates, rolling_sharpe, color=GREEN_C, lw=1.2, label="252d Rolling Sharpe")
        ax2.axhline(1.0, color=GOLD_C, ls="--", lw=0.8, alpha=0.7)
        ax2.axhline(0.0, color="white", ls="-",  lw=0.5, alpha=0.3)
        ax2.fill_between(dates, rolling_sharpe, 0,
                         where=rolling_sharpe > 0, color=GREEN_C, alpha=0.2)
        ax2.fill_between(dates, rolling_sharpe, 0,
                         where=rolling_sharpe < 0, color=RED_C,   alpha=0.2)
        ax2.set_ylabel("Sharpe", color="white"); ax2.set_xlabel("Date", color="white")
        ax2.legend(framealpha=0.2, facecolor="#1A1A1A")
        ax2.tick_params(colors="white")

        plt.tight_layout(pad=1.5)
        pdf.savefig(fig, facecolor=fig.get_facecolor()); plt.close(fig)
        print("[PLOT 1] Equity Curve ✓")
    except Exception as e:
        print(f"[WARNING] Plot 1 failed: {e}")


def plot2_drawdown(result: pd.DataFrame, pdf: PdfPages) -> None:
    """Plot drawdown curve with top-5 worst drawdown annotations."""
    try:
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor("#0A0A0A"); ax.set_facecolor("#111111")

        dates = result.index
        dd    = drawdown_series(result["portfolio_value"].values)
        dd_s  = pd.Series(dd, index=dates)

        ax.fill_between(dates, dd_s * 100, 0, color=RED_C, alpha=0.4, label="Drawdown")
        ax.plot(dates, dd_s * 100, color=RED_C, lw=0.8)

        # Top-5 drawdowns
        trough_idx = []
        in_dd = False; local_min = 0; local_idx = 0
        for i, v in enumerate(dd):
            if v < 0:
                if not in_dd:
                    in_dd = True; local_min = v; local_idx = i
                elif v < local_min:
                    local_min = v; local_idx = i
            else:
                if in_dd:
                    trough_idx.append((local_min, local_idx)); in_dd = False
        if in_dd:
            trough_idx.append((local_min, local_idx))
        top5 = sorted(trough_idx, key=lambda x: x[0])[:5]
        for depth, idx in top5:
            ax.annotate(f"{depth*100:.1f}%\n{str(dates[idx])[:10]}",
                        xy=(dates[idx], depth*100),
                        xytext=(10, -20), textcoords="offset points",
                        color=GOLD_C, fontsize=7, fontweight="bold",
                        arrowprops=dict(arrowstyle="-|>", color=GOLD_C, lw=0.8))

        ax.set_title("DRAWDOWN CURVE — Rolling Underwater Equity", color=GOLD_C, fontweight="bold")
        ax.set_ylabel("Drawdown (%)", color="white")
        ax.set_xlabel("Date", color="white")
        ax.tick_params(colors="white")
        ax.legend(framealpha=0.2, facecolor="#1A1A1A")
        plt.tight_layout(pad=1.5)
        pdf.savefig(fig, facecolor=fig.get_facecolor()); plt.close(fig)
        print("[PLOT 2] Drawdown Curve ✓")
    except Exception as e:
        print(f"[WARNING] Plot 2 failed: {e}")


def plot3_returns_dist(result: pd.DataFrame, pdf: PdfPages) -> None:
    """Plot returns distribution with Normal/t-dist fits and VaR/CVaR lines."""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("#0A0A0A"); ax.set_facecolor("#111111")

        ret = result["strategy_returns"].dropna().values
        ret = ret[np.abs(ret) < 0.1]   # trim extreme outliers for display

        ax.hist(ret * 100, bins=80, density=True, color=ACCENT, alpha=0.4, label="Daily Returns")

        x = np.linspace(ret.min()*100, ret.max()*100, 300)
        mu, sig = ret.mean()*100, ret.std()*100

        # Normal fit
        ax.plot(x, stats.norm.pdf(x, mu, sig), color=GREEN_C, lw=2, label="Normal Fit")

        # t-dist fit
        try:
            df_t, loc_t, scale_t = stats.t.fit(ret * 100)
            ax.plot(x, stats.t.pdf(x, df_t, loc_t, scale_t),
                    color=ACCENT2, lw=2, ls="--", label=f"t-dist (df={df_t:.1f})")
        except Exception:
            pass

        # KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(ret * 100)
        ax.plot(x, kde(x), color=GOLD_C, lw=1.5, ls=":", label="KDE")

        # VaR / CVaR
        var95  = np.percentile(ret, 5) * 100
        cvar95 = ret[ret <= np.percentile(ret, 5)].mean() * 100
        ax.axvline(var95,  color=RED_C,  lw=1.5, ls="--", label=f"VaR 95% = {var95:.2f}%")
        ax.axvline(cvar95, color="magenta", lw=1.5, ls="--", label=f"CVaR 95% = {cvar95:.2f}%")

        skew_v = stats.skew(ret)
        kurt_v = stats.kurtosis(ret)
        ax.text(0.03, 0.95, f"Skewness: {skew_v:.3f}\nKurtosis: {kurt_v:.3f}",
                transform=ax.transAxes, color=GOLD_C, fontsize=9,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="#1A1A1A", alpha=0.7))

        ax.set_title("RETURNS DISTRIBUTION — Strategy Daily Returns", color=GOLD_C, fontweight="bold")
        ax.set_xlabel("Daily Return (%)", color="white")
        ax.set_ylabel("Density", color="white")
        ax.tick_params(colors="white")
        ax.legend(framealpha=0.2, facecolor="#1A1A1A", fontsize=8)
        plt.tight_layout(pad=1.5)
        pdf.savefig(fig, facecolor=fig.get_facecolor()); plt.close(fig)
        print("[PLOT 3] Returns Distribution ✓")
    except Exception as e:
        print(f"[WARNING] Plot 3 failed: {e}")


def plot4_trade_pnl(trades: pd.DataFrame, pdf: PdfPages) -> None:
    """Scatter plot of trade P&L with cumulative overlay."""
    try:
        if len(trades) == 0:
            print("[WARNING] Plot 4 skipped: no trades."); return
        fig, ax1 = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor("#0A0A0A"); ax1.set_facecolor("#111111")
        ax2 = ax1.twinx(); ax2.set_facecolor("#111111")

        trades2 = trades.copy()
        trades2["entry_date"] = pd.to_datetime(trades2["entry_date"])
        wins  = trades2["pnl_pct"] > 0
        sizes = np.clip(trades2["holding_days"] * 20, 10, 200)

        ax1.scatter(trades2.loc[wins,  "entry_date"], trades2.loc[wins,  "pnl_pct"],
                    c=GREEN_C, s=sizes[wins],  alpha=0.7, zorder=3, label="Win")
        ax1.scatter(trades2.loc[~wins, "entry_date"], trades2.loc[~wins, "pnl_pct"],
                    c=RED_C,   s=sizes[~wins], alpha=0.7, zorder=3, label="Loss")
        ax1.axhline(0, color="white", lw=0.5, alpha=0.3)
        ax1.set_ylabel("Trade P&L (%)", color="white")
        ax1.tick_params(colors="white")
        ax1.legend(loc="upper left", framealpha=0.2, facecolor="#1A1A1A")

        cum_pnl = trades2["pnl_pct"].cumsum()
        ax2.plot(trades2["entry_date"], cum_pnl, color=GOLD_C, lw=1.5, label="Cumulative P&L (%)")
        ax2.set_ylabel("Cumulative P&L (%)", color=GOLD_C)
        ax2.tick_params(colors=GOLD_C)
        ax2.legend(loc="upper right", framealpha=0.2, facecolor="#1A1A1A")

        ax1.set_title("TRADE-BY-TRADE P&L — Execution Quality", color=GOLD_C, fontweight="bold")
        ax1.set_xlabel("Entry Date", color="white")
        plt.tight_layout(pad=1.5)
        pdf.savefig(fig, facecolor=fig.get_facecolor()); plt.close(fig)
        print("[PLOT 4] Trade-by-Trade P&L ✓")
    except Exception as e:
        print(f"[WARNING] Plot 4 failed: {e}")


def plot5_rolling_metrics(result: pd.DataFrame, pdf: PdfPages) -> None:
    """4-panel rolling metrics: Sharpe, Sortino, Vol, Win Rate."""
    try:
        fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
        fig.patch.set_facecolor("#0A0A0A")
        for ax in axes: ax.set_facecolor("#111111")

        W   = 63
        ret = result["strategy_returns"]
        dates = result.index

        # (a) Rolling Sharpe
        rsh = ret.rolling(W).apply(
            lambda x: (x.mean() - RISK_FREE/TRADING_DAYS)/(x.std()+1e-9)*np.sqrt(TRADING_DAYS), raw=True)
        axes[0].plot(dates, rsh, color=ACCENT, lw=1.2)
        axes[0].axhline(1.0, color=GOLD_C, ls="--", lw=0.8)
        axes[0].axhline(0.0, color="white", ls="-",  lw=0.4, alpha=0.3)
        axes[0].fill_between(dates, rsh, 0, where=rsh>0, color=ACCENT, alpha=0.15)
        axes[0].fill_between(dates, rsh, 0, where=rsh<0, color=RED_C,  alpha=0.15)
        axes[0].set_ylabel("Sharpe", color="white"); axes[0].tick_params(colors="white")
        axes[0].set_title(f"ROLLING METRICS (63-day window)", color=GOLD_C, fontweight="bold")

        # (b) Rolling Sortino
        def roll_sortino(x):
            down = x[x < RISK_FREE/TRADING_DAYS]
            semi = down.std() * np.sqrt(TRADING_DAYS) if len(down) > 1 else 1e-9
            return (x.mean() - RISK_FREE/TRADING_DAYS) * TRADING_DAYS / (semi + 1e-9)
        rsort = ret.rolling(W).apply(roll_sortino, raw=False)
        axes[1].plot(dates, rsort, color=GREEN_C, lw=1.2)
        axes[1].axhline(1.0, color=GOLD_C, ls="--", lw=0.8)
        axes[1].axhline(0.0, color="white", ls="-",  lw=0.4, alpha=0.3)
        axes[1].set_ylabel("Sortino", color="white"); axes[1].tick_params(colors="white")

        # (c) Rolling Volatility
        rvol = ret.rolling(W).std() * np.sqrt(TRADING_DAYS) * 100
        axes[2].plot(dates, rvol, color=ACCENT2, lw=1.2)
        axes[2].set_ylabel("Ann. Vol (%)", color="white"); axes[2].tick_params(colors="white")

        # (d) Rolling Win Rate
        rwr = ret.rolling(W).apply(lambda x: (x > 0).mean() * 100, raw=True)
        axes[3].plot(dates, rwr, color=GOLD_C, lw=1.2)
        axes[3].axhline(50, color="white", ls="--", lw=0.8, alpha=0.6)
        axes[3].set_ylabel("Win Rate (%)", color="white"); axes[3].tick_params(colors="white")
        axes[3].set_xlabel("Date", color="white")

        plt.tight_layout(pad=1.5)
        pdf.savefig(fig, facecolor=fig.get_facecolor()); plt.close(fig)
        print("[PLOT 5] Rolling Metrics ✓")
    except Exception as e:
        print(f"[WARNING] Plot 5 failed: {e}")


def plot6_heatmaps(result: pd.DataFrame, pdf: PdfPages) -> None:
    """Monthly return heatmap and day-of-week bar chart."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor("#0A0A0A")
        ax1.set_facecolor("#111111"); ax2.set_facecolor("#111111")

        ret = result["strategy_returns"].copy()
        ret.index = pd.to_datetime(ret.index)

        # Monthly returns pivot
        monthly = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
        pivot   = monthly.copy()
        pivot_df = pd.DataFrame({
            "year" : pivot.index.year,
            "month": pivot.index.month,
            "ret"  : pivot.values
        }).pivot(index="year", columns="month", values="ret")
        pivot_df.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot_df.columns)]

        sns.heatmap(pivot_df, ax=ax1, cmap="RdYlGn", center=0,
                    annot=True, fmt=".1f", linewidths=0.3,
                    annot_kws={"size": 6}, cbar_kws={"shrink": 0.8})
        ax1.set_title("MONTHLY RETURNS HEATMAP (%)", color=GOLD_C, fontweight="bold")
        ax1.tick_params(colors="white"); ax1.set_xlabel("Month", color="white")
        ax1.set_ylabel("Year", color="white")

        # Day-of-week
        ret_dow = ret.copy()
        ret_dow = ret_dow.to_frame("ret")
        ret_dow["dow"] = ret_dow.index.day_name()
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
        avg_dow = ret_dow.groupby("dow")["ret"].mean() * 100
        avg_dow = avg_dow.reindex(dow_order)
        colors = [GREEN_C if v >= 0 else RED_C for v in avg_dow.values]
        ax2.bar(avg_dow.index, avg_dow.values, color=colors, alpha=0.85)
        ax2.axhline(0, color="white", lw=0.5, alpha=0.5)
        ax2.set_title("AVG RETURN BY DAY OF WEEK (%)", color=GOLD_C, fontweight="bold")
        ax2.set_xlabel("Day", color="white"); ax2.set_ylabel("Avg Return (%)", color="white")
        ax2.tick_params(colors="white")
        for i, (dow, val) in enumerate(avg_dow.items()):
            ax2.text(i, val + (0.001 if val >= 0 else -0.002),
                     f"{val:.3f}%", ha="center", va="bottom" if val >= 0 else "top",
                     color="white", fontsize=8)

        plt.tight_layout(pad=1.5)
        pdf.savefig(fig, facecolor=fig.get_facecolor()); plt.close(fig)
        print("[PLOT 6] Heatmaps ✓")
    except Exception as e:
        print(f"[WARNING] Plot 6 failed: {e}")


def plot7_risk_return(result: pd.DataFrame, pdf: PdfPages) -> None:
    """Rolling 252-day risk-return scatter with iso-Sharpe lines."""
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("#0A0A0A"); ax.set_facecolor("#111111")

        ret = result["strategy_returns"]
        brk = result["benchmark_returns"]
        W   = TRADING_DAYS

        roll_vol = ret.rolling(W).std()  * np.sqrt(W) * 100
        roll_ret = ret.rolling(W).mean() * W * 100
        roll_vol_b = brk.rolling(W).std()  * np.sqrt(W) * 100
        roll_ret_b = brk.rolling(W).mean() * W * 100

        mask = roll_vol.notna() & roll_ret.notna()
        vols = roll_vol[mask].values
        rets = roll_ret[mask].values
        n    = len(vols)
        cmap = plt.get_cmap("plasma")

        for i in range(n - 1):
            ax.plot(vols[i:i+2], rets[i:i+2], color=cmap(i / n), lw=1.2, alpha=0.7)
        sc = ax.scatter(vols, rets, c=np.arange(n), cmap="plasma", s=15, zorder=3, alpha=0.6)
        plt.colorbar(sc, ax=ax, label="Time (early → recent)").ax.yaxis.label.set_color("white")

        # Benchmark rolling
        mask_b = roll_vol_b.notna() & roll_ret_b.notna()
        ax.scatter(roll_vol_b[mask_b].mean(), roll_ret_b[mask_b].mean(),
                   color=ACCENT2, s=200, zorder=5, marker="*", label="Benchmark (avg)")
        ax.scatter(roll_vol[mask].mean(), roll_ret[mask].mean(),
                   color=GOLD_C, s=200, zorder=5, marker="D", label="Strategy (avg)")

        # Iso-Sharpe lines
        v_range = np.linspace(max(vols.min() - 2, 1), vols.max() + 2, 100)
        for sharpe_line, ls in [(0.5, ":"), (1.0, "--"), (1.5, "-.")]:
            r_line = RISK_FREE * 100 + sharpe_line * v_range
            ax.plot(v_range, r_line, color="white", lw=0.8, ls=ls, alpha=0.5,
                    label=f"Sharpe = {sharpe_line}")

        ax.set_xlabel("Annualized Volatility (%)", color="white")
        ax.set_ylabel("Annualized Return (%)",     color="white")
        ax.set_title("RISK-RETURN SCATTER (Rolling 252-day windows)", color=GOLD_C, fontweight="bold")
        ax.tick_params(colors="white")
        ax.legend(framealpha=0.2, facecolor="#1A1A1A", fontsize=8)
        plt.tight_layout(pad=1.5)
        pdf.savefig(fig, facecolor=fig.get_facecolor()); plt.close(fig)
        print("[PLOT 7] Risk-Return Scatter ✓")
    except Exception as e:
        print(f"[WARNING] Plot 7 failed: {e}")


def plot8_monte_carlo(result: pd.DataFrame, pdf: PdfPages) -> None:
    """Monte Carlo simulation with bootstrap resampling of returns."""
    try:
        N_SIM  = 1000
        ret    = result["strategy_returns"].dropna().values
        n_days = len(ret)

        sims = np.zeros((N_SIM, n_days))
        for i in range(N_SIM):
            sample   = np.random.choice(ret, size=n_days, replace=True)
            sims[i]  = INITIAL_CAPITAL * np.cumprod(1 + sample)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor("#0A0A0A")
        ax1.set_facecolor("#111111"); ax2.set_facecolor("#111111")

        days = np.arange(n_days)
        for i in range(N_SIM):
            ax1.plot(days, sims[i], color="cyan", alpha=0.03, lw=0.4)

        percentiles = [5, 25, 50, 75, 95]
        pct_labels  = ["P5", "P25", "P50", "P75", "P95"]
        pct_colors  = [RED_C, ACCENT2, "white", GREEN_C, ACCENT]
        for pct, label, col in zip(percentiles, pct_labels, pct_colors):
            pct_path = np.percentile(sims, pct, axis=0)
            ax1.plot(days, pct_path, color=col, lw=2, label=f"{label}: ₹{pct_path[-1]:,.0f}")

        # Actual strategy
        actual = result["portfolio_value"].values
        ax1.plot(days, actual, color=GOLD_C, lw=2.5, ls="--", label="Actual Strategy", zorder=5)

        ax1.set_title("MONTE CARLO — 1000 Bootstrap Simulations", color=GOLD_C, fontweight="bold")
        ax1.set_xlabel("Days", color="white"); ax1.set_ylabel("Portfolio Value (₹)", color="white")
        ax1.tick_params(colors="white")
        ax1.legend(framealpha=0.2, facecolor="#1A1A1A", fontsize=8)

        # Histogram of final values
        finals = sims[:, -1]
        ax2.hist(finals / 1e5, bins=60, color=ACCENT, alpha=0.7, edgecolor="none")
        for pct, label, col in zip(percentiles, pct_labels, pct_colors):
            val = np.percentile(finals, pct)
            ax2.axvline(val / 1e5, color=col, lw=1.5, ls="--", label=f"{label}: ₹{val/1e5:.1f}L")
        ax2.axvline(actual[-1] / 1e5, color=GOLD_C, lw=2.5, label=f"Actual: ₹{actual[-1]/1e5:.1f}L")
        ax2.set_title("DISTRIBUTION OF FINAL PORTFOLIO VALUES", color=GOLD_C, fontweight="bold")
        ax2.set_xlabel("Final Value (₹ Lakhs)", color="white")
        ax2.set_ylabel("Frequency", color="white")
        ax2.tick_params(colors="white")
        ax2.legend(framealpha=0.2, facecolor="#1A1A1A", fontsize=8)

        print(f"\n[MC] P5  Final : ₹{np.percentile(finals,5):>12,.0f}")
        print(f"[MC] P50 Final : ₹{np.percentile(finals,50):>12,.0f}")
        print(f"[MC] P95 Final : ₹{np.percentile(finals,95):>12,.0f}")

        plt.tight_layout(pad=1.5)
        pdf.savefig(fig, facecolor=fig.get_facecolor()); plt.close(fig)
        print("[PLOT 8] Monte Carlo ✓")
    except Exception as e:
        print(f"[WARNING] Plot 8 failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: MONTE CARLO & OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(result: pd.DataFrame, trades: pd.DataFrame) -> None:
    """Save trades log and equity curve to CSV files."""
    try:
        trades.to_csv(TRADES_CSV, index=False)
        print(f"[OUTPUT] Trades log saved  → {TRADES_CSV}")
    except Exception as e:
        print(f"[WARNING] Could not save trades CSV: {e}")

    try:
        dd  = drawdown_series(result["portfolio_value"].values)
        ret = result["strategy_returns"]
        rs  = ret.rolling(TRADING_DAYS).apply(
            lambda x: (x.mean() - RISK_FREE/TRADING_DAYS)/(x.std()+1e-9)*np.sqrt(TRADING_DAYS), raw=True)
        eq = pd.DataFrame({
            "date"            : result.index,
            "portfolio_value" : result["portfolio_value"].values,
            "benchmark_value" : result["benchmark_value"].values,
            "drawdown"        : dd,
            "rolling_sharpe"  : rs.values,
        })
        eq.to_csv(EQUITY_CSV, index=False)
        print(f"[OUTPUT] Equity curve saved → {EQUITY_CSV}")
    except Exception as e:
        print(f"[WARNING] Could not save equity CSV: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full Nifty 50 quantitative trading pipeline end-to-end."""
    print("\n" + "="*68)
    print("  NIFTY 50 QUANT SYSTEM — Initializing Pipeline")
    print("="*68 + "\n")

    # 1. Data
    df = download_data()

    # 2. Features
    print("[PIPELINE] Computing features ...")
    features = compute_features(df)
    print(f"[PIPELINE] Features shape: {features.shape}")

    # 3. Signals
    print("[PIPELINE] Generating signals ...")
    signals = generate_signals(features)
    sig_counts = signals["signal"].value_counts()
    print(f"[PIPELINE] Signal distribution:\n{sig_counts}\n")

    # 4. Backtest
    print("[PIPELINE] Running backtest ...")
    result, trades = run_backtest(signals)
    print(f"[PIPELINE] Trades executed: {len(trades)}")
    print(f"[PIPELINE] Final portfolio value: ₹{result['portfolio_value'].iloc[-1]:,.0f}")
    print(f"[PIPELINE] Final benchmark value: ₹{result['benchmark_value'].iloc[-1]:,.0f}\n")

    # 5. Metrics
    print("[PIPELINE] Computing performance metrics ...")
    metrics = compute_metrics(result, trades)
    print_metrics(metrics)

    # 6. Visualizations
    print("[PIPELINE] Generating plots → PDF ...")
    _style()
    with PdfPages(PDF_FILE) as pdf:
        plot1_equity_curve(result, pdf)
        plot2_drawdown(result, pdf)
        plot3_returns_dist(result, pdf)
        plot4_trade_pnl(trades, pdf)
        plot5_rolling_metrics(result, pdf)
        plot6_heatmaps(result, pdf)
        plot7_risk_return(result, pdf)
        plot8_monte_carlo(result, pdf)
    print(f"[OUTPUT] PDF saved → {PDF_FILE}")

    # 7. CSV outputs
    save_outputs(result, trades)

    print("\n" + "="*68)
    print("  PIPELINE COMPLETE ✓")
    print("="*68 + "\n")


if __name__ == "__main__":
    main()