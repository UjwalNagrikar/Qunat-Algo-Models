import warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# ── NSE FUTSTK LOT SIZES ──────────────────────────────────────────────────────
# Simplified single lot-size per stock for full-period backtesting.
# Real NSE lot sizes are revised periodically; these represent representative
# values used across the 2010-2024 period for modelling purposes.
FUTSTK_LOT_SIZES = {
    "RELIANCE":    250,
    "TCS":         150,
    "HDFCBANK":    550,
    "INFY":        300,
    "ICICIBANK":  1375,
    "HINDUNILVR":  300,
    "ITC":        3200,
    "SBIN":       1500,
    "AXISBANK":   1200,
    "BHARTIARTL":  950,
    "LT":          150,
    "KOTAKBANK":   400,
}

PHASE_SPANS = {
    "Training":   ("2010-01-01", "2018-12-31"),
    "Validation": ("2019-01-01", "2021-12-31"),
    "Execution":  ("2022-01-01", "2024-12-31"),
}

# ── MASTER CONFIGURATION ──────────────────────────────────────────────────────
CFG = {
    "symbols": [
        "RELIANCE.NS", "TCS.NS",    "HDFCBANK.NS", "INFY.NS",
        "ICICIBANK.NS","HINDUNILVR.NS","ITC.NS",    "SBIN.NS",
        "AXISBANK.NS", "BHARTIARTL.NS","LT.NS",    "KOTAKBANK.NS",
    ],
    "full_start":      "2010-01-01",
    "full_end":        "2024-12-31",
    "initial_capital":  1_000_000,     # INR 10 Lakh

    # ── Hyperparameter grid (Training phase ONLY) ─────────────────────────
    "param_grid": {
        "holding_period": [3, 5, 7],       # days
        "z_threshold":    [1.0, 1.5, 2.0], # |z_composite| threshold
        "cooldown_days":  [3, 5],           # re-entry blackout
        "vol_threshold":  [0.025, 0.035],   # 10-day vol cap
    },

    # ── Fixed parameters (not grid-searched) ─────────────────────────────
    "long_pct":         0.25,   # bottom 25% by cs_rank → LONG
    "short_pct":        0.25,   # top 25% by cs_rank    → SHORT
    "max_positions":    6,      # max concurrent open contracts
    "margin_pct":       0.20,   # SPAN + Exposure margin rate
    "risk_pct":         0.030,  # 2% of equity = margin budget per trade
    "max_margin_pct":   0.12,   # hard cap: 12% of equity per position
    "use_trend_filter": True,   # only trade with trend direction
    "ma_window":        20,     # MA period for trend filter
    "vol_window":       10,     # rolling vol period
    "stop_loss_pct":    None,   # None = disabled (time-exit only)

    # ── NSE Futures Transaction Costs (Zerodha / market standard) ─────────
    # Round-trip all-in ≈ 0.07–0.09% of notional for large-cap FUTSTK
    "brokerage_flat":  20.0,      # INR 20/order flat
    "brokerage_pct":   0.0003,    # 0.03% cap
    "stt_sell_pct":    0.0001,    # 0.01% on sell side only (futures rate)
    "exchange_pct":    0.000019,  # NSE futures transaction charge
    "sebi_pct":        0.000001,  # SEBI turnover fee
    "stamp_buy_pct":   0.00002,   # 0.002% on buy (futures)
    "gst_pct":         0.18,      # 18% on brokerage + exchange
    "slippage_pct":    0.0002,    # 0.02%/side (tight spread, large-cap liquid)
}


# ==============================================================================
#  STEP 1  │  DATA LOADING
# ==============================================================================
def _try_yfinance(cfg: dict) -> list:
    frames = []
    for sym in cfg["symbols"]:
        try:
            raw = yf.download(
                sym, start=cfg["full_start"], end=cfg["full_end"],
                auto_adjust=True, progress=False, threads=False,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
            if raw.empty or len(raw) < 500:
                continue
            raw = raw[["open", "high", "low", "close", "volume"]].replace(0, np.nan)
            raw = raw.dropna(subset=["open", "close"])
            raw.index = pd.to_datetime(raw.index)
            raw.index.name = "date"
            raw["symbol"] = sym.replace(".NS", "")
            frames.append(raw)
        except Exception:
            pass
    return frames if len(frames) >= 8 else None


def _synthetic(cfg: dict) -> list:
    """
    Realistic NSE-price-level synthetic data with:
      • Shared market factor (GBM) modelling NSE Nifty behaviour
      • Per-stock idiosyncratic AR(1) with negative coefficient (−0.45 to −0.22)
        → creates short-horizon mean reversion (Jegadeesh 1990)
      • Historical stress overlays: taper tantrum 2013, demonetisation 2016,
        COVID crash 2020, post-2022 rate-hike volatility
    """
    np.random.seed(2010)
    dates = pd.bdate_range(cfg["full_start"], cfg["full_end"])
    T     = len(dates)
    syms  = [s.replace(".NS", "") for s in cfg["symbols"]]

    mkt = np.random.normal(0.00012, 0.008, T)
    for s, e, a in [
        (750, 790, -0.004),   # 2013 taper tantrum
        (1700, 1740, -0.003), # 2016 demonetisation
        (2580, 2640, -0.007), # 2020 COVID crash
        (2650, 2700,  0.005), # 2020 V-recovery
        (3000, 3040, -0.002), # 2022 rate hike
    ]:
        mkt[s:e] += a + np.random.normal(0, abs(a) * 0.4, e - s)

    seed_px = {
        "RELIANCE": 1050, "TCS": 750,      "HDFCBANK": 350,  "INFY": 650,
        "ICICIBANK": 430, "HINDUNILVR": 250,"ITC": 120,       "SBIN": 250,
        "AXISBANK": 260,  "BHARTIARTL": 320,"LT": 1600,       "KOTAKBANK": 380,
    }

    frames = []
    for sym in syms:
        s0    = seed_px.get(sym, 500)
        idvol = np.random.uniform(0.009, 0.018)
        beta  = np.random.uniform(0.60, 1.40)
        drift = np.random.uniform(-0.00008, 0.00030)
        ac    = np.random.uniform(-0.45, -0.22)  # negative AC → mean reversion
        idio  = np.zeros(T)
        for t in range(1, T):
            idio[t] = ac * idio[t - 1] + idvol * np.random.randn()
        ret   = mkt * beta + idio + drift
        close = s0 * np.exp(np.cumsum(ret))
        intra = np.abs(np.random.normal(0.008, 0.004, T)).clip(0.002, 0.05)
        open_ = np.concatenate([[s0], close[:-1]]) * np.exp(np.random.normal(0, 0.003, T))
        high  = np.maximum(open_, close) * (1 + intra * 0.6)
        low   = np.minimum(open_, close) * (1 - intra * 0.4)
        df    = pd.DataFrame({
            "open":   np.round(open_, 2),  "high":   np.round(high,  2),
            "low":    np.round(low,   2),  "close":  np.round(close, 2),
            "volume": np.random.lognormal(15, 0.9, T).astype(int),
            "symbol": sym,
        }, index=dates)
        df.index.name = "date"
        frames.append(df)
    return frames


def load_data(cfg: dict) -> tuple:
    frames = _try_yfinance(cfg)
    live   = frames is not None
    if not live:
        frames = _synthetic(cfg)
    data   = pd.concat(frames).reset_index().set_index(["date", "symbol"]).sort_index()
    data   = data.replace(0, np.nan).dropna(subset=["open", "close"])
    src    = "Yahoo Finance (live)" if live else "Synthetic (negative-AC AR1, NSE levels)"
    return data, src


# ==============================================================================
#  STEP 2  │  FEATURE ENGINEERING
# ==============================================================================
def build_features(data: pd.DataFrame, cfg: dict) -> dict:
    """
    All features computed on full 2010-2024 history.
    Cross-sectional z-scores use only information available at day t
    → no lookahead bias.
    """
    close   = data["close"].unstack("symbol").sort_index().ffill(limit=3)
    open_   = data["open"].unstack("symbol").sort_index().ffill(limit=3)

    ret1    = close.pct_change(1)
    ret3    = close.pct_change(3)
    vol10   = ret1.rolling(cfg["vol_window"]).std()
    ma_win  = close.rolling(cfg["ma_window"]).mean()

    # Cross-sectional z-score (vectorised: no Python row loop)
    def _csz(df: pd.DataFrame) -> pd.DataFrame:
        mu  = df.mean(axis=1)
        sig = df.std(axis=1).replace(0, np.nan)
        return df.sub(mu, axis=0).div(sig, axis=0)

    z1       = _csz(ret1)
    z3       = _csz(ret3)
    z_comp   = (z1 + z3) / 2.0           # composite signal
    cs_rank  = ret1.rank(axis=1, ascending=True, pct=True)   # [0, 1] pct-rank

    return {
        "close":   close,   "open":    open_,
        "vol10":   vol10,   "ma":      ma_win,
        "z_comp":  z_comp,  "cs_rank": cs_rank,
    }


# ==============================================================================
#  STEP 3  │  SIGNAL GENERATION
# ==============================================================================
def build_signals(features: dict, cfg: dict, s_date: str, e_date: str) -> tuple:
    """
    LONG  → bottom long_pct by cs_rank  AND  z_comp <= −z_threshold
    SHORT → top   short_pct by cs_rank  AND  z_comp >=  z_threshold
    Trend filter: LONG only above 20-day MA, SHORT only below 20-day MA.
    Signal on day t → executes at OPEN of day t+1  (no lookahead).
    """
    def _sl(df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[(df.index >= s_date) & (df.index <= e_date)].copy()

    rnk  = _sl(features["cs_rank"])
    zc   = _sl(features["z_comp"])
    v10  = _sl(features["vol10"])
    cl   = _sl(features["close"])
    ma   = _sl(features["ma"])

    lp   = cfg.get("long_pct",        0.25)
    sp   = cfg.get("short_pct",       0.25)
    vt   = cfg.get("vol_threshold",  0.025)
    zt   = cfg.get("z_threshold",      1.5)
    tf   = cfg.get("use_trend_filter", True)

    # Volatility filter — mask stocks with excessive vol
    hi_vol  = v10 > vt
    rnk_adj = rnk.where(~hi_vol)
    zc_adj  = zc.where(~hi_vol)

    # Signal conditions
    long_c  = (rnk_adj <= lp)       & (zc_adj <= -zt)
    short_c = (rnk_adj >= (1 - sp)) & (zc_adj >=  zt)

    # Optional: trend-alignment filter
    if tf:
        above_ma = (cl > ma).fillna(False)
        below_ma = (cl < ma).fillna(False)
        long_c   = long_c  & above_ma   # buy dips in uptrend
        short_c  = short_c & below_ma   # sell rallies in downtrend

    # Combine: +1 LONG, -1 SHORT, 0 flat
    signals = (
        long_c.fillna(False).astype(np.int8)
        - short_c.fillna(False).astype(np.int8)
    )
    ret_score = zc_adj.abs()   # trade priority: most extreme z first
    return signals, ret_score


# ==============================================================================
#  FUTURES COST MODEL
# ==============================================================================
def _rt_cost(ep: float, xp: float, lot_size: int, num_lots: int,
             direction: int, cfg: dict) -> float:
    """
    Full NSE FUTSTK round-trip transaction cost (INR).

    Components:
      Brokerage  : INR 20 flat or 0.03% of notional (lower of two)
      STT        : 0.01% on sell side only (futures rate, not delivery 0.1%)
      Exchange   : NSE futures transaction charge (0.0019%)
      SEBI       : 0.0001% turnover fee
      Stamp duty : 0.002% on buy side
      GST        : 18% on (brokerage + exchange)
      Slippage   : 0.02%/side — conservative for liquid NSE large-caps
    """
    t_en = ep * lot_size * num_lots     # entry notional
    t_ex = xp * lot_size * num_lots     # exit notional

    b_en = min(cfg["brokerage_flat"], t_en * cfg["brokerage_pct"])
    b_ex = min(cfg["brokerage_flat"], t_ex * cfg["brokerage_pct"])
    e_en = t_en * cfg["exchange_pct"]
    e_ex = t_ex * cfg["exchange_pct"]
    s_en = t_en * cfg["sebi_pct"]
    s_ex = t_ex * cfg["sebi_pct"]
    g_en = (b_en + e_en) * cfg["gst_pct"]
    g_ex = (b_ex + e_ex) * cfg["gst_pct"]

    # STT on sell leg only; stamp duty on buy leg only
    if direction == 1:   # LONG: entry = buy, exit = sell
        stt   = t_ex * cfg["stt_sell_pct"]
        stamp = t_en * cfg["stamp_buy_pct"]
    else:                # SHORT: entry = sell, exit = buy
        stt   = t_en * cfg["stt_sell_pct"]
        stamp = t_ex * cfg["stamp_buy_pct"]

    slippage = (t_en + t_ex) * cfg["slippage_pct"]

    return (b_en + b_ex + e_en + e_ex + s_en + s_ex
            + g_en + g_ex + stt + stamp + slippage)


# ==============================================================================
#  STEP 4  │  BACKTEST ENGINE  (Futures Mechanics)
# ==============================================================================
def _make_trade(tid, sym, ls, d, ed, xd, ep, xp, nl, hold, rsn, gp, cost, mg):
    net = gp - cost
    return {
        "trade_id":      tid,
        "symbol":        sym,
        "lot_size":      ls,
        "direction":     "LONG" if d == 1 else "SHORT",
        "entry_date":    ed,
        "exit_date":     xd,
        "entry_price":   round(ep, 2),
        "exit_price":    round(xp, 2),
        "num_lots":      nl,
        "holding_days":  hold,
        "exit_reason":   rsn,
        "gross_pnl":     round(gp,   2),
        "cost":          round(cost,  2),
        "net_pnl":       round(net,   2),
        "margin_used":   round(mg,    2),
        "ret_pct":       round((xp / ep - 1) * d * 100, 4),
        "ret_on_margin": round(net / mg * 100, 4) if mg > 0 else 0.0,
    }


def run_backtest(features: dict, signals: pd.DataFrame,
                 ret_score: pd.DataFrame, cfg: dict,
                 s_date: str, e_date: str) -> tuple:
    """
    Event-driven daily engine with full futures mechanics.

    Capital tracking:
      free_cap   : available capital for new margin deposits
      locked     : sum of margin deposits in open positions
      unrealised : mark-to-market on open positions at today's close
      equity     : free_cap + locked + unrealised

    Entry rule  : signal on day t → open at OPEN of day t+1
    Exit rule   : time-based at OPEN of day (entry + holding_period)
    Stop-loss   : optional (disabled by default)
    """
    def _sl(df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[(df.index >= s_date) & (df.index <= e_date)]

    op  = _sl(features["open"])
    cp  = _sl(features["close"])
    sig = _sl(signals)
    rs  = _sl(ret_score)

    dates    = op.index.tolist()
    free_cap = float(cfg["initial_capital"])

    positions  = {}   # sym → {i, ep, num_lots, lot_size, dir, margin}
    cooldown   = {}   # sym → index of last exit
    trades     = []
    equity_rec = []

    hold_per = int(cfg.get("holding_period",  3))
    sl_pct   = cfg.get("stop_loss_pct",    None)
    max_pos  = int(cfg.get("max_positions",   6))
    cd_days  = int(cfg.get("cooldown_days",   3))
    m_pct    = cfg.get("margin_pct",       0.20)
    r_pct    = cfg.get("risk_pct",         0.015)
    mm_pct   = cfg.get("max_margin_pct",   0.12)

    for i, date in enumerate(dates):
        dop = op.loc[date]
        dcp = cp.loc[date]

        # ── Mark-to-market ────────────────────────────────────────────────
        locked = unrealised = 0.0
        for sym, pos in positions.items():
            locked += pos["margin"]
            px = dcp.get(sym, np.nan)
            if not np.isnan(px):
                unrealised += ((px - pos["ep"])
                               * pos["lot_size"] * pos["num_lots"] * pos["dir"])

        cur_eq = free_cap + locked + unrealised
        equity_rec.append({"date": date, "equity": cur_eq, "n_open": len(positions)})

        # ── Optional stop-loss (disabled by default) ──────────────────────
        if sl_pct is not None:
            for sym in list(positions):
                pos = positions[sym]
                px  = dcp.get(sym, np.nan)
                if np.isnan(px):
                    continue
                tr = (px - pos["ep"]) / pos["ep"] * pos["dir"]
                if tr < -sl_pct:
                    positions.pop(sym)
                    gp   = (px - pos["ep"]) * pos["lot_size"] * pos["num_lots"] * pos["dir"]
                    cost = _rt_cost(pos["ep"], px, pos["lot_size"],
                                    pos["num_lots"], pos["dir"], cfg)
                    free_cap += pos["margin"] + gp - cost
                    cooldown[sym] = i
                    trades.append(_make_trade(
                        len(trades) + 1, sym, pos["lot_size"], pos["dir"],
                        dates[pos["i"]], date, pos["ep"], px,
                        pos["num_lots"], i - pos["i"], "STOP",
                        gp, cost, pos["margin"],
                    ))

        # ── Time exit (at OPEN of holding_period day) ─────────────────────
        for sym in [s for s, p in list(positions.items()) if i >= p["i"] + hold_per]:
            pos  = positions.pop(sym)
            xp   = dop.get(sym, pos["ep"])
            if np.isnan(xp) or xp == 0:
                xp = pos["ep"]
            gp   = (xp - pos["ep"]) * pos["lot_size"] * pos["num_lots"] * pos["dir"]
            cost = _rt_cost(pos["ep"], xp, pos["lot_size"],
                            pos["num_lots"], pos["dir"], cfg)
            free_cap += pos["margin"] + gp - cost
            cooldown[sym] = i
            trades.append(_make_trade(
                len(trades) + 1, sym, pos["lot_size"], pos["dir"],
                dates[pos["i"]], date, pos["ep"], xp,
                pos["num_lots"], i - pos["i"], "HOLD",
                gp, cost, pos["margin"],
            ))

        # ── Entry: yesterday's signal → today's OPEN ─────────────────────
        if i == 0:
            continue
        prev = dates[i - 1]
        if prev not in sig.index:
            continue

        sr    = sig.loc[prev]
        cands = sr[sr != 0]
        if cands.empty:
            continue

        # Prioritise by |z_composite| magnitude (most extreme first)
        if prev in rs.index:
            sc    = rs.loc[prev][cands.index].fillna(0)
            cands = cands.loc[sc.sort_values(ascending=False).index]

        for sym, direction in cands.items():
            if len(positions) >= max_pos:
                break
            if sym in positions:
                continue
            if i - cooldown.get(sym, -9999) < cd_days:
                continue

            ep = dop.get(sym, np.nan)
            if np.isnan(ep) or ep <= 0:
                continue

            # ── Futures-specific position sizing ─────────────────────────
            lot_size = FUTSTK_LOT_SIZES.get(sym, 500)
            mpl      = ep * lot_size * m_pct       # margin per lot (INR)
            if mpl <= 0:
                continue

            # Lot count from risk budget (capped by max_margin_pct)
            risk_bgt = cur_eq * r_pct
            max_bgt  = cur_eq * mm_pct
            alloc    = min(risk_bgt, max_bgt)
            num_lots = max(1, int(alloc / mpl))
            tot_mg   = num_lots * mpl

            # Enforce max_margin_pct hard cap
            if tot_mg > cur_eq * mm_pct:
                num_lots = max(1, int(cur_eq * mm_pct / mpl))
                tot_mg   = num_lots * mpl

            # Enforce available free capital
            if tot_mg > free_cap * 0.93:
                num_lots = max(1, int(free_cap * 0.90 / mpl))
                tot_mg   = num_lots * mpl
                if tot_mg > free_cap or num_lots < 1:
                    continue

            free_cap -= tot_mg
            positions[sym] = {
                "i":        i,
                "ep":       ep,
                "num_lots": num_lots,
                "lot_size": lot_size,
                "dir":      int(direction),
                "margin":   tot_mg,
            }

    # ── Force-close all remaining at last available close ─────────────────
    ld = dates[-1]
    lc = cp.loc[ld]
    for sym, pos in list(positions.items()):
        xp = lc.get(sym, pos["ep"])
        if np.isnan(xp) or xp == 0:
            xp = pos["ep"]
        gp   = (xp - pos["ep"]) * pos["lot_size"] * pos["num_lots"] * pos["dir"]
        cost = _rt_cost(pos["ep"], xp, pos["lot_size"],
                        pos["num_lots"], pos["dir"], cfg)
        free_cap += pos["margin"] + gp - cost
        trades.append(_make_trade(
            len(trades) + 1, sym, pos["lot_size"], pos["dir"],
            dates[pos["i"]], ld, pos["ep"], xp,
            pos["num_lots"], len(dates) - 1 - pos["i"], "EOD",
            gp, cost, pos["margin"],
        ))

    tdf = pd.DataFrame(trades)
    edf = pd.DataFrame(equity_rec).set_index("date")
    return tdf, edf


# ==============================================================================
#  STEP 5  │  PERFORMANCE METRICS
# ==============================================================================
def _max_consec(lst: list, val: int) -> int:
    best = cur = 0
    for x in lst:
        cur = cur + 1 if x == val else 0
        best = max(best, cur)
    return best


def compute_metrics(tdf: pd.DataFrame, edf: pd.DataFrame,
                    cfg: dict, phase: str = "") -> dict:
    init   = cfg["initial_capital"]
    eq     = edf["equity"]
    final  = eq.iloc[-1]
    nyrs   = (edf.index[-1] - edf.index[0]).days / 365.25

    rp     = (final / init - 1) * 100
    ri     = final - init
    cagr   = ((final / init) ** (1 / max(nyrs, 0.01)) - 1) * 100

    daily  = eq.pct_change().dropna()
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)
              if daily.std() > 0 else 0.0)
    dn     = daily[daily < 0]
    sortino= (daily.mean() / dn.std() * np.sqrt(252)
              if len(dn) > 1 else 0.0)
    rm     = eq.cummax()
    dds    = (eq - rm) / rm
    mdd    = dds.min() * 100
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0

    if tdf.empty:
        return {}

    win    = tdf[tdf["net_pnl"] > 0]
    lose   = tdf[tdf["net_pnl"] <= 0]
    wr     = len(win) / len(tdf) * 100
    gp_sum = win["net_pnl"].sum()  if len(win)  > 0 else 0.0
    gl_sum = abs(lose["net_pnl"].sum()) if len(lose) > 0 else 1e-9
    pf     = gp_sum / gl_sum
    exp_   = tdf["net_pnl"].mean()
    gt     = tdf["gross_pnl"].sum()
    ct     = tdf["cost"].sum()
    ctg    = abs(ct) / abs(gt) * 100 if gt != 0 else 999.0
    ws     = (tdf["net_pnl"] > 0).astype(int).tolist()
    lg     = tdf[tdf["direction"] == "LONG"]
    sh_    = tdf[tdf["direction"] == "SHORT"]
    stops  = (tdf[tdf["exit_reason"] == "STOP"]
              if "exit_reason" in tdf.columns else pd.DataFrame())
    nl     = tdf["num_lots"].mean() if "num_lots" in tdf.columns else 0.0
    rom    = tdf["ret_on_margin"].mean() if "ret_on_margin" in tdf.columns else 0.0

    return {
        "phase":               phase,
        "initial_capital":     init,
        "final_capital":       final,
        "total_ret_pct":       rp,
        "total_ret_inr":       ri,
        "cagr":                cagr,
        "sharpe":              sharpe,
        "sortino":             sortino,
        "calmar":              calmar,
        "max_dd":              mdd,
        "n_trades":            len(tdf),
        "n_long":              len(lg),
        "n_short":             len(sh_),
        "n_stops":             len(stops),
        "win_rate":            wr,
        "profit_factor":       pf,
        "expectancy":          exp_,
        "avg_win":             win["net_pnl"].mean()  if len(win)  > 0 else 0.0,
        "avg_loss":            lose["net_pnl"].mean() if len(lose) > 0 else 0.0,
        "avg_hold":            tdf["holding_days"].mean(),
        "max_consec_wins":     _max_consec(ws, 1),
        "max_consec_losses":   _max_consec(ws, 0),
        "gross_pnl":           gt,
        "total_costs":         ct,
        "cost_to_gross_pct":   ctg,
        "long_pnl":            lg["net_pnl"].sum(),
        "short_pnl":           sh_["net_pnl"].sum(),
        "avg_lots_per_trade":  nl,
        "avg_ret_on_margin":   rom,
        "eq_series":           eq,
        "dd_series":           dds,
    }


# ==============================================================================
#  STEP 6  │  GRID SEARCH  (Training period ONLY)
# ==============================================================================
def run_grid_search(features: dict, cfg: dict) -> tuple:
    """
    Exhaustive search over param_grid on Training 2010-2018.
    Objective: maximise Sharpe (primary), Calmar (secondary).
    Parameters are LOCKED after this step — no re-fitting ever.
    """
    pg     = cfg["param_grid"]
    keys   = list(pg.keys())
    combos = list(itertools.product(*[pg[k] for k in keys]))
    s, e   = PHASE_SPANS["Training"]
    results = []

    for vals in combos:
        params = {**cfg, **dict(zip(keys, vals))}
        params["holding_period"] = int(params["holding_period"])
        params["cooldown_days"]  = int(params["cooldown_days"])
        params["stop_loss_pct"]  = None   # never use stop-loss during grid search
        try:
            sigs, sc = build_signals(features, params, s, e)
            tdf, edf = run_backtest(features, sigs, sc, params, s, e)
            if tdf.empty or len(tdf) < 15:
                continue
            m = compute_metrics(tdf, edf, params, "Training")
            row = dict(zip(keys, vals))
            row.update({
                "sharpe":   round(m["sharpe"],           4),
                "calmar":   round(m["calmar"],           4),
                "cagr":     round(m["cagr"],             2),
                "max_dd":   round(m["max_dd"],           2),
                "pf":       round(m["profit_factor"],    4),
                "n_trades": m["n_trades"],
                "cost_pct": round(m["cost_to_gross_pct"],1),
            })
            results.append(row)
        except Exception:
            pass

    if not results:
        # Fallback defaults if grid search yields no valid results
        return (
            {**cfg, "holding_period": 3, "z_threshold": 1.5,
             "cooldown_days": 3, "vol_threshold": 0.025, "stop_loss_pct": None},
            pd.DataFrame(),
        )

    gdf  = (pd.DataFrame(results)
            .sort_values(["sharpe", "calmar"], ascending=False)
            .reset_index(drop=True))
    best = gdf.iloc[0]
    bp   = {**cfg}
    for k in keys:
        bp[k] = (int(best[k]) if k in ["holding_period", "cooldown_days"]
                 else float(best[k]))
    bp["stop_loss_pct"] = None
    return bp, gdf


# ==============================================================================
#  STEP 7  │  CLEAN PERFORMANCE OUTPUT  (no plots, no trade logs)
# ==============================================================================
def print_summary(pm: dict, bp: dict, gdf: pd.DataFrame,
                  cfg: dict, src: str, runtime: float) -> None:
    SEP  = "=" * 80
    SEP2 = "-" * 80

    phases = ["Training", "Validation", "Execution"]

    # Format helpers
    def fp(v):
        try:    return f"{v:>+.2f}%"
        except: return "N/A"

    def fi(v):
        try:    return f"INR {v:>+,.0f}"
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

    def g(ph, k):
        try:    return pm[ph][k]
        except: return None

    def prow(label, vals):
        v0, v1, v2 = [str(x) if x is not None else "N/A" for x in vals]
        print(f"  {label:<32}{v0:>15}{v1:>15}{v2:>15}")

    # ── Header ────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  NSE FUTSTK BACKTESTING ENGINE  v3.0  |  Universe Capital Quant Research")
    print(f"  Strategy  : Cross-Sectional Mean Reversion  |  Futures (FUTSTK) Mechanics")
    print(f"  Data      : {src}")
    print(f"  Universe  : {len(cfg['symbols'])} stocks  |  Capital: INR {cfg['initial_capital']:,.0f}")
    print(f"  Margin    : {cfg['margin_pct']*100:.0f}%  |  Risk/Trade: {cfg['risk_pct']*100:.1f}%  "
          f"|  Max Positions: {cfg['max_positions']}")
    print(SEP)

    # ── Optimal parameters ────────────────────────────────────────────────
    print()
    print(f"  OPTIMAL PARAMETERS  (grid-searched on Training 2010-2018, locked thereafter)")
    print(SEP2)
    print(f"  Holding Period   : {int(bp['holding_period'])} days     "
          f"  Z-Score Threshold : {bp['z_threshold']:.2f}σ")
    print(f"  Vol Filter       : {bp['vol_threshold']*100:.1f}%        "
          f"  Cooldown          : {int(bp['cooldown_days'])} days")
    print(f"  Trend Filter     : 20-day MA (LONG above MA, SHORT below MA)")
    print(f"  Stop-Loss        : Disabled — time-exit only (prevents premature exit on reversion)")
    if not gdf.empty:
        br = gdf.iloc[0]
        print(f"  Grid Best (Train): Sharpe={br['sharpe']:.4f}  Calmar={br['calmar']:.4f}  "
              f"CAGR={br['cagr']:.2f}%  MaxDD={br['max_dd']:.2f}%  "
              f"Trades={int(br['n_trades']):,}  Cost/Gross={br['cost_pct']:.1f}%")

    # ── Performance table ─────────────────────────────────────────────────
    print()
    print(SEP)
    print("  PERFORMANCE SUMMARY  —  Walk-Forward: Training → Validation → Execution")
    print(SEP)
    print(f"  {'Metric':<32}{'Training':>15}{'Validation':>15}{'Execution':>15}")
    print(f"  {'':32}{'2010–2018':>15}{'2019–2021':>15}{'2022–2024':>15}")
    print(f"  {SEP2}")

    # Returns & Risk
    prow("Total Return (%)",       [fp(g(p,"total_ret_pct")) for p in phases])
    prow("CAGR (%)",               [fp(g(p,"cagr"))          for p in phases])
    prow("Sharpe Ratio",           [f4(g(p,"sharpe"))         for p in phases])
    prow("Sortino Ratio",          [f4(g(p,"sortino"))        for p in phases])
    prow("Calmar Ratio",           [f4(g(p,"calmar"))         for p in phases])
    prow("Max Drawdown (%)",       [fp(g(p,"max_dd"))         for p in phases])

    print(f"  {SEP2}")

    # Trade statistics
    prow("Win Rate (%)",           [fp(g(p,"win_rate"))            for p in phases])
    prow("Initial Capital",       [fi(g(p,"initial_capital"))      for p in phases])
    prow("Final Capital",         [fi(g(p,"final_capital"))        for p in phases])
    prow("Profit Factor",          [f4(g(p,"profit_factor"))       for p in phases])
    prow("Expectancy / Trade",     [fi(g(p,"expectancy"))          for p in phases])
    prow("Avg Winner",             [fi(g(p,"avg_win"))             for p in phases])
    prow("Avg Loser",              [fi(g(p,"avg_loss"))            for p in phases])
    prow("Total Trades",           [fn(g(p,"n_trades"))            for p in phases])
    prow("  Long Trades",          [fn(g(p,"n_long"))              for p in phases])
    prow("  Short Trades",         [fn(g(p,"n_short"))             for p in phases])
    prow("  Stop-Loss Exits",      [fn(g(p,"n_stops"))             for p in phases])
    prow("Avg Hold (days)",        [f2(g(p,"avg_hold"))            for p in phases])
    prow("Max Consec. Wins",       [fn(g(p,"max_consec_wins"))     for p in phases])
    prow("Max Consec. Losses",     [fn(g(p,"max_consec_losses"))   for p in phases])

    print(f"  {SEP2}")
    print(f"  {'FUTURES PnL MECHANICS'}")
    print(f"  {SEP2}")

    prow("Gross PnL",              [fi(g(p,"gross_pnl"))          for p in phases])
    prow("Total Transaction Costs",[fi(g(p,"total_costs"))        for p in phases])
    prow("Net PnL",                [fi(g(p,"total_ret_inr"))      for p in phases])
    prow("Cost / Gross (%)",       [fcp(g(p,"cost_to_gross_pct")) for p in phases])
    prow("Long Net PnL",           [fi(g(p,"long_pnl"))           for p in phases])
    prow("Short Net PnL",          [fi(g(p,"short_pnl"))          for p in phases])
    prow("Avg Lots / Trade",       [f2(g(p,"avg_lots_per_trade")) for p in phases])
    prow("Avg Return on Margin (%)",[fcp(g(p,"avg_ret_on_margin")) for p in phases])

    # ── Consistency check ─────────────────────────────────────────────────
    print(f"  {SEP2}")
    print(f"  CONSISTENCY ANALYSIS  (walk-forward regime stability)")
    print(f"  {SEP2}")
    try:
        tr_s  = pm["Training"]["sharpe"]
        va_s  = pm["Validation"]["sharpe"]
        ex_s  = pm["Execution"]["sharpe"]
        deg   = (tr_s - va_s) / (abs(tr_s) + 1e-9) * 100
        tr_cp = pm["Training"]["cost_to_gross_pct"]
        va_cp = pm["Validation"]["cost_to_gross_pct"]
        ex_cp = pm["Execution"]["cost_to_gross_pct"]

        print(f"  Sharpe       : Train={tr_s:.4f}  →  Val={va_s:.4f}  →  Exec={ex_s:.4f}")
        print(f"  Degradation  : Train→Val = {deg:>+.1f}%  "
              f"({'< 30% — robust' if abs(deg) < 30 else '> 30% — monitor fit'})")
        print(f"  Cost / Gross : Train={tr_cp:.1f}%   Val={va_cp:.1f}%   Exec={ex_cp:.1f}%"
              f"  (viable if < 60%)")

        if deg < 30 and va_s > 0.5:
            verdict = "ROBUST     — Sharpe holds OOS within 30% of in-sample"
        elif va_s > 0:
            verdict = "ACCEPTABLE — Strategy profitable out-of-sample"
        else:
            verdict = "CAUTION    — Strategy not profitable OOS; revisit alpha"
        print(f"  Verdict      : {verdict}")
    except Exception:
        pass

    # ── Footer ────────────────────────────────────────────────────────────
    print()
    print(SEP)
    lots_str = "  ".join(f"{k}={v}" for k, v in FUTSTK_LOT_SIZES.items())
    print(f"  LOT SIZES  : {lots_str}")
    print(f"  COST MODEL : Brokerage(flat/%) + STT(0.01%sell) + Exchange + SEBI + GST + Slippage")
    print(f"  MECHANICS  : Integer lots | Margin-based capital | Futures PnL formula")
    print(f"  Runtime    : {runtime:.1f}s  |  No lookahead bias  |  Params locked after Training")
    print(SEP)
    print()


# ==============================================================================
#  MAIN  —  Walk-Forward Pipeline
# ==============================================================================
def main():
    t0  = datetime.now()
    cfg = CFG

    # 1. Load full history (2010-2024) once
    data, src = load_data(cfg)

    # 2. Build all features once on full history, then slice per phase
    features = build_features(data, cfg)

    # 3. Grid search on TRAINING ONLY — find best hyperparameters
    best_params, grid_df = run_grid_search(features, cfg)

    # 4. Run all three phases with LOCKED parameters (no re-fitting)
    phase_metrics = {}
    for phase in ["Training", "Validation", "Execution"]:
        s, e   = PHASE_SPANS[phase]
        p_cfg  = {**best_params, "initial_capital": cfg["initial_capital"]}
        sigs, sc = build_signals(features, p_cfg, s, e)
        tdf, edf = run_backtest(features, sigs, sc, p_cfg, s, e)
        phase_metrics[phase] = compute_metrics(tdf, edf, p_cfg, phase)

    # 5. Print clean performance summary — ONLY output
    elapsed = (datetime.now() - t0).total_seconds()
    print_summary(phase_metrics, best_params, grid_df, cfg, src, elapsed)


if __name__ == "__main__":
    main()