import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from dataclasses import dataclass, field
from typing import Optional
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

TICKER                  = "^NSEI"          # Yahoo Finance proxy
START_DATE              = "2010-01-01"
END_DATE                = "2024-12-31"

# Walk-Forward Splits
TRAIN_END               = "2018-12-31"
VAL_END                 = "2021-12-31"

# Futures Contract
LOT_SIZE                = 75               # Nifty 50 lot size
SPAN_MARGIN_PCT         = 0.065            # ~6.5% of contract value

# Feature Engineering
HURST_WINDOW            = 60               # Days for Hurst exponent calculation
BREAKOUT_WINDOW         = 20               # Rolling high/low lookback
ZSCORE_WINDOW           = 20               # Rolling z-score window
VOL_WINDOW              = 20               # Realized volatility window
VOLUME_ZSCORE_WIN       = 20               # Volume z-score window
RETURN_WINDOW_SHORT     = 3                # Short-term return window (days)
RETURN_WINDOW_MED       = 5                # Medium-term return window (days)

# Regime Filter
HURST_TREND_THRESH      = 0.55             # H > 0.55 → Trending regime
HURST_MR_THRESH         = 0.45             # H < 0.45 → Mean reversion regime
HIGH_VOL_MULTIPLIER     = 2.0              # Skip if realized vol > 2x historical avg

# Entry Signal — Trending Regime
TREND_BREAKOUT_WINDOW   = 20               # Breakout of N-day high/low
TREND_VOL_ZSCORE_MIN    = 0.8              # Minimum volume z-score for breakout confirm
TREND_RETURN_ZSCORE_MIN = 0.5              # Minimum 5-day return z-score for momentum

# Entry Signal — Mean Reversion Regime
MR_PRICE_ZSCORE_ENTRY   = 2.0              # Enter when price z-score crosses this
MR_VOL_ZSCORE_MAX       = 0.8              # Low volume confirms reversion (not panic)
MR_RETURN_3D_THRESH     = 0.03             # 3-day return must be > 3% for short entry

# Exit Rules
STOP_VOL_MULTIPLIER     = 1.5              # Stop = entry ± (1.5 × realized_vol × price)
TARGET_VOL_MULTIPLIER   = 3.0              # Target = entry ± (3.0 × realized_vol × price)
TIME_STOP_DAYS          = 7                # Exit if trade not working after 7 days

# Risk Management
CAPITAL                 = 1000000       # ₹10 Lakhs starting capital
RISK_PER_TRADE_PCT      = 0.10             # Risk 1% of capital per trade
MAX_OPEN_TRADES         = 1                # Only 1 position at a time (futures)

# Cost Model (NSE)
BROKERAGE_PER_LOT       = 40.0             # ₹20 each side (flat fee broker)
STT_PCT                 = 0.0001           # 0.01% on sell side (futures)
EXCHANGE_PCT            = 0.00002          # NSE transaction charge
SEBI_PCT                = 0.000001         # SEBI turnover fee
STAMP_DUTY_PCT          = 0.00002          # Stamp duty on buy
SLIPPAGE_POINTS         = 5.0              # Assumed slippage in Nifty points per trade


# =============================================================================
# 2. DATA LOADER
# =============================================================================

def generate_synthetic_nifty(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    
    S0 = 5000.0
    annual_drift = 0.13
    base_vol = 0.16
    dt = 1 / 252
    mu = annual_drift * dt
    closes = [S0]
    vols = []
    
    current_vol = base_vol
    for i in range(n - 1):
        shock = np.random.randn()
        vol_shock = np.random.randn() * 0.02
        current_vol = np.clip(0.9 * current_vol + 0.1 * base_vol + vol_shock * current_vol, 0.08, 0.45)
        vols.append(current_vol)
        
        if np.random.rand() < 0.005:
            shock = np.random.choice([-1, 1]) * (3 + np.random.exponential(1))
        
        ret = mu + current_vol * np.sqrt(dt) * shock
        closes.append(closes[-1] * np.exp(ret))
    
    vols.append(vols[-1])
    closes = np.array(closes)
    vols = np.array(vols)
    
    daily_vol_pts = closes * vols * np.sqrt(dt)
    highs = closes * (1 + np.abs(np.random.randn(n) * 0.003)) + daily_vol_pts * 0.3
    lows = closes * (1 - np.abs(np.random.randn(n) * 0.003)) - daily_vol_pts * 0.3
    opens = np.roll(closes, 1) * (1 + np.random.randn(n) * 0.002)
    opens[0] = S0
    
    highs = np.maximum(highs, closes)
    lows = np.minimum(lows, closes)
    highs = np.maximum(highs, opens)
    lows = np.minimum(lows, opens)
    
    base_volume = 5_00_000
    volumes = (base_volume * (0.5 + np.abs(np.random.randn(n))) * (1 + vols / base_vol)).astype(int)
    
    df = pd.DataFrame({
        "Open": np.round(opens, 2),
        "High": np.round(highs, 2),
        "Low": np.round(lows, 2),
        "Close": np.round(closes, 2),
        "Volume": volumes,
    }, index=dates)
    df.index.name = "Date"
    
    print(f"[DataLoader] Generated synthetic Nifty data: {len(df)} days")
    return df

def download_data(ticker: str = TICKER, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    print(f"[DataLoader] Attempting download {ticker} from {start} to {end}...")
    try:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        df.dropna(inplace=True)
        if len(df) > 100:
            print(f"[DataLoader] Downloaded {len(df)} days.")
            return df
    except Exception as e:
        print(f"[DataLoader] Download failed ({e}). Using synthetic data.")
    return generate_synthetic_nifty(start, end)

def split_data(df: pd.DataFrame, train_end: str, val_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df.index <= train_end].copy()
    val = df[(df.index > train_end) & (df.index <= val_end)].copy()
    test = df[df.index > val_end].copy()
    return train, val, test


# =============================================================================
# 3. FEATURES & INDICATORS
# =============================================================================

def hurst_rs(series: np.ndarray) -> float:
    n = len(series)
    if n < 20: return 0.5
    log_returns = np.diff(np.log(series))
    lags = [int(n / k) for k in [2, 4, 8, 16] if int(n / k) >= 8]
    if len(lags) < 2: return 0.5
    
    rs_values = []
    for lag in lags:
        rs_list = []
        for start in range(0, len(log_returns) - lag + 1, lag):
            subseries = log_returns[start:start + lag]
            mean_sub = np.mean(subseries)
            deviation = np.cumsum(subseries - mean_sub)
            R = deviation.max() - deviation.min()
            S = np.std(subseries, ddof=1)
            if S > 0: rs_list.append(R / S)
        if rs_list: rs_values.append(np.mean(rs_list))
    
    if len(rs_values) < 2: return 0.5
    log_lags = np.log(lags[:len(rs_values)])
    log_rs = np.log(rs_values)
    H = np.polyfit(log_lags, log_rs, 1)[0]
    return float(np.clip(H, 0.0, 1.0))

def rolling_hurst(prices: pd.Series, window: int = HURST_WINDOW) -> pd.Series:
    return prices.rolling(window).apply(lambda x: hurst_rs(x), raw=True)

def realized_volatility(log_returns: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    return log_returns.rolling(window).std() * np.sqrt(252)

def vol_regime(realized_vol: pd.Series, long_window: int = 120, multiplier: float = HIGH_VOL_MULTIPLIER) -> pd.Series:
    long_avg = realized_vol.rolling(long_window).mean()
    return realized_vol > (multiplier * long_avg)

def rolling_zscore(series: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)

def rolling_high(close: pd.Series, window: int = BREAKOUT_WINDOW) -> pd.Series:
    return close.shift(1).rolling(window).max()

def rolling_low(close: pd.Series, window: int = BREAKOUT_WINDOW) -> pd.Series:
    return close.shift(1).rolling(window).min()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("[Features] Building statistical features...")
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["daily_return"] = df["Close"].pct_change(1)
    df["return_3d"] = df["Close"].pct_change(RETURN_WINDOW_SHORT)
    df["return_5d"] = df["Close"].pct_change(RETURN_WINDOW_MED)
    df["realized_vol"] = realized_volatility(df["log_return"], VOL_WINDOW)
    df["high_vol_flag"] = vol_regime(df["realized_vol"])
    df["price_zscore"] = rolling_zscore(df["Close"], ZSCORE_WINDOW)
    df["return_zscore"] = rolling_zscore(df["log_return"], ZSCORE_WINDOW)
    df["return_5d_zscore"] = rolling_zscore(df["return_5d"], ZSCORE_WINDOW)
    df["vol_zscore"] = rolling_zscore(df["Volume"], VOLUME_ZSCORE_WIN)
    df["breakout_high"] = rolling_high(df["Close"], BREAKOUT_WINDOW)
    df["breakout_low"] = rolling_low(df["Close"], BREAKOUT_WINDOW)
    print("[Features] Calculating Hurst exponent (this takes ~30 seconds)...")
    df["hurst"] = rolling_hurst(df["Close"], HURST_WINDOW)
    df["vwap_dev"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).mean()
    
    initial_len = len(df)
    df.dropna(inplace=True)
    print(f"[Features] Done. Dropped {initial_len - len(df)} warmup rows. {len(df)} rows remaining.")
    return df


# =============================================================================
# 4. RISK & SIZING
# =============================================================================

def margin_check(entry_price: float, lot_size: int = LOT_SIZE, capital: float = CAPITAL) -> float:
    return (entry_price * lot_size) * SPAN_MARGIN_PCT

def calculate_lots(capital: float, entry_price: float, stop_price: float) -> int:
    risk_amount = capital * RISK_PER_TRADE_PCT
    stop_distance = abs(entry_price - stop_price)
    if stop_distance == 0: return 0
    risk_per_lot = stop_distance * LOT_SIZE
    lots = int(risk_amount / risk_per_lot)
    margin_required = margin_check(entry_price, LOT_SIZE, capital)
    max_lots_margin = int(capital / margin_required) if margin_required > 0 else 0
    return max(0, min(lots, max_lots_margin))

def can_afford(entry_price: float, lots: int, capital: float) -> bool:
    if lots <= 0: return False
    return capital >= (margin_check(entry_price, LOT_SIZE, capital) * lots)


# =============================================================================
# 5. STRATEGY (Signals)
# =============================================================================

def classify_regime(hurst: float) -> str:
    if hurst > HURST_TREND_THRESH: return "trending"
    elif hurst < HURST_MR_THRESH: return "mean_rev"
    return "no_trade"

def calculate_stop_target(entry_price: float, realized_vol: float, direction: str) -> tuple[float, float]:
    daily_vol = realized_vol / np.sqrt(252)
    stop_dist = STOP_VOL_MULTIPLIER * daily_vol * entry_price
    target_dist = TARGET_VOL_MULTIPLIER * daily_vol * entry_price
    if direction == "long":
        return round(entry_price - stop_dist, 2), round(entry_price + target_dist, 2)
    return round(entry_price + stop_dist, 2), round(entry_price - target_dist, 2)

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["regime"] = df["hurst"].apply(classify_regime)
    df["signal_long"] = False
    df["signal_short"] = False
    
    for i, row in df.iterrows():
        if row["high_vol_flag"]: continue
        
        if row["regime"] == "trending":
            df.at[i, "signal_long"] = (row["Close"] > row["breakout_high"] and 
                                       row["vol_zscore"] > TREND_VOL_ZSCORE_MIN and 
                                       row["return_5d_zscore"] > TREND_RETURN_ZSCORE_MIN and 
                                       row["daily_return"] > 0)
            df.at[i, "signal_short"] = (row["Close"] < row["breakout_low"] and 
                                        row["vol_zscore"] > TREND_VOL_ZSCORE_MIN and 
                                        row["return_5d_zscore"] < -TREND_RETURN_ZSCORE_MIN and 
                                        row["daily_return"] < 0)
        elif row["regime"] == "mean_rev":
            df.at[i, "signal_long"] = (row["price_zscore"] < -MR_PRICE_ZSCORE_ENTRY and 
                                       row["vol_zscore"] < MR_VOL_ZSCORE_MAX and 
                                       row["return_3d"] < -MR_RETURN_3D_THRESH)
            df.at[i, "signal_short"] = (row["price_zscore"] > MR_PRICE_ZSCORE_ENTRY and 
                                        row["vol_zscore"] < MR_VOL_ZSCORE_MAX and 
                                        row["return_3d"] > MR_RETURN_3D_THRESH)
            
    print(f"[Strategy] Signals → Long: {df['signal_long'].sum()} | Short: {df['signal_short'].sum()}")
    return df


# =============================================================================
# 6. BACKTEST ENGINE
# =============================================================================

@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    direction: str
    lots: int
    stop_price: float
    target_price: float
    regime: str
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_points: float = 0.0
    pnl_inr: float = 0.0
    costs_inr: float = 0.0
    net_pnl_inr: float = 0.0
    bars_held: int = 0

def calculate_costs(entry_price: float, exit_price: float, lots: int) -> float:
    contracts = lots * LOT_SIZE
    entry_value = entry_price * contracts
    exit_value = exit_price * contracts
    brokerage = BROKERAGE_PER_LOT * lots * 2
    stt = exit_value * STT_PCT
    exchange = (entry_value + exit_value) * EXCHANGE_PCT
    sebi = (entry_value + exit_value) * SEBI_PCT
    stamp = entry_value * STAMP_DUTY_PCT
    slippage_inr = SLIPPAGE_POINTS * contracts * 2
    return round(brokerage + stt + exchange + sebi + stamp + slippage_inr, 2)

def run_backtest(df: pd.DataFrame, starting_capital: float = CAPITAL) -> tuple[list[Trade], pd.Series]:
    trades = []
    equity = starting_capital
    equity_curve = {}
    current_trade: Optional[Trade] = None
    dates = df.index.tolist()
    
    for i in range(1, len(dates)):
        today = dates[i]
        yesterday = dates[i - 1]
        row_today = df.loc[today]
        row_yesterday = df.loc[yesterday]
        entry_price = row_today["Open"]
        
        if current_trade is not None:
            current_trade.bars_held += 1
            high, low, close = row_today["High"], row_today["Low"], row_today["Close"]
            exit_price, exit_reason = None, None
            
            if current_trade.direction == "long":
                if low <= current_trade.stop_price:
                    exit_price, exit_reason = current_trade.stop_price, "stop"
                elif high >= current_trade.target_price:
                    exit_price, exit_reason = current_trade.target_price, "target"
            else:
                if high >= current_trade.stop_price:
                    exit_price, exit_reason = current_trade.stop_price, "stop"
                elif low <= current_trade.target_price:
                    exit_price, exit_reason = current_trade.target_price, "target"
            
            if exit_price is None and current_trade.bars_held >= TIME_STOP_DAYS:
                exit_price, exit_reason = close, "time"
            if exit_price is None and row_today["regime"] == "no_trade":
                exit_price, exit_reason = close, "regime"
            
            if exit_price is not None:
                current_trade.exit_date = today
                current_trade.exit_price = exit_price
                current_trade.exit_reason = exit_reason
                
                if current_trade.direction == "long":
                    current_trade.pnl_points = exit_price - current_trade.entry_price
                else:
                    current_trade.pnl_points = current_trade.entry_price - exit_price
                    
                current_trade.pnl_inr = current_trade.pnl_points * current_trade.lots * LOT_SIZE
                current_trade.costs_inr = calculate_costs(current_trade.entry_price, exit_price, current_trade.lots)
                current_trade.net_pnl_inr = current_trade.pnl_inr - current_trade.costs_inr
                
                equity += current_trade.net_pnl_inr
                trades.append(current_trade)
                current_trade = None
                
        if current_trade is None:
            direction = "long" if row_yesterday["signal_long"] else "short" if row_yesterday["signal_short"] else None
            if direction and not row_today["high_vol_flag"]:
                stop, target = calculate_stop_target(entry_price, row_today["realized_vol"], direction)
                lots = calculate_lots(equity, entry_price, stop)
                if lots > 0 and can_afford(entry_price, lots, equity):
                    current_trade = Trade(
                        entry_date=today, entry_price=entry_price, direction=direction,
                        lots=lots, stop_price=stop, target_price=target, regime=row_yesterday["regime"]
                    )
        
        equity_curve[today] = equity
        
    if current_trade is not None:
        last_date = dates[-1]
        last_close = df.loc[last_date, "Close"]
        current_trade.pnl_points = (last_close - current_trade.entry_price) if current_trade.direction == "long" else (current_trade.entry_price - last_close)
        current_trade.pnl_inr = current_trade.pnl_points * current_trade.lots * LOT_SIZE
        current_trade.costs_inr = calculate_costs(current_trade.entry_price, last_close, current_trade.lots)
        current_trade.net_pnl_inr = current_trade.pnl_inr - current_trade.costs_inr
        current_trade.exit_date, current_trade.exit_price, current_trade.exit_reason = last_date, last_close, "end_of_data"
        equity += current_trade.net_pnl_inr
        trades.append(current_trade)
        
    print(f"[Backtest] Complete → {len(trades)} trades | Final Capital: ₹{equity:,.0f}")
    return trades, pd.Series(equity_curve)

def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    if not trades: return pd.DataFrame()
    return pd.DataFrame([{
        "entry_date": t.entry_date, "exit_date": t.exit_date, "direction": t.direction,
        "regime": t.regime, "entry_price": t.entry_price, "exit_price": t.exit_price,
        "stop_price": t.stop_price, "target_price": t.target_price, "lots": t.lots,
        "bars_held": t.bars_held, "exit_reason": t.exit_reason,
        "pnl_points": round(t.pnl_points, 2), "pnl_inr": round(t.pnl_inr, 2),
        "costs_inr": round(t.costs_inr, 2), "net_pnl_inr": round(t.net_pnl_inr, 2)
    } for t in trades])


# =============================================================================
# 7. METRICS
# =============================================================================

def calculate_metrics(trades_df: pd.DataFrame, equity_curve: pd.Series, starting_capital: float = CAPITAL) -> dict:
    if trades_df.empty or len(equity_curve) == 0: return {"error": "No trades to evaluate"}
    
    metrics = {}
    metrics["total_trades"] = len(trades_df)
    metrics["winning_trades"] = (trades_df["net_pnl_inr"] > 0).sum()
    metrics["losing_trades"] = (trades_df["net_pnl_inr"] <= 0).sum()
    metrics["win_rate"] = metrics["winning_trades"] / metrics["total_trades"]
    
    winners = trades_df[trades_df["net_pnl_inr"] > 0]["net_pnl_inr"]
    losers = trades_df[trades_df["net_pnl_inr"] <= 0]["net_pnl_inr"]
    
    metrics["avg_win"] = winners.mean() if len(winners) > 0 else 0
    metrics["avg_loss"] = losers.mean() if len(losers) > 0 else 0
    metrics["win_loss_ratio"] = abs(metrics["avg_win"] / metrics["avg_loss"]) if metrics["avg_loss"] != 0 else np.inf
    
    metrics["total_pnl"] = trades_df["net_pnl_inr"].sum()
    metrics["total_costs"] = trades_df["costs_inr"].sum()
    metrics["gross_pnl"] = trades_df["pnl_inr"].sum()
    
    gross_profit = winners.sum() if len(winners) > 0 else 0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 1
    metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    final_capital = equity_curve.iloc[-1]
    metrics["final_capital"] = final_capital
    metrics["total_return"] = (final_capital - starting_capital) / starting_capital
    
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    metrics["cagr"] = (final_capital / starting_capital) ** (1 / years) - 1 if years > 0 else 0
    
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    metrics["max_drawdown"] = drawdown.min()
    
    in_drawdown = (drawdown < 0)
    metrics["max_dd_duration_days"] = 0
    if in_drawdown.any():
        dd_periods, start = [], None
        for date, val in in_drawdown.items():
            if val and start is None: start = date
            elif not val and start is not None:
                dd_periods.append((date - start).days)
                start = None
        metrics["max_dd_duration_days"] = max(dd_periods) if dd_periods else 0
        
    daily_returns = equity_curve.pct_change().dropna()
    metrics["sharpe_ratio"] = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    downside = daily_returns[daily_returns < 0]
    metrics["sortino_ratio"] = (daily_returns.mean() * 252) / (downside.std() * np.sqrt(252)) if (len(downside) > 0 and downside.std() > 0) else 0
    
    metrics["avg_bars_held"] = trades_df["bars_held"].mean()
    metrics["exit_reasons"] = trades_df["exit_reason"].value_counts().to_dict()
    return metrics

def print_metrics(metrics: dict, label: str = "Performance Summary") -> None:
    print(f"\n{'═'*52}\n  {label}\n{'═'*52}")
    if "error" in metrics:
        print(f"  {metrics['error']}")
        return
    print(f"\n  TRADE STATISTICS\n  {'─'*40}")
    print(f"  Total Trades      : {metrics['total_trades']}")
    print(f"  Win Rate          : {metrics['win_rate']*100:.1f}% ({metrics['winning_trades']}W / {metrics['losing_trades']}L)")
    print(f"  Avg Win / Loss    : ₹{metrics['avg_win']:,.0f} / ₹{metrics['avg_loss']:,.0f}")
    print(f"  Profit Factor     : {metrics['profit_factor']:.2f}")
    print(f"\n  P&L SUMMARY\n  {'─'*40}")
    print(f"  Net P&L           : ₹{metrics['total_pnl']:,.0f} (Costs: ₹{metrics['total_costs']:,.0f})")
    print(f"  Final Capital     : ₹{metrics['final_capital']:,.0f}")
    print(f"  CAGR / Return     : {metrics['cagr']*100:.1f}% / {metrics['total_return']*100:.1f}%")
    print(f"\n  RISK METRICS\n  {'─'*40}")
    print(f"  Sharpe / Sortino  : {metrics['sharpe_ratio']:.2f} / {metrics['sortino_ratio']:.2f}")
    print(f"  Max Drawdown      : {metrics['max_drawdown']*100:.1f}% ({metrics['max_dd_duration_days']} days)")
    print(f"\n{'═'*52}\n")


# =============================================================================
# 8. PLOTTING & MAIN EXECUTION
# =============================================================================

def plot_results(df_signals: pd.DataFrame, trades_df: pd.DataFrame, equity_curve: pd.Series, metrics: dict, label: str = "Full Period") -> None:
    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(5, 1, figure=fig, height_ratios=[3, 2, 1.5, 1.5, 2], hspace=0.4)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]
    
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=9)
        for spine in ax.spines.values(): spine.set_color("#30363d")
            
    GOLD, GREEN, RED, BLUE, PURPLE, GRAY = "#f0b429", "#3fb950", "#f85149", "#58a6ff", "#bc8cff", "#8b949e"
    
    ax1 = axes[0]
    ax1.plot(df_signals.index, df_signals["Close"], color=BLUE, linewidth=0.8, alpha=0.8, label="Nifty Close")
    
    if not trades_df.empty:
        longs = trades_df[trades_df["direction"] == "long"]
        shorts = trades_df[trades_df["direction"] == "short"]
        wins = trades_df[trades_df["net_pnl_inr"] > 0]
        losses = trades_df[trades_df["net_pnl_inr"] <= 0]
        
        ax1.scatter(longs["entry_date"], [df_signals.loc[d, "Close"] if d in df_signals.index else np.nan for d in longs["entry_date"]], marker="^", color=GREEN, s=80, zorder=5, label="Long Entry")
        ax1.scatter(shorts["entry_date"], [df_signals.loc[d, "Close"] if d in df_signals.index else np.nan for d in shorts["entry_date"]], marker="v", color=RED, s=80, zorder=5, label="Short Entry")
        
        for _, t in wins.iterrows():
            if t["exit_date"] in df_signals.index: ax1.scatter(t["exit_date"], df_signals.loc[t["exit_date"], "Close"], marker="x", color=GREEN, s=60, zorder=5)
        for _, t in losses.iterrows():
            if t["exit_date"] in df_signals.index: ax1.scatter(t["exit_date"], df_signals.loc[t["exit_date"], "Close"], marker="x", color=RED, s=60, zorder=5)
            
    ax1.set_title(f"Nifty 50 — Entry/Exit Signals | {label}", color=GOLD, fontsize=13, fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor=GRAY)
    
    ax2 = axes[1]
    eq_norm = equity_curve / equity_curve.iloc[0] * 100
    bnh_norm = (df_signals["Close"].reindex(equity_curve.index).ffill() / df_signals["Close"].reindex(equity_curve.index).ffill().iloc[0] * 100)
    ax2.plot(eq_norm.index, eq_norm.values, color=GOLD, linewidth=1.5, label="Strategy")
    ax2.plot(bnh_norm.index, bnh_norm.values, color=GRAY, linewidth=1.0, linestyle="--", alpha=0.6, label="Buy & Hold")
    ax2.axhline(100, color="#30363d", linewidth=0.8)
    ax2.set_title("Equity Curve vs Buy & Hold (Normalized to 100)", color=GOLD, fontsize=11, fontweight="bold", pad=8)
    ax2.legend(loc="upper left", fontsize=8, facecolor="#21262d", edgecolor="#30363d", labelcolor=GRAY)
    
    ax3 = axes[2]
    ax3.plot(df_signals.index, df_signals["hurst"], color=PURPLE, linewidth=0.8, alpha=0.9)
    ax3.axhline(0.55, color=GREEN, linewidth=1.0, linestyle="--", alpha=0.7)
    ax3.axhline(0.45, color=RED, linewidth=1.0, linestyle="--", alpha=0.7)
    ax3.fill_between(df_signals.index, 0.45, 0.55, alpha=0.1, color=GRAY, label="No-trade zone")
    ax3.set_ylim(0.0, 1.0)
    ax3.set_title("Hurst Exponent (Market Regime)", color=GOLD, fontsize=11, fontweight="bold", pad=8)
    
    ax4 = axes[3]
    drawdown = (equity_curve - equity_curve.cummax()) / equity_curve.cummax() * 100
    ax4.fill_between(drawdown.index, drawdown.values, 0, color=RED, alpha=0.5)
    ax4.plot(drawdown.index, drawdown.values, color=RED, linewidth=0.8)
    ax4.set_title("Drawdown (%)", color=GOLD, fontsize=11, fontweight="bold", pad=8)
    
    ax5 = axes[4]
    if not trades_df.empty and "exit_date" in trades_df.columns:
        trades_df2 = trades_df.copy()
        trades_df2["exit_date"] = pd.to_datetime(trades_df2["exit_date"])
        trades_df2["month"] = trades_df2["exit_date"].dt.to_period("M")
        monthly_pnl = trades_df2.groupby("month")["net_pnl_inr"].sum()
        colors_bar = [GREEN if v >= 0 else RED for v in monthly_pnl.values]
        ax5.bar(range(len(monthly_pnl)), monthly_pnl.values, color=colors_bar, alpha=0.8, width=0.8)
        tick_positions = [i for i, p in enumerate(monthly_pnl.index) if p.month == 1]
        ax5.set_xticks(tick_positions)
        ax5.set_xticklabels([str(monthly_pnl.index[i].year) for i in tick_positions], color=GRAY, fontsize=8)
        ax5.axhline(0, color=GRAY, linewidth=0.6)
    ax5.set_title("Monthly Net P&L (₹)", color=GOLD, fontsize=11, fontweight="bold", pad=8)
    
    if "error" not in metrics:
        stats = f"CAGR: {metrics['cagr']*100:.1f}% | Sharpe: {metrics['sharpe_ratio']:.2f} | Max DD: {metrics['max_drawdown']*100:.1f}% | Win Rate: {metrics['win_rate']*100:.1f}% | Trades: {metrics['total_trades']}"
        fig.text(0.5, 0.01, stats, ha="center", fontsize=10, color=GOLD, bbox=dict(boxstyle="round,pad=0.4", facecolor="#21262d", edgecolor="#444c56"))
        
    plt.suptitle("UNiverse Capital | Nifty 50 Futures Swing Algo", color=GOLD, fontsize=15, fontweight="bold", y=0.99)
    filename = f"nifty_algo_results_{label.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[Chart] Saved → {filename}")
    return filename

def main():
    print("=" * 60)
    print("  UNiverse Capital | Nifty 50 Swing Algo")
    print("  Rule-Based | No Indicators | Statistical Edge")
    print("=" * 60)
    
    raw = download_data()
    feat = build_features(raw)
    signals = generate_signals(feat)
    train_sig, val_sig, test_sig = split_data(signals, TRAIN_END, VAL_END)
    
    results = {}
    for label, subset in [("Train (2010–2018)", train_sig), ("Val (2019–2021)", val_sig), ("Test (2022–2024)", test_sig), ("Full Period", signals)]:
        print(f"\n{'─'*50}\n  Running: {label}\n{'─'*50}")
        trades, equity = run_backtest(subset, starting_capital=CAPITAL)
        tdf = trades_to_df(trades)
        m = calculate_metrics(tdf, equity, CAPITAL)
        print_metrics(m, label)
        results[label] = (tdf, equity, m)
        
    filenames = []
    for label, (tdf, equity, m) in results.items():
        subset = {"Train (2010–2018)": train_sig, "Val (2019–2021)": val_sig, "Test (2022–2024)": test_sig, "Full Period": signals}[label]
        filenames.append(plot_results(subset, tdf, equity, m, label))
        
    print("\n[Done] All results generated.")
    print(f"Charts: {filenames}")

if __name__ == "__main__":
    main()