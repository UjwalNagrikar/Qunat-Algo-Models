
import pandas as pd
import numpy as np
from config import (
    HURST_WINDOW, ZSCORE_WINDOW, VOL_WINDOW,
    VOLUME_ZSCORE_WIN, RETURN_WINDOW_SHORT, RETURN_WINDOW_MED,
    BREAKOUT_WINDOW
)


# ─────────────────────────────────────────────────────────────────────────────
# HURST EXPONENT  (R/S Analysis)
# H > 0.55 → Trending (persistent)
# H < 0.45 → Mean Reverting (anti-persistent)
# H ≈ 0.50 → Random Walk (no edge)
# ─────────────────────────────────────────────────────────────────────────────

def hurst_rs(series: np.ndarray) -> float:
    """
    Calculate Hurst Exponent using Rescaled Range (R/S) method.
    
    Args:
        series: 1D array of prices (not returns)
    
    Returns:
        H: float between 0 and 1
    """
    n = len(series)
    if n < 20:
        return 0.5  # Not enough data → assume random walk
    
    log_returns = np.diff(np.log(series))
    
    # Use multiple sub-period lengths for robust estimate
    lags = [int(n / k) for k in [2, 4, 8, 16] if int(n / k) >= 8]
    if len(lags) < 2:
        return 0.5
    
    rs_values = []
    for lag in lags:
        rs_list = []
        for start in range(0, len(log_returns) - lag + 1, lag):
            subseries = log_returns[start:start + lag]
            mean_sub  = np.mean(subseries)
            deviation = np.cumsum(subseries - mean_sub)
            R = deviation.max() - deviation.min()
            S = np.std(subseries, ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append(np.mean(rs_list))
    
    if len(rs_values) < 2:
        return 0.5
    
    # Fit log(R/S) vs log(n) → slope = H
    log_lags = np.log(lags[:len(rs_values)])
    log_rs   = np.log(rs_values)
    H        = np.polyfit(log_lags, log_rs, 1)[0]
    
    return float(np.clip(H, 0.0, 1.0))


def rolling_hurst(prices: pd.Series, window: int = HURST_WINDOW) -> pd.Series:
    """Compute rolling Hurst exponent."""
    return prices.rolling(window).apply(
        lambda x: hurst_rs(x), raw=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# REALIZED VOLATILITY
# ─────────────────────────────────────────────────────────────────────────────

def realized_volatility(log_returns: pd.Series,
                        window: int = VOL_WINDOW) -> pd.Series:
    """
    Annualized realized volatility from log returns.
    vol = std(log_returns, window) × sqrt(252)
    """
    return log_returns.rolling(window).std() * np.sqrt(252)


def vol_regime(realized_vol: pd.Series,
               long_window: int = 120,
               multiplier: float = 2.0) -> pd.Series:
    """
    Returns True if current vol is too high (avoid trading).
    High vol = current vol > multiplier × long-run avg vol
    """
    long_avg = realized_vol.rolling(long_window).mean()
    return realized_vol > (multiplier * long_avg)


# ─────────────────────────────────────────────────────────────────────────────
# Z-SCORES
# ─────────────────────────────────────────────────────────────────────────────

def rolling_zscore(series: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    """
    Z-score of a series over rolling window.
    z = (x - rolling_mean) / rolling_std
    """
    mean = series.rolling(window).mean()
    std  = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def return_zscore(log_returns: pd.Series,
                  window: int = ZSCORE_WINDOW) -> pd.Series:
    """Z-score of log returns → is today's return statistically extreme?"""
    return rolling_zscore(log_returns, window)


def price_zscore(close: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    """Z-score of price level → how far is price from rolling mean?"""
    return rolling_zscore(close, window)


def volume_zscore(volume: pd.Series,
                  window: int = VOLUME_ZSCORE_WIN) -> pd.Series:
    """Z-score of volume → abnormal volume signals regime shift."""
    return rolling_zscore(volume, window)


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING RETURNS
# ─────────────────────────────────────────────────────────────────────────────

def rolling_return(close: pd.Series, n: int) -> pd.Series:
    """Simple N-day price return: (close_t - close_{t-n}) / close_{t-n}"""
    return close.pct_change(n)


# ─────────────────────────────────────────────────────────────────────────────
# BREAKOUT LEVELS
# ─────────────────────────────────────────────────────────────────────────────

def rolling_high(close: pd.Series, window: int = BREAKOUT_WINDOW) -> pd.Series:
    """Rolling N-day highest close (shifted to avoid lookahead)."""
    return close.shift(1).rolling(window).max()


def rolling_low(close: pd.Series, window: int = BREAKOUT_WINDOW) -> pd.Series:
    """Rolling N-day lowest close (shifted to avoid lookahead)."""
    return close.shift(1).rolling(window).min()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features on a OHLCV DataFrame.
    Returns original df with all feature columns appended.
    No lookahead bias — all rolling calcs use shift(1) where needed.
    """
    df = df.copy()
    
    print("[Features] Building statistical features...")
    
    # ── Log Returns ──────────────────────────────────────────────────────────
    df["log_return"]    = np.log(df["Close"] / df["Close"].shift(1))
    df["daily_return"]  = df["Close"].pct_change(1)
    
    # ── Rolling Returns ───────────────────────────────────────────────────────
    df["return_3d"]     = rolling_return(df["Close"], RETURN_WINDOW_SHORT)
    df["return_5d"]     = rolling_return(df["Close"], RETURN_WINDOW_MED)
    
    # ── Realized Volatility ───────────────────────────────────────────────────
    df["realized_vol"]  = realized_volatility(df["log_return"], VOL_WINDOW)
    df["high_vol_flag"] = vol_regime(df["realized_vol"])
    
    # ── Z-Scores ──────────────────────────────────────────────────────────────
    df["price_zscore"]  = price_zscore(df["Close"], ZSCORE_WINDOW)
    df["return_zscore"] = return_zscore(df["log_return"], ZSCORE_WINDOW)
    df["return_5d_zscore"] = rolling_zscore(df["return_5d"], ZSCORE_WINDOW)
    df["vol_zscore"]    = volume_zscore(df["Volume"], VOLUME_ZSCORE_WIN)
    
    # ── Breakout Levels ───────────────────────────────────────────────────────
    df["breakout_high"] = rolling_high(df["Close"], BREAKOUT_WINDOW)
    df["breakout_low"]  = rolling_low(df["Close"], BREAKOUT_WINDOW)
    
    # ── Hurst Exponent ────────────────────────────────────────────────────────
    print("[Features] Calculating Hurst exponent (this takes ~30 seconds)...")
    df["hurst"]         = rolling_hurst(df["Close"], HURST_WINDOW)
    
    # ── VWAP Deviation ────────────────────────────────────────────────────────
    # Simple proxy: close vs 20-day avg close
    df["vwap_dev"]      = (df["Close"] - df["Close"].rolling(20).mean()) / \
                           df["Close"].rolling(20).mean()
    
    # Drop rows with NaN (from rolling windows)
    initial_len = len(df)
    df.dropna(inplace=True)
    print(f"[Features] Done. {initial_len - len(df)} warmup rows dropped. "
          f"{len(df)} rows remaining.")
    
    return df


if __name__ == "__main__":
    from data_loader import download_data
    raw = download_data()
    feat = build_features(raw)
    print(feat[["Close", "hurst", "price_zscore", "realized_vol",
                "breakout_high", "breakout_low"]].tail(10))
