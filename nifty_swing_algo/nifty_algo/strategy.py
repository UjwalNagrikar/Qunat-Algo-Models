# =============================================================================
# strategy.py — Regime filter + Entry/Exit signal generation
# UNiverse Capital | Nifty 50 Futures Swing Algo
# =============================================================================

import pandas as pd
import numpy as np
from config import (
    HURST_TREND_THRESH, HURST_MR_THRESH,
    TREND_BREAKOUT_WINDOW, TREND_VOL_ZSCORE_MIN, TREND_RETURN_ZSCORE_MIN,
    MR_PRICE_ZSCORE_ENTRY, MR_VOL_ZSCORE_MAX, MR_RETURN_3D_THRESH,
    STOP_VOL_MULTIPLIER, TARGET_VOL_MULTIPLIER, TIME_STOP_DAYS
)


# ─────────────────────────────────────────────────────────────────────────────
# REGIME CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def classify_regime(hurst: float) -> str:
    """
    Classify market regime from Hurst exponent.
    
    Returns:
        'trending'     → H > HURST_TREND_THRESH (trade breakouts)
        'mean_rev'     → H < HURST_MR_THRESH    (trade reversions)
        'no_trade'     → In between             (stay flat)
    """
    if hurst > HURST_TREND_THRESH:
        return "trending"
    elif hurst < HURST_MR_THRESH:
        return "mean_rev"
    else:
        return "no_trade"


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

def trend_long_signal(row: pd.Series) -> bool:
    """
    Trending regime LONG entry.
    ALL conditions must be True:
    1. Price breaks above rolling 20-day high (momentum breakout)
    2. Volume z-score > threshold (volume confirms breakout)
    3. 5-day return z-score > threshold (momentum building)
    4. Today's close is positive (price closed up)
    """
    return (
        row["Close"] > row["breakout_high"]
        and row["vol_zscore"]       > TREND_VOL_ZSCORE_MIN
        and row["return_5d_zscore"] > TREND_RETURN_ZSCORE_MIN
        and row["daily_return"]     > 0
    )


def trend_short_signal(row: pd.Series) -> bool:
    """
    Trending regime SHORT entry. Mirror of long conditions.
    """
    return (
        row["Close"] < row["breakout_low"]
        and row["vol_zscore"]       > TREND_VOL_ZSCORE_MIN
        and row["return_5d_zscore"] < -TREND_RETURN_ZSCORE_MIN
        and row["daily_return"]     < 0
    )


def mr_long_signal(row: pd.Series) -> bool:
    """
    Mean reversion regime LONG entry.
    Price has fallen too far below mean → expect reversion UP.
    1. Price z-score deeply negative (oversold)
    2. Low volume (not a panic breakdown — just mean reversion)
    3. 3-day return is very negative (sufficient pullback)
    """
    return (
        row["price_zscore"] < -MR_PRICE_ZSCORE_ENTRY
        and row["vol_zscore"]  < MR_VOL_ZSCORE_MAX
        and row["return_3d"]   < -MR_RETURN_3D_THRESH
    )


def mr_short_signal(row: pd.Series) -> bool:
    """
    Mean reversion regime SHORT entry.
    Price has risen too far above mean → expect reversion DOWN.
    """
    return (
        row["price_zscore"] > MR_PRICE_ZSCORE_ENTRY
        and row["vol_zscore"]  < MR_VOL_ZSCORE_MAX
        and row["return_3d"]   > MR_RETURN_3D_THRESH
    )


# ─────────────────────────────────────────────────────────────────────────────
# STOP LOSS & TARGET CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

def calculate_stop_target(entry_price: float,
                           realized_vol: float,
                           direction: str) -> tuple[float, float]:
    """
    Calculate stop loss and profit target based on realized volatility.
    
    Stop distance   = STOP_VOL_MULTIPLIER   × daily_vol × entry_price
    Target distance = TARGET_VOL_MULTIPLIER × daily_vol × entry_price
    
    Args:
        entry_price  : price at entry
        realized_vol : annualized realized vol (e.g. 0.18 = 18%)
        direction    : 'long' or 'short'
    
    Returns:
        (stop_price, target_price)
    """
    daily_vol    = realized_vol / np.sqrt(252)  # Convert annualized → daily
    stop_dist    = STOP_VOL_MULTIPLIER  * daily_vol * entry_price
    target_dist  = TARGET_VOL_MULTIPLIER * daily_vol * entry_price
    
    if direction == "long":
        stop   = entry_price - stop_dist
        target = entry_price + target_dist
    else:
        stop   = entry_price + stop_dist
        target = entry_price - target_dist
    
    return round(stop, 2), round(target, 2)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATOR (Full Row-by-Row Logic)
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate regime classification and entry signals for each row.
    
    Adds columns:
        regime        : 'trending' | 'mean_rev' | 'no_trade'
        signal_long   : True if long entry conditions met
        signal_short  : True if short entry conditions met
    
    No position management here — that's in backtest.py
    """
    df = df.copy()
    
    df["regime"] = df["hurst"].apply(classify_regime)
    
    df["signal_long"]  = False
    df["signal_short"] = False
    
    for i, row in df.iterrows():
        # Skip high volatility days
        if row["high_vol_flag"]:
            continue
        
        if row["regime"] == "trending":
            df.at[i, "signal_long"]  = trend_long_signal(row)
            df.at[i, "signal_short"] = trend_short_signal(row)
        
        elif row["regime"] == "mean_rev":
            df.at[i, "signal_long"]  = mr_long_signal(row)
            df.at[i, "signal_short"] = mr_short_signal(row)
        
        # "no_trade" → both stay False
    
    long_count  = df["signal_long"].sum()
    short_count = df["signal_short"].sum()
    print(f"[Strategy] Signals → Long: {long_count} | Short: {short_count} "
          f"| Total: {long_count + short_count}")
    
    return df


if __name__ == "__main__":
    from data_loader import download_data
    from features import build_features
    
    raw  = download_data()
    feat = build_features(raw)
    sig  = generate_signals(feat)
    
    print("\nRegime distribution:")
    print(sig["regime"].value_counts())
    print("\nSample signals:")
    print(sig[sig["signal_long"] | sig["signal_short"]][
        ["Close", "hurst", "regime", "price_zscore", "signal_long", "signal_short"]
    ].head(10))
