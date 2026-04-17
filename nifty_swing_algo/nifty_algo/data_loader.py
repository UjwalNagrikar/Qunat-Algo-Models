
import pandas as pd
import numpy as np
import yfinance as yf
from config import TICKER, START_DATE, END_DATE


def generate_synthetic_nifty(start: str = START_DATE,
                              end: str = END_DATE) -> pd.DataFrame:
    """
    Generate realistic synthetic Nifty 50 futures data.
    Uses GBM with volatility clustering (regime switching).
    Starting price ~5000 (2010 level), growing to ~22000 (2024 level).
    
    For LIVE trading, replace with:
    - Zerodha Kite API historical data
    - NSE bhavcopy (free daily OHLCV)
    - Upstox / Angel One API
    """
    np.random.seed(42)
    dates = pd.bdate_range(start=start, end=end)
    n     = len(dates)
    
    # Parameters (calibrated to Nifty historical stats)
    S0      = 5000.0
    annual_drift = 0.13      # ~13% annual return
    base_vol     = 0.16      # 16% annual vol
    
    dt = 1 / 252
    mu = annual_drift * dt
    
    closes = [S0]
    vols   = []
    
    # Regime-switching volatility (GARCH-like)
    current_vol = base_vol
    for i in range(n - 1):
        # Volatility clustering: high vol tends to persist
        shock = np.random.randn()
        vol_shock = np.random.randn() * 0.02
        current_vol = np.clip(
            0.9 * current_vol + 0.1 * base_vol + vol_shock * current_vol,
            0.08, 0.45
        )
        vols.append(current_vol)
        
        # Occasional fat-tail shocks (market crashes/rallies)
        if np.random.rand() < 0.005:   # 0.5% chance of extreme day
            shock = np.random.choice([-1, 1]) * (3 + np.random.exponential(1))
        
        ret = mu + current_vol * np.sqrt(dt) * shock
        closes.append(closes[-1] * np.exp(ret))
    
    vols.append(vols[-1])
    closes = np.array(closes)
    vols   = np.array(vols)
    
    # Build OHLCV from closes
    daily_vol_pts = closes * vols * np.sqrt(dt)
    highs  = closes * (1 + np.abs(np.random.randn(n) * 0.003)) + daily_vol_pts * 0.3
    lows   = closes * (1 - np.abs(np.random.randn(n) * 0.003)) - daily_vol_pts * 0.3
    opens  = np.roll(closes, 1) * (1 + np.random.randn(n) * 0.002)
    opens[0] = S0
    
    # Clip so High >= Close >= Low
    highs = np.maximum(highs, closes)
    lows  = np.minimum(lows, closes)
    highs = np.maximum(highs, opens)
    lows  = np.minimum(lows, opens)
    
    # Volume: base + vol clustering
    base_volume = 5_00_000
    volumes = (base_volume * (0.5 + np.abs(np.random.randn(n))) *
               (1 + vols / base_vol)).astype(int)
    
    df = pd.DataFrame({
        "Open"  : np.round(opens, 2),
        "High"  : np.round(highs, 2),
        "Low"   : np.round(lows,  2),
        "Close" : np.round(closes, 2),
        "Volume": volumes,
    }, index=dates)
    df.index.name = "Date"
    
    print(f"[DataLoader] Generated synthetic Nifty data: {len(df)} days | "
          f"{df.index[0].date()} → {df.index[-1].date()} | "
          f"Start: {S0:.0f} → End: {closes[-1]:.0f}")
    print("[DataLoader] NOTE: Replace with real Zerodha Kite / NSE data for live trading!")
    
    return df


def download_data(ticker: str = TICKER,
                  start: str = START_DATE,
                  end: str = END_DATE) -> pd.DataFrame:
    """
    Download daily OHLCV data. Falls back to synthetic if unavailable.
    
    For live trading use:
    - Zerodha Kite API  
    - NSE bhavcopy
    - Upstox / Angel One API
    """
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


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log returns column."""
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def split_data(df: pd.DataFrame,
               train_end: str,
               val_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / validation / test sets.
    
    Returns:
        train, val, test DataFrames
    """
    train = df[df.index <= train_end].copy()
    val   = df[(df.index > train_end) & (df.index <= val_end)].copy()
    test  = df[df.index > val_end].copy()
    
    print(f"\n[DataLoader] Walk-Forward Split:")
    print(f"  Train : {train.index[0].date()} → {train.index[-1].date()} ({len(train)} days)")
    print(f"  Val   : {val.index[0].date()} → {val.index[-1].date()} ({len(val)} days)")
    print(f"  Test  : {test.index[0].date()} → {test.index[-1].date()} ({len(test)} days)")
    
    return train, val, test


if __name__ == "__main__":
    df = download_data()
    df = add_log_returns(df)
    print(df.tail())
