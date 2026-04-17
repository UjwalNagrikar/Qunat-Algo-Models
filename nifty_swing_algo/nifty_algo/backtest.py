
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from config import (
    CAPITAL, LOT_SIZE, TIME_STOP_DAYS,
    BROKERAGE_PER_LOT, STT_PCT, EXCHANGE_PCT,
    SEBI_PCT, STAMP_DUTY_PCT, SLIPPAGE_POINTS
)
from risk import calculate_lots, can_afford
from strategy import calculate_stop_target


# ─────────────────────────────────────────────────────────────────────────────
# TRADE RECORD
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_date    : pd.Timestamp
    entry_price   : float
    direction     : str        # 'long' or 'short'
    lots          : int
    stop_price    : float
    target_price  : float
    regime        : str
    exit_date     : Optional[pd.Timestamp] = None
    exit_price    : Optional[float]        = None
    exit_reason   : Optional[str]          = None  # 'stop' | 'target' | 'time' | 'regime'
    pnl_points    : float = 0.0
    pnl_inr       : float = 0.0
    costs_inr     : float = 0.0
    net_pnl_inr   : float = 0.0
    bars_held     : int   = 0


# ─────────────────────────────────────────────────────────────────────────────
# COST MODEL
# ─────────────────────────────────────────────────────────────────────────────

def calculate_costs(entry_price: float,
                    exit_price: float,
                    lots: int,
                    lot_size: int = LOT_SIZE) -> float:
    """
    Full NSE futures cost model.
    
    Components:
    - Brokerage (flat per lot, both sides)
    - STT (sell side only, 0.01%)
    - Exchange transaction charge (both sides)
    - SEBI turnover fee
    - Stamp duty (buy side)
    - Slippage (assumed, in points)
    
    Returns:
        total_cost: float in ₹
    """
    contracts    = lots * lot_size
    entry_value  = entry_price * contracts
    exit_value   = exit_price  * contracts
    
    brokerage    = BROKERAGE_PER_LOT * lots * 2          # Both legs
    stt          = exit_value  * STT_PCT                  # Sell side only
    exchange     = (entry_value + exit_value) * EXCHANGE_PCT
    sebi         = (entry_value + exit_value) * SEBI_PCT
    stamp        = entry_value * STAMP_DUTY_PCT           # Buy side
    slippage_inr = SLIPPAGE_POINTS * contracts * 2        # Entry + exit
    
    total = brokerage + stt + exchange + sebi + stamp + slippage_inr
    return round(total, 2)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame,
                 starting_capital: float = CAPITAL) -> tuple[list[Trade], pd.Series]:
    """
    Event-driven backtest engine.
    
    Rules:
    - Only 1 position at a time
    - Entry on next day's open (avoid lookahead)
    - Exit checks: stop loss, profit target, time stop, regime change
    - Full cost model applied on each round trip
    
    Args:
        df: DataFrame with features + signals (output of generate_signals)
        starting_capital: initial capital in ₹
    
    Returns:
        trades       : list of Trade objects
        equity_curve : pd.Series of daily portfolio value
    """
    trades        = []
    equity        = starting_capital
    equity_curve  = {}
    
    current_trade: Optional[Trade] = None
    
    dates = df.index.tolist()
    
    for i in range(1, len(dates)):    # Start at 1 to use previous day signal
        today     = dates[i]
        yesterday = dates[i - 1]
        
        row_today     = df.loc[today]
        row_yesterday = df.loc[yesterday]
        
        # Entry price = today's open (signal from yesterday's close)
        entry_price = row_today["Open"]
        
        # ── MANAGE OPEN POSITION ──────────────────────────────────────────
        if current_trade is not None:
            current_trade.bars_held += 1
            
            high  = row_today["High"]
            low   = row_today["Low"]
            close = row_today["Close"]
            exit_price  = None
            exit_reason = None
            
            # Check stop loss (intraday — use High/Low)
            if current_trade.direction == "long":
                if low <= current_trade.stop_price:
                    exit_price  = current_trade.stop_price
                    exit_reason = "stop"
                elif high >= current_trade.target_price:
                    exit_price  = current_trade.target_price
                    exit_reason = "target"
            else:  # short
                if high >= current_trade.stop_price:
                    exit_price  = current_trade.stop_price
                    exit_reason = "stop"
                elif low <= current_trade.target_price:
                    exit_price  = current_trade.target_price
                    exit_reason = "target"
            
            # Time stop
            if exit_price is None and current_trade.bars_held >= TIME_STOP_DAYS:
                exit_price  = close
                exit_reason = "time"
            
            # Regime change stop (Hurst enters no-trade zone)
            if exit_price is None and row_today["regime"] == "no_trade":
                exit_price  = close
                exit_reason = "regime"
            
            # Close position if exit triggered
            if exit_price is not None:
                current_trade.exit_date  = today
                current_trade.exit_price = exit_price
                current_trade.exit_reason = exit_reason
                
                # P&L in points
                if current_trade.direction == "long":
                    current_trade.pnl_points = exit_price - current_trade.entry_price
                else:
                    current_trade.pnl_points = current_trade.entry_price - exit_price
                
                # P&L in ₹ (before costs)
                current_trade.pnl_inr = (
                    current_trade.pnl_points * current_trade.lots * LOT_SIZE
                )
                
                # Costs
                current_trade.costs_inr = calculate_costs(
                    current_trade.entry_price, exit_price, current_trade.lots
                )
                
                # Net P&L
                current_trade.net_pnl_inr = (
                    current_trade.pnl_inr - current_trade.costs_inr
                )
                
                equity += current_trade.net_pnl_inr
                trades.append(current_trade)
                current_trade = None
        
        # ── LOOK FOR NEW ENTRY ────────────────────────────────────────────
        if current_trade is None:
            # Use yesterday's signal (no lookahead)
            go_long  = row_yesterday["signal_long"]
            go_short = row_yesterday["signal_short"]
            direction = None
            
            if go_long:
                direction = "long"
            elif go_short:
                direction = "short"
            
            if direction and not row_today["high_vol_flag"]:
                realized_vol = row_today["realized_vol"]
                stop, target = calculate_stop_target(entry_price, realized_vol, direction)
                lots = calculate_lots(equity, entry_price, stop)
                
                if lots > 0 and can_afford(entry_price, lots, equity):
                    current_trade = Trade(
                        entry_date   = today,
                        entry_price  = entry_price,
                        direction    = direction,
                        lots         = lots,
                        stop_price   = stop,
                        target_price = target,
                        regime       = row_yesterday["regime"],
                    )
        
        equity_curve[today] = equity
    
    # Close any open trade at end of data
    if current_trade is not None:
        last_date  = dates[-1]
        last_close = df.loc[last_date, "Close"]
        if current_trade.direction == "long":
            current_trade.pnl_points = last_close - current_trade.entry_price
        else:
            current_trade.pnl_points = current_trade.entry_price - last_close
        current_trade.pnl_inr = current_trade.pnl_points * current_trade.lots * LOT_SIZE
        current_trade.costs_inr = calculate_costs(
            current_trade.entry_price, last_close, current_trade.lots
        )
        current_trade.net_pnl_inr = current_trade.pnl_inr - current_trade.costs_inr
        current_trade.exit_date   = last_date
        current_trade.exit_price  = last_close
        current_trade.exit_reason = "end_of_data"
        equity += current_trade.net_pnl_inr
        trades.append(current_trade)
    
    equity_series = pd.Series(equity_curve)
    print(f"[Backtest] Complete → {len(trades)} trades | "
          f"Final Capital: ₹{equity:,.0f}")
    
    return trades, equity_series


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    """Convert list of Trade objects to a clean DataFrame."""
    if not trades:
        return pd.DataFrame()
    
    records = []
    for t in trades:
        records.append({
            "entry_date"   : t.entry_date,
            "exit_date"    : t.exit_date,
            "direction"    : t.direction,
            "regime"       : t.regime,
            "entry_price"  : t.entry_price,
            "exit_price"   : t.exit_price,
            "stop_price"   : t.stop_price,
            "target_price" : t.target_price,
            "lots"         : t.lots,
            "bars_held"    : t.bars_held,
            "exit_reason"  : t.exit_reason,
            "pnl_points"   : round(t.pnl_points, 2),
            "pnl_inr"      : round(t.pnl_inr, 2),
            "costs_inr"    : round(t.costs_inr, 2),
            "net_pnl_inr"  : round(t.net_pnl_inr, 2),
        })
    
    return pd.DataFrame(records)
