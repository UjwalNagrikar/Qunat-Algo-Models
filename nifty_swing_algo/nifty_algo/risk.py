# =============================================================================
# risk.py — Position sizing and margin checks
# UNiverse Capital | Nifty 50 Futures Swing Algo
# =============================================================================

import numpy as np
from config import (
    CAPITAL, RISK_PER_TRADE_PCT, LOT_SIZE, SPAN_MARGIN_PCT
)


def calculate_lots(capital: float,
                   entry_price: float,
                   stop_price: float,
                   risk_pct: float = RISK_PER_TRADE_PCT,
                   lot_size: int = LOT_SIZE) -> int:
    """
    Fixed Fractional Position Sizing.
    
    Risk amount = capital × risk_pct
    Points risk per lot = |entry - stop| × lot_size
    Lots = Risk amount / Points risk per lot
    
    Args:
        capital      : current portfolio capital
        entry_price  : price at entry (Nifty points)
        stop_price   : stop loss price
        risk_pct     : fraction of capital to risk (default 1%)
        lot_size     : futures lot size (default 75)
    
    Returns:
        lots: int (0 if not enough capital)
    """
    risk_amount      = capital * risk_pct
    stop_distance    = abs(entry_price - stop_price)
    
    if stop_distance == 0:
        return 0
    
    risk_per_lot = stop_distance * lot_size
    lots         = int(risk_amount / risk_per_lot)
    
    # Cap at what margin allows
    margin_required = margin_check(entry_price, lot_size, capital)
    max_lots_margin = int(capital / margin_required) if margin_required > 0 else 0
    
    lots = min(lots, max_lots_margin)
    
    return max(0, lots)


def margin_check(entry_price: float,
                 lot_size: int = LOT_SIZE,
                 capital: float = CAPITAL) -> float:
    """
    Estimate SPAN margin required per lot.
    SPAN margin ≈ 6.5% of contract value.
    
    Returns:
        margin_per_lot: float in ₹
    """
    contract_value = entry_price * lot_size
    margin_per_lot = contract_value * SPAN_MARGIN_PCT
    return margin_per_lot


def can_afford(entry_price: float,
               lots: int,
               capital: float,
               lot_size: int = LOT_SIZE) -> bool:
    """
    Check if we have enough capital to take `lots` positions.
    """
    if lots <= 0:
        return False
    required = margin_check(entry_price, lot_size, capital) * lots
    return capital >= required


def print_trade_sizing(capital: float,
                       entry_price: float,
                       stop_price: float) -> None:
    """Pretty print trade sizing calculation."""
    lots        = calculate_lots(capital, entry_price, stop_price)
    stop_dist   = abs(entry_price - stop_price)
    risk_amt    = capital * RISK_PER_TRADE_PCT
    margin      = margin_check(entry_price)
    
    print(f"\n{'─'*45}")
    print(f"  Capital         : ₹{capital:,.0f}")
    print(f"  Entry Price     : {entry_price:.1f}")
    print(f"  Stop Price      : {stop_price:.1f}")
    print(f"  Stop Distance   : {stop_dist:.1f} pts")
    print(f"  Risk Amount     : ₹{risk_amt:,.0f} ({RISK_PER_TRADE_PCT*100:.1f}%)")
    print(f"  Risk per Lot    : ₹{stop_dist * LOT_SIZE:,.0f}")
    print(f"  Lots to Trade   : {lots}")
    print(f"  Margin/Lot      : ₹{margin:,.0f}")
    print(f"  Total Margin    : ₹{margin * lots:,.0f}")
    print(f"  Contract Value  : ₹{entry_price * LOT_SIZE * lots:,.0f}")
    print(f"{'─'*45}")


if __name__ == "__main__":
    print_trade_sizing(
        capital=20_00_000,
        entry_price=22000,
        stop_price=21670
    )
