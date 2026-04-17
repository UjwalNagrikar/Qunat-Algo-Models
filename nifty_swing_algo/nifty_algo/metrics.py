# =============================================================================
# metrics.py — Performance metrics
# UNiverse Capital | Nifty 50 Futures Swing Algo
# =============================================================================

import pandas as pd
import numpy as np
from config import CAPITAL


def calculate_metrics(trades_df: pd.DataFrame,
                       equity_curve: pd.Series,
                       starting_capital: float = CAPITAL) -> dict:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        trades_df       : DataFrame from backtest.trades_to_df()
        equity_curve    : pd.Series of daily portfolio values
        starting_capital: initial capital
    
    Returns:
        dict of metrics
    """
    if trades_df.empty or len(equity_curve) == 0:
        return {"error": "No trades to evaluate"}
    
    metrics = {}
    
    # ── Basic Trade Stats ──────────────────────────────────────────────────
    metrics["total_trades"]   = len(trades_df)
    metrics["winning_trades"] = (trades_df["net_pnl_inr"] > 0).sum()
    metrics["losing_trades"]  = (trades_df["net_pnl_inr"] <= 0).sum()
    metrics["win_rate"]       = metrics["winning_trades"] / metrics["total_trades"]
    
    # ── P&L Stats ─────────────────────────────────────────────────────────
    winners = trades_df[trades_df["net_pnl_inr"] > 0]["net_pnl_inr"]
    losers  = trades_df[trades_df["net_pnl_inr"] <= 0]["net_pnl_inr"]
    
    metrics["avg_win"]        = winners.mean() if len(winners) > 0 else 0
    metrics["avg_loss"]       = losers.mean()  if len(losers)  > 0 else 0
    metrics["win_loss_ratio"] = abs(metrics["avg_win"] / metrics["avg_loss"]) \
                                if metrics["avg_loss"] != 0 else np.inf
    
    metrics["total_pnl"]      = trades_df["net_pnl_inr"].sum()
    metrics["total_costs"]    = trades_df["costs_inr"].sum()
    metrics["gross_pnl"]      = trades_df["pnl_inr"].sum()
    
    # ── Profit Factor ─────────────────────────────────────────────────────
    gross_profit = winners.sum() if len(winners) > 0 else 0
    gross_loss   = abs(losers.sum()) if len(losers) > 0 else 1
    metrics["profit_factor"]  = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # ── Equity Curve Stats ─────────────────────────────────────────────────
    final_capital             = equity_curve.iloc[-1]
    metrics["final_capital"]  = final_capital
    metrics["total_return"]   = (final_capital - starting_capital) / starting_capital
    
    # CAGR
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    metrics["cagr"]           = (final_capital / starting_capital) ** (1 / years) - 1 \
                                if years > 0 else 0
    
    # ── Drawdown ──────────────────────────────────────────────────────────
    rolling_max                  = equity_curve.cummax()
    drawdown                     = (equity_curve - rolling_max) / rolling_max
    metrics["max_drawdown"]      = drawdown.min()
    metrics["avg_drawdown"]      = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
    
    # Drawdown duration
    in_drawdown = (drawdown < 0)
    if in_drawdown.any():
        dd_periods = []
        start = None
        for date, val in in_drawdown.items():
            if val and start is None:
                start = date
            elif not val and start is not None:
                dd_periods.append((date - start).days)
                start = None
        metrics["max_dd_duration_days"] = max(dd_periods) if dd_periods else 0
    else:
        metrics["max_dd_duration_days"] = 0
    
    # ── Sharpe Ratio ──────────────────────────────────────────────────────
    daily_returns = equity_curve.pct_change().dropna()
    if daily_returns.std() > 0:
        metrics["sharpe_ratio"] = (daily_returns.mean() * 252) / \
                                   (daily_returns.std() * np.sqrt(252))
    else:
        metrics["sharpe_ratio"] = 0
    
    # ── Sortino Ratio ─────────────────────────────────────────────────────
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        metrics["sortino_ratio"] = (daily_returns.mean() * 252) / \
                                    (downside.std() * np.sqrt(252))
    else:
        metrics["sortino_ratio"] = 0
    
    # ── Trade Duration ─────────────────────────────────────────────────────
    metrics["avg_bars_held"]  = trades_df["bars_held"].mean()
    metrics["max_bars_held"]  = trades_df["bars_held"].max()
    
    # ── Exit Reason Breakdown ──────────────────────────────────────────────
    metrics["exit_reasons"]   = trades_df["exit_reason"].value_counts().to_dict()
    
    # ── Regime Breakdown ──────────────────────────────────────────────────
    metrics["regime_breakdown"] = trades_df.groupby("regime")["net_pnl_inr"].agg(
        ["count", "sum", "mean"]
    ).to_dict()
    
    return metrics


def print_metrics(metrics: dict, label: str = "Performance Summary") -> None:
    """Pretty print all performance metrics."""
    print(f"\n{'═'*52}")
    print(f"  {label}")
    print(f"{'═'*52}")
    
    if "error" in metrics:
        print(f"  {metrics['error']}")
        return
    
    print(f"\n  {'TRADE STATISTICS':}")
    print(f"  {'─'*40}")
    print(f"  Total Trades      : {metrics['total_trades']}")
    print(f"  Win Rate          : {metrics['win_rate']*100:.1f}%  "
          f"({metrics['winning_trades']}W / {metrics['losing_trades']}L)")
    print(f"  Avg Win           : ₹{metrics['avg_win']:,.0f}")
    print(f"  Avg Loss          : ₹{metrics['avg_loss']:,.0f}")
    print(f"  Win/Loss Ratio    : {metrics['win_loss_ratio']:.2f}x")
    print(f"  Profit Factor     : {metrics['profit_factor']:.2f}")
    print(f"  Avg Bars Held     : {metrics['avg_bars_held']:.1f} days")
    
    print(f"\n  {'P&L SUMMARY':}")
    print(f"  {'─'*40}")
    print(f"  Gross P&L         : ₹{metrics['gross_pnl']:,.0f}")
    print(f"  Total Costs       : ₹{metrics['total_costs']:,.0f}")
    print(f"  Net P&L           : ₹{metrics['total_pnl']:,.0f}")
    print(f"  Final Capital     : ₹{metrics['final_capital']:,.0f}")
    print(f"  Total Return      : {metrics['total_return']*100:.1f}%")
    print(f"  CAGR              : {metrics['cagr']*100:.1f}%")
    
    print(f"\n  {'RISK METRICS':}")
    print(f"  {'─'*40}")
    print(f"  Sharpe Ratio      : {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio     : {metrics['sortino_ratio']:.2f}")
    print(f"  Max Drawdown      : {metrics['max_drawdown']*100:.1f}%")
    print(f"  Max DD Duration   : {metrics['max_dd_duration_days']} days")
    
    print(f"\n  {'EXIT REASON BREAKDOWN':}")
    print(f"  {'─'*40}")
    for reason, count in metrics.get("exit_reasons", {}).items():
        print(f"  {reason:<18}: {count}")
    
    print(f"\n{'═'*52}\n")
