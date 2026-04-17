
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

from config import (
    CAPITAL, TRAIN_END, VAL_END, START_DATE, END_DATE
)
from data_loader  import download_data, split_data
from features     import build_features
from strategy     import generate_signals
from backtest     import run_backtest, trades_to_df
from metrics      import calculate_metrics, print_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(df_signals: pd.DataFrame,
                 trades_df: pd.DataFrame,
                 equity_curve: pd.Series,
                 metrics: dict,
                 label: str = "Full Period") -> None:
    """
    Multi-panel performance chart:
    Panel 1 — Nifty price with trade entry/exit markers
    Panel 2 — Equity curve vs Buy & Hold
    Panel 3 — Hurst exponent with regime coloring
    Panel 4 — Drawdown
    Panel 5 — Monthly P&L heatmap
    """
    fig = plt.figure(figsize=(18, 22))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(5, 1, figure=fig,
                           height_ratios=[3, 2, 1.5, 1.5, 2],
                           hspace=0.4)
    
    axes = [fig.add_subplot(gs[i]) for i in range(5)]
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=9)
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["top"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["right"].set_color("#30363d")
    
    GOLD   = "#f0b429"
    GREEN  = "#3fb950"
    RED    = "#f85149"
    BLUE   = "#58a6ff"
    PURPLE = "#bc8cff"
    GRAY   = "#8b949e"
    
    # ── Panel 1: Price + Trades ───────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(df_signals.index, df_signals["Close"],
             color=BLUE, linewidth=0.8, alpha=0.8, label="Nifty Close")
    
    if not trades_df.empty:
        longs  = trades_df[trades_df["direction"] == "long"]
        shorts = trades_df[trades_df["direction"] == "short"]
        wins   = trades_df[trades_df["net_pnl_inr"] > 0]
        losses = trades_df[trades_df["net_pnl_inr"] <= 0]
        
        ax1.scatter(longs["entry_date"],
                    [df_signals.loc[d, "Close"] if d in df_signals.index else np.nan
                     for d in longs["entry_date"]],
                    marker="^", color=GREEN, s=80, zorder=5, label="Long Entry")
        ax1.scatter(shorts["entry_date"],
                    [df_signals.loc[d, "Close"] if d in df_signals.index else np.nan
                     for d in shorts["entry_date"]],
                    marker="v", color=RED, s=80, zorder=5, label="Short Entry")
        
        # Exit markers
        for _, t in wins.iterrows():
            if t["exit_date"] in df_signals.index:
                ax1.scatter(t["exit_date"],
                            df_signals.loc[t["exit_date"], "Close"],
                            marker="x", color=GREEN, s=60, zorder=5)
        for _, t in losses.iterrows():
            if t["exit_date"] in df_signals.index:
                ax1.scatter(t["exit_date"],
                            df_signals.loc[t["exit_date"], "Close"],
                            marker="x", color=RED, s=60, zorder=5)
    
    ax1.set_title(f"Nifty 50 — Entry/Exit Signals | {label}",
                  color=GOLD, fontsize=13, fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=8,
               facecolor="#21262d", edgecolor="#30363d", labelcolor=GRAY)
    ax1.set_ylabel("Nifty Points", color=GRAY, fontsize=9)
    
    # Train/Val/Test shading
    ax1.axvline(pd.Timestamp(TRAIN_END), color="#444c56", linestyle="--",
                linewidth=1, alpha=0.7)
    ax1.axvline(pd.Timestamp(VAL_END), color="#444c56", linestyle="--",
                linewidth=1, alpha=0.7)
    ax1.text(pd.Timestamp(TRAIN_END), ax1.get_ylim()[0],
             "Val→", color=GRAY, fontsize=7, va="bottom")
    ax1.text(pd.Timestamp(VAL_END), ax1.get_ylim()[0],
             "Test→", color=GRAY, fontsize=7, va="bottom")
    
    # ── Panel 2: Equity Curve ─────────────────────────────────────────────
    ax2 = axes[1]
    
    # Normalize to 100
    eq_norm = equity_curve / equity_curve.iloc[0] * 100
    bnh_norm = (df_signals["Close"].reindex(equity_curve.index).ffill() /
                df_signals["Close"].reindex(equity_curve.index).ffill().iloc[0] * 100)
    
    ax2.plot(eq_norm.index, eq_norm.values,
             color=GOLD, linewidth=1.5, label="Strategy")
    ax2.plot(bnh_norm.index, bnh_norm.values,
             color=GRAY, linewidth=1.0, linestyle="--", alpha=0.6,
             label="Buy & Hold")
    ax2.fill_between(eq_norm.index, eq_norm.values, 100,
                     where=(eq_norm.values >= 100),
                     alpha=0.15, color=GREEN)
    ax2.fill_between(eq_norm.index, eq_norm.values, 100,
                     where=(eq_norm.values < 100),
                     alpha=0.15, color=RED)
    ax2.axhline(100, color="#30363d", linewidth=0.8)
    ax2.set_title("Equity Curve vs Buy & Hold (Normalized to 100)",
                  color=GOLD, fontsize=11, fontweight="bold", pad=8)
    ax2.set_ylabel("Portfolio Value", color=GRAY, fontsize=9)
    ax2.legend(loc="upper left", fontsize=8,
               facecolor="#21262d", edgecolor="#30363d", labelcolor=GRAY)
    
    # ── Panel 3: Hurst Exponent ───────────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(df_signals.index, df_signals["hurst"],
             color=PURPLE, linewidth=0.8, alpha=0.9)
    ax3.axhline(0.55, color=GREEN, linewidth=1.0,
                linestyle="--", alpha=0.7, label="Trend threshold (0.55)")
    ax3.axhline(0.45, color=RED, linewidth=1.0,
                linestyle="--", alpha=0.7, label="MR threshold (0.45)")
    ax3.axhline(0.50, color=GRAY, linewidth=0.5,
                linestyle=":", alpha=0.5)
    ax3.fill_between(df_signals.index, 0.45, 0.55,
                     alpha=0.1, color=GRAY, label="No-trade zone")
    ax3.set_ylim(0.0, 1.0)
    ax3.set_title("Hurst Exponent (Market Regime)",
                  color=GOLD, fontsize=11, fontweight="bold", pad=8)
    ax3.set_ylabel("H", color=GRAY, fontsize=9)
    ax3.legend(loc="upper right", fontsize=7,
               facecolor="#21262d", edgecolor="#30363d", labelcolor=GRAY)
    
    # ── Panel 4: Drawdown ─────────────────────────────────────────────────
    ax4 = axes[3]
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve - rolling_max) / rolling_max * 100
    ax4.fill_between(drawdown.index, drawdown.values, 0,
                     color=RED, alpha=0.5)
    ax4.plot(drawdown.index, drawdown.values,
             color=RED, linewidth=0.8)
    ax4.set_title("Drawdown (%)",
                  color=GOLD, fontsize=11, fontweight="bold", pad=8)
    ax4.set_ylabel("Drawdown %", color=GRAY, fontsize=9)
    
    # ── Panel 5: Monthly P&L Bar Chart ────────────────────────────────────
    ax5 = axes[4]
    if not trades_df.empty and "exit_date" in trades_df.columns:
        trades_df2 = trades_df.copy()
        trades_df2["exit_date"] = pd.to_datetime(trades_df2["exit_date"])
        trades_df2["month"]     = trades_df2["exit_date"].dt.to_period("M")
        monthly_pnl = trades_df2.groupby("month")["net_pnl_inr"].sum()
        
        colors_bar = [GREEN if v >= 0 else RED for v in monthly_pnl.values]
        ax5.bar(range(len(monthly_pnl)), monthly_pnl.values,
                color=colors_bar, alpha=0.8, width=0.8)
        
        # X-axis: show year labels
        tick_positions = [i for i, p in enumerate(monthly_pnl.index)
                          if p.month == 1]
        tick_labels    = [str(monthly_pnl.index[i].year) for i in tick_positions]
        ax5.set_xticks(tick_positions)
        ax5.set_xticklabels(tick_labels, color=GRAY, fontsize=8)
        ax5.axhline(0, color=GRAY, linewidth=0.6)
    
    ax5.set_title("Monthly Net P&L (₹)",
                  color=GOLD, fontsize=11, fontweight="bold", pad=8)
    ax5.set_ylabel("₹", color=GRAY, fontsize=9)
    
    # ── Summary Stats Box ─────────────────────────────────────────────────
    if "error" not in metrics:
        stats_text = (
            f"CAGR: {metrics['cagr']*100:.1f}%  |  "
            f"Sharpe: {metrics['sharpe_ratio']:.2f}  |  "
            f"Max DD: {metrics['max_drawdown']*100:.1f}%  |  "
            f"Win Rate: {metrics['win_rate']*100:.1f}%  |  "
            f"PF: {metrics['profit_factor']:.2f}  |  "
            f"Trades: {metrics['total_trades']}"
        )
        fig.text(0.5, 0.01, stats_text,
                 ha="center", fontsize=10, color=GOLD,
                 bbox=dict(boxstyle="round,pad=0.4",
                           facecolor="#21262d", edgecolor="#444c56"))
    
    plt.suptitle("UNiverse Capital | Nifty 50 Futures Swing Algo",
                 color=GOLD, fontsize=15, fontweight="bold", y=0.99)
    
    filename = f"nifty_algo_results_{label.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    plt.close()
    print(f"[Chart] Saved → {filename}")
    return filename


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  UNiverse Capital | Nifty 50 Swing Algo")
    print("  Rule-Based | No Indicators | Statistical Edge")
    print("=" * 60)
    
    # ── 1. Load Data ──────────────────────────────────────────────────────
    raw = download_data()
    
    # ── 2. Build Features ─────────────────────────────────────────────────
    feat = build_features(raw)
    
    # ── 3. Generate Signals ───────────────────────────────────────────────
    signals = generate_signals(feat)
    
    # ── 4. Split Data ─────────────────────────────────────────────────────
    train_sig, val_sig, test_sig = split_data(signals, TRAIN_END, VAL_END)
    
    # ── 5. Run Backtest on Each Split ─────────────────────────────────────
    results = {}
    
    for label, subset in [("Train (2010–2018)", train_sig),
                           ("Val (2019–2021)",   val_sig),
                           ("Test (2022–2024)",  test_sig),
                           ("Full Period",        signals)]:
        print(f"\n{'─'*50}")
        print(f"  Running: {label}")
        print(f"{'─'*50}")
        
        trades, equity = run_backtest(subset, starting_capital=CAPITAL)
        tdf             = trades_to_df(trades)
        m               = calculate_metrics(tdf, equity, CAPITAL)
        
        print_metrics(m, label)
        results[label] = (tdf, equity, m)
    
    # ── 6. Charts ─────────────────────────────────────────────────────────
    filenames = []
    for label, (tdf, equity, m) in results.items():
        subset = {"Train (2010–2018)": train_sig,
                  "Val (2019–2021)":   val_sig,
                  "Test (2022–2024)":  test_sig,
                  "Full Period":       signals}[label]
        fname = plot_results(subset, tdf, equity, m, label)
        filenames.append(fname)
    
    print("\n[Done] All results generated.")
    print(f"Charts: {filenames}")
    return filenames


if __name__ == "__main__":
    main()
