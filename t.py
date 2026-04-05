import numpy as np
import pandas as pd
import yfinance as yf
from itertools import product
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
#  CONFIGURATION & UNIVERSE
# -------------------------------------------------------------------
UNIVERSE = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'KOTAKBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS', 'AXISBANK.NS',
    'HINDUNILVR.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'TATAMOTORS.NS', 'WIPRO.NS'
]

# Lot sizes for NSE stock futures (approximate)
LOT_SIZES = {
    'RELIANCE.NS': 250, 'TCS.NS': 150, 'HDFCBANK.NS': 250, 'INFY.NS': 400,
    'ICICIBANK.NS': 550, 'KOTAKBANK.NS': 400, 'BHARTIARTL.NS': 700,
    'ITC.NS': 1600, 'SBIN.NS': 1300, 'AXISBANK.NS': 700, 'HINDUNILVR.NS': 300,
    'MARUTI.NS': 200, 'SUNPHARMA.NS': 400, 'TATAMOTORS.NS': 500, 'WIPRO.NS': 600
}

# Trading parameters (to be tuned on training set)
DEFAULT_PARAMS = {
    'holding_period': 5,      # days
    'z_threshold': 1.5,       # absolute z-score threshold
    'max_positions': 5,       # max concurrent positions
    'cooldown_days': 2,       # days to wait before re-entering same stock
    'stop_loss_pct': 0.08,    # 8% stop loss
    'trend_filter': False,    # optional 20-day MA filter
    'risk_per_trade': 0.02,   # 2% capital per trade (margin based)
    'margin_pct': 0.20,       # 20% initial margin
    'transaction_cost_rate': 0.001   # 0.1% (brokerage + STT + slippage)
}

# -------------------------------------------------------------------
#  DATA LOADING (with synthetic fallback)
# -------------------------------------------------------------------
def load_data(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data. Fallback to synthetic if yfinance fails."""
    try:
        data = yf.download(symbols, start=start, end=end, group_by='ticker', auto_adjust=False)
        # Extract 'Close' prices for each symbol
        closes = pd.DataFrame()
        for sym in symbols:
            if sym in data.columns.levels[1]:
                closes[sym] = data[sym]['Close']
            else:
                # fallback for missing ticker
                closes[sym] = np.nan
        closes = closes.dropna(how='all')
        if closes.empty or len(closes) < 100:
            raise ValueError("Insufficient data from yfinance")
        return closes
    except Exception as e:
        print(f"Warning: yfinance failed ({e}). Using synthetic data.")
        # Synthetic random walk with drift
        dates = pd.date_range(start=start, end=end, freq='B')
        np.random.seed(42)
        closes = pd.DataFrame(index=dates)
        for sym in symbols:
            price = 1000 + np.cumsum(np.random.normal(0, 1, len(dates))) * 5
            closes[sym] = np.abs(price) + 500
        return closes

# -------------------------------------------------------------------
#  SIGNAL GENERATION
# -------------------------------------------------------------------
def compute_signals(price_df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """
    Compute composite z-score signal:
      - 1-day return
      - 3-day return
      - cross-sectional z-score each day, average => composite signal
    Returns DataFrame of composite z-scores (aligned with prices).
    """
    returns_1d = price_df.pct_change(1)
    returns_3d = price_df.pct_change(lookback)

    def zscore_series(returns):
        # daily cross-sectional z-score
        return (returns - returns.mean(axis=1)) / returns.std(axis=1)

    z1 = zscore_series(returns_1d)
    z3 = zscore_series(returns_3d)
    composite = (z1 + z3) / 2.0
    return composite

# -------------------------------------------------------------------
#  BACKTEST ENGINE
# -------------------------------------------------------------------
class FuturesBacktester:
    def __init__(self, price_df: pd.DataFrame, params: Dict, lot_sizes: Dict):
        self.price_df = price_df
        self.params = params
        self.lot_sizes = lot_sizes
        self.signals = compute_signals(price_df)

        # Precompute 20-day moving average if trend filter is used
        if params['trend_filter']:
            self.ma20 = price_df.rolling(20).mean()
        else:
            self.ma20 = None

    def _trend_ok(self, symbol: str, date_idx: int, direction: str) -> bool:
        """Optional trend filter: long only if price > MA20, short only if price < MA20."""
        if not self.params['trend_filter']:
            return True
        price = self.price_df[symbol].iloc[date_idx]
        ma = self.ma20[symbol].iloc[date_idx]
        if pd.isna(ma):
            return False
        if direction == 'long':
            return price > ma
        else:  # short
            return price < ma

    def _calculate_lots(self, capital: float, entry_price: float, lot_size: int) -> int:
        """Risk-based position sizing: use risk_per_trade of capital as margin."""
        margin_per_lot = entry_price * lot_size * self.params['margin_pct']
        risk_capital = capital * self.params['risk_per_trade']
        lots = int(np.floor(risk_capital / margin_per_lot))
        return max(1, lots) if lots > 0 else 0

    def run(self, start_date: str, end_date: str) -> Dict:
        """
        Run backtest on a specific date range.
        Returns performance metrics dictionary.
        """
        # Slice data to period
        price_df = self.price_df.loc[start_date:end_date]
        signal_df = self.signals.loc[start_date:end_date]
        if self.ma20 is not None:
            ma20 = self.ma20.loc[start_date:end_date]
        else:
            ma20 = None

        dates = price_df.index
        if len(dates) == 0:
            return {'total_return': 0, 'sharpe': 0, 'max_dd': 0, 'win_rate': 0, 'total_trades': 0}

        # Portfolio state
        cash = 1_000_000.0  # initial capital
        positions = {}       # symbol -> {lots, entry_price, direction, exit_day_idx, stop_price}
        cooldown = {sym: 0 for sym in self.price_df.columns}
        trade_log = []       # internal only for metrics (not printed)

        # Helper to close a position
        def close_position(symbol, exit_idx, exit_price, reason):
            nonlocal cash
            pos = positions[symbol]
            lots = pos['lots']
            direction = pos['direction']
            entry_price = pos['entry_price']

            # PnL calculation for futures
            if direction == 1:  # long
                pnl = (exit_price - entry_price) * lots * self.lot_sizes[symbol]
            else:  # short
                pnl = (entry_price - exit_price) * lots * self.lot_sizes[symbol]

            # Transaction cost on exit
            exit_value = exit_price * lots * self.lot_sizes[symbol]
            exit_cost = exit_value * self.params['transaction_cost_rate']

            # Add back margin and realized PnL, subtract costs
            margin_used = entry_price * lots * self.lot_sizes[symbol] * self.params['margin_pct']
            cash += margin_used + pnl - exit_cost

            # Record trade for metrics
            trade_log.append({
                'symbol': symbol,
                'entry_date': pos['entry_date'],
                'exit_date': dates[exit_idx],
                'direction': direction,
                'pnl': pnl - exit_cost,
                'return_pct': (pnl / margin_used) if margin_used != 0 else 0
            })

            del positions[symbol]
            # set cooldown
            cooldown[symbol] = self.params['cooldown_days']

        # Main loop
        for i, today in enumerate(dates):
            # 1. Decrement cooldown counters
            for sym in cooldown:
                if cooldown[sym] > 0:
                    cooldown[sym] -= 1

            # 2. Check for exits (time-based or stop-loss)
            to_close = []
            for sym, pos in positions.items():
                # Time-based exit: if holding period reached, schedule exit at next open
                if i >= pos['exit_day_idx']:
                    to_close.append((sym, i, 'time'))
                    continue

                # Stop-loss: check today's close
                current_price = price_df[sym].iloc[i]
                if pos['direction'] == 1:  # long
                    loss_pct = (current_price - pos['entry_price']) / pos['entry_price']
                else:  # short
                    loss_pct = (pos['entry_price'] - current_price) / pos['entry_price']
                if loss_pct <= -self.params['stop_loss_pct']:
                    # Stop triggered: exit at next open (or same day if last day)
                    exit_idx = i + 1 if i + 1 < len(dates) else i
                    to_close.append((sym, exit_idx, 'stop'))

            # Close positions (exit at open of exit_idx)
            for sym, exit_idx, reason in to_close:
                if exit_idx < len(dates):
                    exit_price = price_df[sym].iloc[exit_idx]
                else:
                    exit_price = price_df[sym].iloc[-1]
                close_position(sym, exit_idx, exit_price, reason)

            # 3. Generate new signals for today (using today's composite z-score)
            if i == len(dates) - 1:
                break  # no trading on last day

            # Available cash must cover margin for new positions; we also limit max positions
            current_pos_count = len(positions)
            slots_available = self.params['max_positions'] - current_pos_count
            if slots_available <= 0:
                continue

            # Gather signals for stocks not held and not in cooldown
            candidate_signals = []
            for sym in price_df.columns:
                if sym in positions or cooldown[sym] > 0:
                    continue
                z = signal_df[sym].iloc[i]
                if np.isnan(z) or abs(z) < self.params['z_threshold']:
                    continue
                direction = 'long' if z < -self.params['z_threshold'] else 'short' if z > self.params['z_threshold'] else None
                if direction is None:
                    continue
                # Trend filter
                if not self._trend_ok(sym, i, direction):
                    continue
                candidate_signals.append((sym, z, direction))

            if not candidate_signals:
                continue

            # Sort: lowest z for long, highest z for short
            long_candidates = [(s, z, d) for (s, z, d) in candidate_signals if d == 'long']
            short_candidates = [(s, z, d) for (s, z, d) in candidate_signals if d == 'short']
            long_candidates.sort(key=lambda x: x[1])   # most negative first
            short_candidates.sort(key=lambda x: -x[1]) # most positive first

            # Select top signals up to available slots
            selected = []
            for cand in long_candidates[:slots_available]:
                selected.append(cand)
            slots_available -= len(selected)
            for cand in short_candidates[:slots_available]:
                selected.append(cand)

            # 4. Enter new positions at next day's open
            next_day_idx = i + 1
            if next_day_idx >= len(dates):
                continue
            for sym, z, direction in selected:
                entry_price = price_df[sym].iloc[next_day_idx]
                if np.isnan(entry_price):
                    continue
                lot_size = self.lot_sizes[sym]
                lots = self._calculate_lots(cash, entry_price, lot_size)
                if lots == 0:
                    continue
                margin_needed = entry_price * lots * lot_size * self.params['margin_pct']
                # Transaction cost on entry
                entry_cost = entry_price * lots * lot_size * self.params['transaction_cost_rate']
                if cash < margin_needed + entry_cost:
                    continue

                # Deduct margin and cost
                cash -= (margin_needed + entry_cost)

                # Record position
                direction_code = 1 if direction == 'long' else -1
                exit_day = next_day_idx + self.params['holding_period']
                positions[sym] = {
                    'lots': lots,
                    'entry_price': entry_price,
                    'direction': direction_code,
                    'entry_date': dates[next_day_idx],
                    'exit_day_idx': exit_day
                }

        # After loop, close any remaining positions at last available price
        last_idx = len(dates) - 1
        for sym in list(positions.keys()):
            exit_price = price_df[sym].iloc[last_idx]
            close_position(sym, last_idx, exit_price, 'end_of_period')

        # Compute performance metrics
        if not trade_log:
            return {'total_return': 0, 'sharpe': 0, 'max_dd': 0, 'win_rate': 0, 'total_trades': 0}

        trades_df = pd.DataFrame(trade_log)
        total_pnl = trades_df['pnl'].sum()
        total_return = total_pnl / 1_000_000.0

        # Approximate daily equity curve from trade PnL (simplified, no intraday MTM)
        # We build equity by cumulating PnL at trade exit dates, but for Sharpe we need daily.
        # Simpler: use final equity and assume daily returns proportional to trades.
        # For robust metric, compute approximate daily returns by marking to market? 
        # Instead, compute Sharpe from trade returns (not perfect but indicative)
        trade_returns = trades_df['return_pct'].values
        if len(trade_returns) > 1:
            avg_ret = np.mean(trade_returns)
            std_ret = np.std(trade_returns)
            sharpe = (avg_ret / std_ret) * np.sqrt(252 / self.params['holding_period']) if std_ret > 0 else 0
        else:
            sharpe = 0

        # Maximum drawdown from cumulative PnL (approximated by trade sequence)
        cum_pnl = trades_df['pnl'].cumsum()
        running_max = cum_pnl.cummax()
        drawdown = (cum_pnl - running_max) / 1_000_000.0
        max_dd = drawdown.min() if len(drawdown) > 0 else 0

        win_rate = (trades_df['pnl'] > 0).mean()

        return {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'total_trades': len(trade_log),
            'avg_holding_days': self.params['holding_period']  # approximate
        }

# -------------------------------------------------------------------
#  PARAMETER TUNING (ON TRAINING PERIOD ONLY)
# -------------------------------------------------------------------
def tune_parameters(price_df: pd.DataFrame, train_start: str, train_end: str, lot_sizes: Dict) -> Dict:
    """Grid search over hyperparameters on training period, return best params."""
    param_grid = {
        'holding_period': [3, 5, 7, 10],
        'z_threshold': [1.0, 1.2, 1.5, 1.8, 2.0],
        'max_positions': [3, 5, 7],
        'cooldown_days': [1, 2, 3]
    }
    base_params = DEFAULT_PARAMS.copy()
    best_sharpe = -np.inf
    best_params = None

    for hp, zt, mp, cd in product(param_grid['holding_period'],
                                  param_grid['z_threshold'],
                                  param_grid['max_positions'],
                                  param_grid['cooldown_days']):
        params = base_params.copy()
        params.update({
            'holding_period': hp,
            'z_threshold': zt,
            'max_positions': mp,
            'cooldown_days': cd
        })
        backtester = FuturesBacktester(price_df, params, lot_sizes)
        perf = backtester.run(train_start, train_end)
        sharpe = perf['sharpe']
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params.copy()
    return best_params

# -------------------------------------------------------------------
#  MAIN EXECUTION
# -------------------------------------------------------------------
def main():
    # 1. Load data
    print("Loading data...", end=" ", flush=True)
    price_data = load_data(UNIVERSE, '2010-01-01', '2024-12-31')
    print("Done.")

    # 2. Define periods
    train_start, train_end = '2010-01-01', '2018-12-31'
    valid_start, valid_end = '2019-01-01', '2021-12-31'
    test_start, test_end = '2022-01-01', '2024-12-31'

    # 3. Tune parameters on training period only
    print("Tuning parameters on training set (2010-2018)...", end=" ", flush=True)
    best_params = tune_parameters(price_data, train_start, train_end, LOT_SIZES)
    print("Done.")

    # 4. Lock parameters and run test period
    print("Running test period (2022-2024) with locked parameters...", end=" ", flush=True)
    backtester = FuturesBacktester(price_data, best_params, LOT_SIZES)
    test_perf = backtester.run(test_start, test_end)
    print("Done.\n")

    # 5. Output clean performance summary
    print("PERFORMANCE SUMMARY")
    print("-" * 40)
    print(f"Total Return:        {test_perf['total_return']:.2%}")
    print(f"CAGR (approx):       {test_perf['total_return']**(1/3)-1:.2%}")  # 3 years
    print(f"Sharpe Ratio:        {test_perf['sharpe']:.2f}")
    print(f"Max Drawdown:        {test_perf['max_dd']:.2%}")
    print(f"Win Rate:            {test_perf['win_rate']:.2%}")
    print(f"Total Trades:        {test_perf['total_trades']}")
    print(f"Avg Holding Days:    {best_params['holding_period']}")
    print("-" * 40)

if __name__ == "__main__":
    main()