import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class MicrostructureStatArb:
    def __init__(self, symbol='^NSEI', start_date='2013-01-01', split_date='2023-01-01', end_date='2024-01-01', initial_capital=1000000):
        self.symbol = symbol
        self.start_date = start_date
        self.split_date = split_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.brokerage_per_order = 20.0  # ₹20 per trade (₹40 round trip)

        self.data = None
        self.train_data = None
        self.test_data = None
        self.results = {}

    def fetch_and_clean_data(self):
        print(f"Fetching data for {self.symbol} from {self.start_date} to {self.end_date}...")
        df = yf.download(self.symbol, start=self.start_date, end=self.end_date, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        # --- MICROSTRUCTURE FEATURE ENGINEERING ---
        # 1. Overnight Gap Return (Close to Open)
        df['Overnight_Ret'] = (df['Open'] / df['Close'].shift(1)) - 1

        # 2. Intraday Return (Open to Close)
        df['Intraday_Ret'] = (df['Close'] / df['Open']) - 1

        # 3. Rolling Statistics (10-day window)
        # Shifted by 1 to avoid lookahead bias when making decisions
        df['Mean_Overnight'] = df['Overnight_Ret'].rolling(10).mean().shift(1)
        df['Std_Overnight'] = df['Overnight_Ret'].rolling(10).std().shift(1)

        df['Mean_Intra'] = df['Intraday_Ret'].rolling(10).mean().shift(1)
        df['Std_Intra'] = df['Intraday_Ret'].rolling(10).std().shift(1)

        self.data = df.dropna()

        # Split Data
        self.train_data = self.data[self.data.index < self.split_date]
        self.test_data = self.data[self.data.index >= self.split_date]

        print(f"Data split complete.")
        print(f"Training Data (10 Years): {len(self.train_data)} rows")
        print(f"Testing Data (1 Year unseen): {len(self.test_data)} rows")

    def run_backtest(self, df, period_name):
        print(f"\nRunning High-Frequency Microstructure engine for {period_name}...")

        cash = self.initial_capital

        overnight_pos = 0
        overnight_type = None
        overnight_entry = 0.0

        equity_curve = []
        completed_trades = []

        # --- HIGH FREQUENCY PARAMETERS ---
        GAP_Z_LONG = -0.5     # Buy open if gap down > 0.5 std dev
        GAP_Z_SHORT = 0.5     # Short open if gap up > 0.5 std dev

        INTRA_Z_LONG = 0.0    # Buy close if intraday action was negative
        INTRA_Z_SHORT = 1.5   # Short close only if intraday rally was extreme (>1.5 std dev)

        RISK_FRACTION = 0.50  # Use 50% of capital per trade

        # --- STOP LOSS PARAMETER ---
        STOP_LOSS_PCT = 0.015 # 1.5% fixed stop loss per trade

        for i in range(1, len(df)):
            date = df.index[i]
            open_price = df['Open'].iloc[i]
            high_price = df['High'].iloc[i]
            low_price = df['Low'].iloc[i]
            close_price = df['Close'].iloc[i]

            # ==========================================
            # MORNING: EXIT OVERNIGHT POSITION AT OPEN
            # ==========================================
            if overnight_pos > 0:
                # Overnight trades exit at the Open price unconditionally
                # (You cannot trigger a stop loss while the market is closed)
                exit_price = open_price

                if overnight_type == 'LONG':
                    pnl = (exit_price - overnight_entry) * overnight_pos - (2 * self.brokerage_per_order)
                else:
                    pnl = (overnight_entry - exit_price) * overnight_pos - (2 * self.brokerage_per_order)

                cash += (overnight_entry * overnight_pos) + pnl if overnight_type == 'LONG' else (overnight_entry * overnight_pos) + pnl
                ret_pct = pnl / (overnight_entry * overnight_pos)

                completed_trades.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Trade': 'OVERNIGHT',
                    'Type': overnight_type,
                    'Entry': round(overnight_entry, 2),
                    'Exit': round(exit_price, 2),
                    'PnL (₹)': round(pnl, 2),
                    'Return (%)': round(ret_pct * 100, 2),
                    'Reason': 'Time Exit (Open)'
                })
                overnight_pos = 0
                overnight_type = None

            # ==========================================
            # MORNING: ENTER INTRADAY POSITION AT OPEN
            # ==========================================
            if df['Std_Overnight'].iloc[i] > 0:
                gap_z = (df['Overnight_Ret'].iloc[i] - df['Mean_Overnight'].iloc[i]) / df['Std_Overnight'].iloc[i]

                trade_value = cash * RISK_FRACTION
                shares = int(trade_value / open_price)

                intraday_pos = 0
                intraday_type = None
                intraday_entry = 0.0

                if shares > 0:
                    if gap_z < GAP_Z_LONG:
                        intraday_type = 'LONG'
                        intraday_pos = shares
                        intraday_entry = open_price
                        cash -= (shares * open_price)
                    elif gap_z > GAP_Z_SHORT:
                        intraday_type = 'SHORT'
                        intraday_pos = shares
                        intraday_entry = open_price
                        cash -= (shares * open_price)

            # ==========================================
            # AFTERNOON: EXIT INTRADAY POSITION AT CLOSE
            # ==========================================
            if intraday_pos > 0:
                exit_price = close_price
                exit_reason = 'Time Exit (Close)'

                # Check if Intraday Stop Loss was hit using High/Low of the day
                if intraday_type == 'LONG':
                    sl_price = intraday_entry * (1 - STOP_LOSS_PCT)
                    if low_price <= sl_price:
                        exit_price = sl_price
                        exit_reason = 'Stop Loss Hit'

                    pnl = (exit_price - intraday_entry) * intraday_pos - (2 * self.brokerage_per_order)

                else: # SHORT
                    sl_price = intraday_entry * (1 + STOP_LOSS_PCT)
                    if high_price >= sl_price:
                        exit_price = sl_price
                        exit_reason = 'Stop Loss Hit'

                    pnl = (intraday_entry - exit_price) * intraday_pos - (2 * self.brokerage_per_order)

                cash += (intraday_entry * intraday_pos) + pnl
                ret_pct = pnl / (intraday_entry * intraday_pos)

                completed_trades.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Trade': 'INTRADAY',
                    'Type': intraday_type,
                    'Entry': round(intraday_entry, 2),
                    'Exit': round(exit_price, 2),
                    'PnL (₹)': round(pnl, 2),
                    'Return (%)': round(ret_pct * 100, 2),
                    'Reason': exit_reason
                })
                intraday_pos = 0

            # Record Equity at Close (Before entering new overnight position)
            equity_curve.append({'Date': date, 'Equity': cash})

            # ==========================================
            # AFTERNOON: ENTER OVERNIGHT POSITION AT CLOSE
            # ==========================================
            if df['Std_Intra'].iloc[i] > 0:
                intra_z = (df['Intraday_Ret'].iloc[i] - df['Mean_Intra'].iloc[i]) / df['Std_Intra'].iloc[i]

                trade_value = cash * RISK_FRACTION
                shares = int(trade_value / close_price)

                if shares > 0:
                    if intra_z < INTRA_Z_LONG:
                        overnight_type = 'LONG'
                        overnight_pos = shares
                        overnight_entry = close_price
                        cash -= (shares * close_price)
                    elif intra_z > INTRA_Z_SHORT:
                        overnight_type = 'SHORT'
                        overnight_pos = shares
                        overnight_entry = close_price
                        cash -= (shares * close_price)

        equity_df = pd.DataFrame(equity_curve).set_index('Date')

        self.results[period_name] = {
            'equity_df': equity_df,
            'trades': completed_trades
        }

    def print_trade_log(self, period_name):
        trades = self.results[period_name]['trades']
        if not trades:
            print(f"No trades executed in {period_name}.")
            return

        trades_df = pd.DataFrame(trades)
        print("\n" + "="*100)
        print(f"DETAILED TRADE LOG: {period_name.upper()} (Unseen Data)")
        print("="*100)
        print(trades_df.to_string(index=False))
        print("="*100 + "\n")

    def calculate_metrics(self, period_name):
        equity_df = self.results[period_name]['equity_df']
        trades = self.results[period_name]['trades']

        if equity_df is None or equity_df.empty:
            return

        equity = equity_df['Equity']
        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital

        days = (equity.index[-1] - equity.index[0]).days
        cagr = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else 0

        daily_returns = equity.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else 0

        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()

        winning_trades = [t for t in trades if t['PnL (₹)'] > 0]
        losing_trades = [t for t in trades if t['PnL (₹)'] <= 0]

        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        gross_profit = sum(t['PnL (₹)'] for t in winning_trades)
        gross_loss = abs(sum(t['PnL (₹)'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')

        print(f"\n--- {period_name.upper()} METRICS ---")
        print(f"Initial Capital: ₹{self.initial_capital:,.2f}")
        print(f"Final Equity:    ₹{equity.iloc[-1]:,.2f}")
        print(f"Total Return:    {total_return * 100:.2f}%")
        print(f"CAGR:            {cagr * 100:.2f}%")
        print(f"Sharpe Ratio:    {sharpe_ratio:.2f}")
        print(f"Max Drawdown:    {max_drawdown * 100:.2f}%")
        print(f"Win Rate:        {win_rate * 100:.2f}%")
        print(f"Profit Factor:   {profit_factor:.2f}")
        print(f"Total Trades:    {total_trades}")

    def run_monte_carlo(self, period_name, num_simulations=1000):
        print(f"\nRunning {num_simulations} Monte Carlo simulations based on {period_name} trade returns...")
        trades = self.results[period_name]['trades']

        if not trades:
            return

        trade_returns = [t['Return (%)'] / 100.0 for t in trades]
        num_trades = len(trade_returns)

        simulated_equities = np.zeros((num_simulations, num_trades + 1))
        simulated_equities[:, 0] = self.initial_capital

        for i in range(num_simulations):
            random_returns = np.random.choice(trade_returns, size=num_trades, replace=True)
            equity_impact = 1 + (0.50 * random_returns) # 50% risk fraction used in strategy
            simulated_equities[i, 1:] = self.initial_capital * np.cumprod(equity_impact)

        plt.figure(figsize=(12, 6))
        for i in range(min(num_simulations, 200)):
            plt.plot(simulated_equities[i, :], color='blue', alpha=0.05)

        median_path = np.median(simulated_equities, axis=0)
        p5_path = np.percentile(simulated_equities, 5, axis=0)
        p95_path = np.percentile(simulated_equities, 95, axis=0)

        plt.plot(median_path, color='red', linewidth=2, label='Median Path')
        plt.plot(p5_path, color='orange', linewidth=2, linestyle='--', label='5th Percentile (Worst Case)')
        plt.plot(p95_path, color='green', linewidth=2, linestyle='--', label='95th Percentile (Best Case)')

        plt.title(f'Monte Carlo Simulation ({num_simulations} Paths Resampled from {period_name} Trades)')
        plt.xlabel('Number of Trades')
        plt.ylabel('Portfolio Equity (₹)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('monte_carlo_simulation.png')
        print("Monte Carlo plot saved as 'monte_carlo_simulation.png'")

    def plot_results(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})

        train_eq = self.results['Train (10 Yr)']['equity_df']
        test_eq = self.results['Test (1 Yr Unseen)']['equity_df']

        ax1.plot(train_eq.index, train_eq['Equity'], label='Train Equity (In-Sample)', color='blue')

        test_eq_shifted = test_eq['Equity'] - self.initial_capital + train_eq['Equity'].iloc[-1]
        ax1.plot(test_eq.index, test_eq_shifted, label='Test Equity (Out-of-Sample)', color='orange')

        ax1.axvline(pd.to_datetime(self.split_date), color='black', linestyle='--', alpha=0.7, label='Train/Test Split')

        ax1.set_title('Microstructure StatArb: Train vs Test Equity Curve')
        ax1.set_ylabel('Equity (₹)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        combined_equity = pd.concat([train_eq['Equity'], test_eq_shifted])
        running_max = combined_equity.cummax()
        drawdown = (combined_equity - running_max) / running_max * 100

        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown (%)')
        ax2.axvline(pd.to_datetime(self.split_date), color='black', linestyle='--', alpha=0.7)
        ax2.set_title('Combined Drawdown Profile')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('nifty_traintest_backtest.png')
        print("\nBacktest plot saved as 'nifty_traintest_backtest.png'")
        plt.show()

if __name__ == "__main__":
    strategy = MicrostructureStatArb(
        symbol='^NSEI',
        start_date='2013-01-01',
        split_date='2023-01-01',
        end_date='2024-01-01',
        initial_capital=1000000
    )

    strategy.fetch_and_clean_data()

    strategy.run_backtest(strategy.train_data, 'Train (10 Yr)')
    strategy.run_backtest(strategy.test_data, 'Test (1 Yr Unseen)')

    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    strategy.calculate_metrics('Train (10 Yr)')
    strategy.calculate_metrics('Test (1 Yr Unseen)')
    print("="*50)

    strategy.print_trade_log('Test (1 Yr Unseen)')
    strategy.plot_results()
    strategy.run_monte_carlo('Test (1 Yr Unseen)', num_simulations=1000)