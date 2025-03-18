import utils.data_fetching.data_fetcher as data_fetcher
import utils.calculations.returns as returns
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Constants
REFRESH_DATA = True  # Set to True to delete existing data and refetch
START_DATE = '2015-01-01'  # Changed from 2024 to 2015 to match 10-year backtest period
END_DATE = datetime.now().strftime('%Y-%m-%d')
INITIAL_PORTFOLIO_VALUE = 10_000_000  # $10M ($5M long, $5M short) as described in the strategy
POSITION_SIZE = 1_000_000  # $1M per position as described in the strategy
# INITIAL_PORTFOLIO_VALUE = 100_000  # $100K ($50K long, $50K short)
# POSITION_SIZE = 10_000  # $10K per position
TRANSACTION_COST_PCT = 0.001  # 0.1% per trade
TOP_N = 5  # Number of top and bottom performers to trade

def backtest_momentum_strategy(rebalance_freq='weekly', debug=False):
    """
    Backtest the momentum strategy with specified rebalance frequency
    
    Parameters:
    -----------
    rebalance_freq : str
        Frequency of rebalancing ('daily' or 'weekly')
    debug : bool
        Whether to print debug information
    
    Returns:
    --------
    tuple
        (performance_metrics_dict, value_history_dataframe)
    """
    # Setup
    datafetcher = data_fetcher.DataFetcher(debug=debug)
    tickers = datafetcher.get_xbi_tickers()
    
    # Fetch historical data for all tickers
    price_data = {}
    for ticker in tickers:
        data = datafetcher.fetch_historical_data(ticker, START_DATE, END_DATE, force_refresh=REFRESH_DATA)
        if data is not None and not data.empty:
            price_data[ticker] = data
    
    if not price_data:
        print("No valid price data found.")
        return {}
    
    # Create a date range for the backtest period
    all_dates = []
    for ticker, data in price_data.items():
        all_dates.extend(data.index.tolist())
    
    all_dates = sorted(list(set(all_dates)))
    
    # Determine rebalance dates based on frequency
    if rebalance_freq == 'daily':
        rebalance_dates = all_dates
    elif rebalance_freq == 'weekly':
        # Get only dates that are Fridays (weekday=4)
        rebalance_dates = [date for date in all_dates if date.weekday() == 4]
    else:
        raise ValueError("rebalance_freq must be 'daily' or 'weekly'")
    
    # Initialize portfolio tracking
    portfolio = {
        'cash': INITIAL_PORTFOLIO_VALUE,
        'positions': {},  # ticker -> {'shares': count, 'cost_basis': price}
        'value_history': [],
        'transaction_costs': 0,
        'rebalance_dates': []
    }
    
    # Initialize empty initial positions
    for ticker in tickers:
        portfolio['positions'][ticker] = {'shares': 0, 'cost_basis': 0, 'position_type': None}
    
    # For each rebalance date
    for date in rebalance_dates:
        if debug:
            print(f"\nProcessing rebalance date: {date.date()}")
            
        # Get current prices for all tickers
        current_prices = {}
        for ticker, data in price_data.items():
            # Get the last price on or before the current date
            prices_until_date = data[data.index <= date]
            if not prices_until_date.empty:
                current_prices[ticker] = prices_until_date['Close'].iloc[-1]
        
        # Calculate current portfolio value
        portfolio_value = portfolio['cash']
        for ticker, position in portfolio['positions'].items():
            if position['shares'] != 0 and ticker in current_prices:
                position_value = position['shares'] * current_prices[ticker]
                portfolio_value += position_value
        
        # Record portfolio value
        portfolio['value_history'].append({
            'date': date,
            'value': portfolio_value
        })
        
        # Create a temporary data directory for the current rebalance date
        temp_data_dir = os.path.join('data', f'temp_{date.strftime("%Y%m%d")}')
        os.makedirs(os.path.join(temp_data_dir, 'raw'), exist_ok=True)
        
        # Create point-in-time data for all tickers (only data available up to the rebalance date)
        valid_tickers = []
        for ticker, data in price_data.items():
            data_until_date = data[data.index <= date]
            if len(data_until_date) >= 30:  # Need at least a month of data
                data_file = os.path.join(temp_data_dir, 'raw', f"{ticker}.csv")
                data_until_date.to_csv(data_file)
                valid_tickers.append(ticker)
                
        if len(valid_tickers) < TOP_N * 2:
            if debug:
                print(f"Not enough tickers with sufficient history on {date.date()}. Skipping rebalance.")
            
            # Clean up temporary directory
            shutil.rmtree(temp_data_dir, ignore_errors=True)
            continue
        
        # Use the returns.py module to calculate momentum and get winners/losers
        top_momentum_winners, bottom_momentum_losers = returns.get_momentum_winners_and_losers(
            data_dir=temp_data_dir, 
            debug=debug,
            top_n=TOP_N
        )
        
        # Clean up temporary directory
        shutil.rmtree(temp_data_dir, ignore_errors=True)
        
        if top_momentum_winners.empty or bottom_momentum_losers.empty:
            if debug:
                print(f"Could not calculate momentum rankings on {date.date()}. Skipping rebalance.")
            continue
        
        # Prepare the target portfolio
        target_portfolio = {}
        
        # Top performers get long positions
        for _, row in top_momentum_winners.iterrows():
            ticker = row['ticker']
            if ticker in current_prices and current_prices[ticker] > 0:
                target_portfolio[ticker] = {
                    'target_value': POSITION_SIZE,
                    'position_type': 'long',
                    'current_price': current_prices[ticker]
                }
        
        # Bottom performers get short positions
        for _, row in bottom_momentum_losers.iterrows():
            ticker = row['ticker']
            if ticker in current_prices and current_prices[ticker] > 0:
                target_portfolio[ticker] = {
                    'target_value': POSITION_SIZE,
                    'position_type': 'short',
                    'current_price': current_prices[ticker]
                }
        
        # Close positions that are no longer in the target portfolio
        for ticker, position in list(portfolio['positions'].items()):
            if position['shares'] != 0 and ticker not in target_portfolio:
                # Close position
                if ticker in current_prices:
                    close_value = abs(position['shares'] * current_prices[ticker])
                    transaction_cost = close_value * TRANSACTION_COST_PCT
                    portfolio['transaction_costs'] += transaction_cost
                    
                    # Add cash back to portfolio (minus transaction costs)
                    if position['position_type'] == 'long':
                        portfolio['cash'] += close_value - transaction_cost
                    else:  # short position
                        portfolio['cash'] += close_value - transaction_cost
                    
                    # Reset position
                    portfolio['positions'][ticker] = {'shares': 0, 'cost_basis': 0, 'position_type': None}
                    
                    if debug:
                        print(f"Closed {position['position_type']} position in {ticker} for ${close_value:,.2f}")
        
        # Adjust existing positions and open new ones
        for ticker, target in target_portfolio.items():
            if ticker in current_prices and current_prices[ticker] > 0:
                current_price = current_prices[ticker]
                current_position = portfolio['positions'].get(ticker, {'shares': 0, 'cost_basis': 0, 'position_type': None})
                
                # Calculate target shares
                target_shares = POSITION_SIZE / current_price
                if target['position_type'] == 'short':
                    target_shares = -target_shares
                
                # Calculate shares to trade
                shares_to_trade = target_shares - current_position['shares']
                
                if abs(shares_to_trade) > 0:
                    # Calculate trade value and transaction cost
                    trade_value = abs(shares_to_trade * current_price)
                    transaction_cost = trade_value * TRANSACTION_COST_PCT
                    
                    # Execute trade
                    if shares_to_trade > 0 and target['position_type'] == 'long':
                        # Buy more shares (long)
                        if portfolio['cash'] >= trade_value + transaction_cost:
                            portfolio['cash'] -= (trade_value + transaction_cost)
                            portfolio['positions'][ticker] = {
                                'shares': current_position['shares'] + shares_to_trade,
                                'cost_basis': current_price,
                                'position_type': 'long'
                            }
                            portfolio['transaction_costs'] += transaction_cost
                            if debug:
                                print(f"Bought {shares_to_trade:.2f} shares of {ticker} at ${current_price:.2f}")
                    elif shares_to_trade < 0 and target['position_type'] == 'long':
                        # Sell some shares (long)
                        portfolio['cash'] += (-shares_to_trade * current_price - transaction_cost)
                        portfolio['positions'][ticker] = {
                            'shares': current_position['shares'] + shares_to_trade,
                            'cost_basis': current_price if current_position['shares'] + shares_to_trade > 0 else 0,
                            'position_type': 'long'
                        }
                        portfolio['transaction_costs'] += transaction_cost
                        if debug:
                            print(f"Sold {-shares_to_trade:.2f} shares of {ticker} at ${current_price:.2f}")
                    elif shares_to_trade < 0 and target['position_type'] == 'short':
                        # Short more shares
                        if portfolio['cash'] >= transaction_cost:
                            portfolio['cash'] -= transaction_cost
                            portfolio['positions'][ticker] = {
                                'shares': current_position['shares'] + shares_to_trade,
                                'cost_basis': current_price,
                                'position_type': 'short'
                            }
                            portfolio['transaction_costs'] += transaction_cost
                            if debug:
                                print(f"Shorted {-shares_to_trade:.2f} shares of {ticker} at ${current_price:.2f}")
                    elif shares_to_trade > 0 and target['position_type'] == 'short':
                        # Cover some short shares
                        if portfolio['cash'] >= transaction_cost:
                            portfolio['cash'] -= transaction_cost
                            portfolio['positions'][ticker] = {
                                'shares': current_position['shares'] + shares_to_trade,
                                'cost_basis': current_price if current_position['shares'] + shares_to_trade < 0 else 0,
                                'position_type': 'short'
                            }
                            portfolio['transaction_costs'] += transaction_cost
                            if debug:
                                print(f"Covered {shares_to_trade:.2f} shares of {ticker} at ${current_price:.2f}")
        
        # Record rebalance date
        portfolio['rebalance_dates'].append(date)
        
        if debug:
            print(f"Rebalanced on {date.date()}, Portfolio Value: ${portfolio_value:,.2f}")
    
    # Calculate performance metrics
    performance = calculate_performance_metrics(portfolio, price_data, rebalance_freq)
    
    # Return both performance metrics and the value history
    return performance, pd.DataFrame(portfolio['value_history'])

def calculate_performance_metrics(portfolio, price_data, rebalance_freq):
    """
    Calculate performance metrics for the backtest
    
    Parameters:
    -----------
    portfolio : dict
        Portfolio tracking information
    price_data : dict
        Price data for all tickers
    rebalance_freq : str
        Rebalance frequency used
        
    Returns:
    --------
    dict
        Dictionary with performance metrics
    """
    if not portfolio['value_history']:
        return {
            'rebalance_freq': rebalance_freq,
            'total_return': 0,
            'annualized_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'transaction_costs': 0
        }
    
    # Convert value history to DataFrame
    value_df = pd.DataFrame(portfolio['value_history'])
    value_df.set_index('date', inplace=True)
    
    # Calculate daily returns
    value_df['daily_return'] = value_df['value'].pct_change()
    
    # Calculate total return
    first_value = value_df['value'].iloc[0]
    last_value = value_df['value'].iloc[-1]
    total_return = (last_value / first_value - 1) * 100
    
    # Calculate annualized return
    days = (value_df.index[-1] - value_df.index[0]).days
    years = days / 365.25
    annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    # Calculate max drawdown
    value_df['cummax'] = value_df['value'].cummax()
    value_df['drawdown'] = (value_df['value'] / value_df['cummax'] - 1) * 100
    max_drawdown = value_df['drawdown'].min()
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
    daily_returns = value_df['daily_return'].dropna()
    sharpe_ratio = 0
    if len(daily_returns) > 0:
        annualized_vol = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return / 100) / annualized_vol if annualized_vol > 0 else 0
    
    # Return metrics
    return {
        'rebalance_freq': rebalance_freq,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'transaction_costs': portfolio['transaction_costs'],
        'final_value': last_value,
        'num_rebalances': len(portfolio['rebalance_dates'])
    }

def plot_portfolio_value_comparison(daily_value_df, weekly_value_df, save_path=None):
    """
    Plot the portfolio value over time for daily and weekly rebalancing strategies
    
    Parameters:
    -----------
    daily_value_df : pandas.DataFrame
        DataFrame containing portfolio value history for daily rebalancing
    weekly_value_df : pandas.DataFrame
        DataFrame containing portfolio value history for weekly rebalancing
    save_path : str
        Path to save the plot image file (optional)
    """
    plt.figure(figsize=(12, 8))
    
    # Plot portfolio values
    plt.plot(daily_value_df['date'], daily_value_df['value'], label='Daily Rebalancing', linewidth=2)
    plt.plot(weekly_value_df['date'], weekly_value_df['value'], label='Weekly Rebalancing', linewidth=2)
    
    # Add initial portfolio value as a horizontal line
    plt.axhline(y=INITIAL_PORTFOLIO_VALUE, color='gray', linestyle='--', 
                label=f'Initial Value (${INITIAL_PORTFOLIO_VALUE:,})')
    
    # Format the plot
    plt.title('Momentum Strategy Portfolio Value Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Portfolio Value ($)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Format y-axis as currency
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

# Run backtest with both daily and weekly rebalancing
def main():
    # Create results directory
    results_dir = os.path.join('data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Delete existing data if REFRESH_DATA is True
    if REFRESH_DATA:
        raw_data_dir = os.path.join('data', 'raw')
        returns_dir = os.path.join('data', 'returns')
        
        if os.path.exists(raw_data_dir):
            print("Deleting existing raw data files...")
            shutil.rmtree(raw_data_dir)
            
        if os.path.exists(returns_dir):
            print("Deleting existing returns files...")
            shutil.rmtree(returns_dir)
            
        # Recreate directories
        os.makedirs(raw_data_dir, exist_ok=True)
        os.makedirs(returns_dir, exist_ok=True)
    
    # Fetch data once
    datafetcher = data_fetcher.DataFetcher(debug=False)
    tickers = datafetcher.get_xbi_tickers()
    print(f"Found {len(tickers)} tickers in XBI holdings")
    
    # Fetch historical data for each ticker
    for ticker in tickers:
        datafetcher.fetch_historical_data(ticker, START_DATE, END_DATE, force_refresh=REFRESH_DATA)
    
    # Run daily and weekly backtests
    print("\n=== Running Momentum Strategy Backtests ===")
    print("Strategy: 12-month momentum excluding the most recent month")
    print(f"Position size: ${POSITION_SIZE:,}")
    print(f"Transaction cost: {TRANSACTION_COST_PCT*100}%")
    
    print("\nRunning daily rebalancing backtest...")
    daily_results, daily_value_df = backtest_momentum_strategy(rebalance_freq='daily', debug=True)
    
    print("\nRunning weekly rebalancing backtest...")
    weekly_results, weekly_value_df = backtest_momentum_strategy(rebalance_freq='weekly', debug=True)
    
    # Compare results
    print("\n=== Performance Comparison ===")
    print(f"{'Metric':<20} {'Daily Rebalancing':<20} {'Weekly Rebalancing':<20}")
    print("-" * 60)
    print(f"{'Total Return':<20} {daily_results.get('total_return', 0):,.2f}% {weekly_results.get('total_return', 0):,.2f}%")
    print(f"{'Annualized Return':<20} {daily_results.get('annualized_return', 0):,.2f}% {weekly_results.get('annualized_return', 0):,.2f}%")
    print(f"{'Max Drawdown':<20} {daily_results.get('max_drawdown', 0):,.2f}% {weekly_results.get('max_drawdown', 0):,.2f}%")
    print(f"{'Sharpe Ratio':<20} {daily_results.get('sharpe_ratio', 0):,.2f} {weekly_results.get('sharpe_ratio', 0):,.2f}")
    print(f"{'Transaction Costs':<20} ${daily_results.get('transaction_costs', 0):,.2f} ${weekly_results.get('transaction_costs', 0):,.2f}")
    print(f"{'Final Value':<20} ${daily_results.get('final_value', 0):,.2f} ${weekly_results.get('final_value', 0):,.2f}")
    print(f"{'# of Rebalances':<20} {daily_results.get('num_rebalances', 0)} {weekly_results.get('num_rebalances', 0)}")
    
    # Plot portfolio value comparison
    plot_file = os.path.join(results_dir, f"momentum_portfolio_value_{datetime.now().strftime('%Y%m%d')}.png")
    plot_portfolio_value_comparison(daily_value_df, weekly_value_df, save_path=plot_file)
    
    # Save results
    results_df = pd.DataFrame([daily_results, weekly_results])
    results_file = os.path.join(results_dir, f"momentum_backtest_results_{datetime.now().strftime('%Y%m%d')}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()
    
