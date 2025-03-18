import pandas as pd
import os
import glob
from datetime import datetime, timedelta

def calculate_12m_return(ticker, data_dir='data', debug=False):
    """
    Calculate 12-month total return for a given ticker,
    skipping the most recent month to avoid short-term reversal noise
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol
    data_dir : str
        Directory where data is stored
    debug : bool
        Whether to print debug information
        
    Returns:
    --------
    float or None
        12-month total return as a percentage, or None if calculation fails
    """
    raw_data_dir = os.path.join(data_dir, 'raw')
    file_path = os.path.join(raw_data_dir, f"{ticker.upper()}.csv")
    
    if not os.path.exists(file_path):
        if debug:
            print(f"No data file found for {ticker}")
        return None
    
    try:
        # Load the stock data
        stock_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Sort by date
        stock_data = stock_data.sort_index()
        
        # Check if we have enough data
        if len(stock_data) < 30:  # Need at least a month of data to skip
            if debug:
                print(f"Not enough data points for {ticker}")
            return None
        
        # Calculate the date 1 month ago from the last data point
        last_date = stock_data.index[-1]
        skip_from_date = last_date - timedelta(days=30)
        
        # Find the index position right before the skip date
        end_data = stock_data[stock_data.index <= skip_from_date]
        
        if len(end_data) < 2:
            if debug:
                print(f"Not enough historical data after skipping recent month for {ticker}")
            return None
            
        # Calculate the date 12 months before the end date (13 months from latest data)
        start_date = skip_from_date - timedelta(days=365)
        start_data = stock_data[stock_data.index >= start_date]
        start_data = start_data[start_data.index <= skip_from_date]
        
        if len(start_data) < 2:
            if debug:
                print(f"Not enough data in the 12-month window for {ticker}")
            return None
        
        # Get first and last close price from the 12-month window (excluding most recent month)
        first_close = start_data['Close'].iloc[0]
        last_close = end_data['Close'].iloc[-1]
        
        # Calculate total return
        total_return = (last_close / first_close - 1) * 100
        
        # if debug:
        #     date_range = f"{start_data.index[0].date()} to {end_data.index[-1].date()}"
        #     skip_note = f"(skipping data after {end_data.index[-1].date()})"
        #     print(f"{ticker}: {total_return:.2f}% {date_range} {skip_note}")
        
        return total_return
        
    except Exception as e:
        if debug:
            print(f"Error calculating return for {ticker}: {e}")
        return None

def calculate_12m_return_excluding_most_recent_month(data_dir='data', debug=False):
    """
    Calculate 12-month returns for all stocks in the raw data directory,
    skipping the most recent month to avoid short-term reversal noise
    
    Parameters:
    -----------
    data_dir : str
        Directory where data is stored
    debug : bool
        Whether to print debug information
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the returns for all tickers
    """
    raw_data_dir = os.path.join(data_dir, 'raw')
    returns_dir = os.path.join(data_dir, 'returns')
    
    # Create returns directory if it doesn't exist
    os.makedirs(returns_dir, exist_ok=True)
    
    # Get all CSV files in the raw data directory
    csv_files = glob.glob(os.path.join(raw_data_dir, "*.csv"))
    
    results = []
    
    for file_path in csv_files:
        ticker = os.path.basename(file_path).replace('.csv', '')
        return_value = calculate_12m_return(ticker, data_dir, debug)
        
        if return_value is not None:
            results.append({
                'ticker': ticker,
                'total_return_12m_excluding_most_recent_month': return_value
            })
    
    # Create a DataFrame from the results
    returns_df = pd.DataFrame(results)
    
    # Sort by total return (descending)
    if not returns_df.empty:
        returns_df = returns_df.sort_values('total_return_12m_excluding_most_recent_month', ascending=False)
        
        # Save to CSV
        output_file = os.path.join(returns_dir, f"12m_returns_ex1m_{datetime.now().strftime('%Y%m%d')}.csv")
        returns_df.to_csv(output_file, index=False)
        
        if debug:
            print(f"Saved returns data to {output_file}")
    
    return returns_df

def get_momentum_winners_and_losers(data_dir='data', debug=False, top_n=5):
    """
    Identify momentum winners (top performers) and losers (bottom performers)
    based on 12-month returns excluding the most recent month
    
    Parameters:
    -----------
    data_dir : str
        Directory where data is stored
    debug : bool
        Whether to print debug information
    top_n : int
        Number of top and bottom performers to return
        
    Returns:
    --------
    tuple (DataFrame, DataFrame)
        Top momentum winners and bottom momentum losers
    """
    # Get the returns data
    returns_df = calculate_12m_return_excluding_most_recent_month(data_dir, debug)
    
    if returns_df.empty:
        if debug:
            print("No returns data available")
        return pd.DataFrame(), pd.DataFrame()
    
    # Get top momentum winners
    top_momentum_winners = returns_df.head(top_n).copy()
    
    # Get bottom momentum losers
    bottom_momentum_losers = returns_df.tail(top_n).copy()
    # Sort from worst to less bad for bottom performers
    bottom_momentum_losers = bottom_momentum_losers.sort_values('total_return_12m_excluding_most_recent_month', ascending=True)
    
    if debug:
        print(f"\nTop {top_n} Momentum Winners:")
        for _, row in top_momentum_winners.iterrows():
            print(f"{row['ticker']}: {row['total_return_12m_excluding_most_recent_month']:.2f}%")
        
        print(f"\nBottom {top_n} Momentum Losers:")
        for _, row in bottom_momentum_losers.iterrows():
            print(f"{row['ticker']}: {row['total_return_12m_excluding_most_recent_month']:.2f}%")
    
    return top_momentum_winners, bottom_momentum_losers 