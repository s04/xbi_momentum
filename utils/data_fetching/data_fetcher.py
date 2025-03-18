import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class DataFetcher:
    """
    Class for fetching and loading stock price data
    """
    
    def __init__(self, data_dir='data', debug=False):
        """
        Initialize the DataFetcher
        
        Parameters:
        -----------
        data_dir : str
            Directory where data is stored
        debug : bool
            Whether to print debug information
        """
        self.data_dir = data_dir
        self.debug = debug
        self.holdings_dir = os.path.join(data_dir, 'holdings')
        self.raw_data_dir = os.path.join(data_dir, 'raw')
        
        # Create directories if they don't exist
        os.makedirs(self.holdings_dir, exist_ok=True)
        os.makedirs(self.raw_data_dir, exist_ok=True)
    
    def get_xbi_tickers(self, holdings_file='xbi_holdings.csv'):
        """
        Extract ticker symbols from the XBI holdings CSV file
        """
        # Build the full path to the holdings file
        holdings_path = os.path.join(self.holdings_dir, holdings_file)
        
        # Read the XBI holdings CSV file
        holdings_df = pd.read_csv(holdings_path)
        
        # Extract the ticker column
        tickers = holdings_df['Ticker'].dropna().tolist()
        
        return tickers

    def fetch_historical_data(self, ticker, start_date, end_date, force_refresh=False):
        file_path = os.path.join(self.raw_data_dir, f"{ticker.upper()}.csv")
        
        # Check if file already exists and we're not forcing a refresh
        if os.path.exists(file_path) and not force_refresh:
            if self.debug:
                print(f"Using existing data file for {ticker}")
            return pd.read_csv(file_path, index_col=0, parse_dates=True)
                
        # Fetch data from Yahoo Finance
        try:
            if self.debug:
                print(f"Fetching new data for {ticker}...")
                
            stock = yf.Ticker(ticker)
            stock_data = stock.history(start=start_date, end=end_date)
            
            # Validate that we have data
            if stock_data.empty:
                print(f"No data returned for {ticker}")
                return None
                
            # Save to CSV
            os.makedirs(self.raw_data_dir, exist_ok=True)
            stock_data.to_csv(file_path)
            
            if self.debug:
                print(f"Successfully saved {ticker} data with {len(stock_data)} rows")
                
            return stock_data
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            if self.debug:
                import traceback
                print(traceback.format_exc())
            return None