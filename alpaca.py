import os
import time
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
import schedule

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetStatus

import utils.data_fetching.data_fetcher as data_fetcher
import utils.calculations.returns as returns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("momentum_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
TOP_N = 5  # Number of top and bottom performers to trade
POSITION_VALUE = 10_000  # $10K per position
LOOKBACK_DAYS = 395  # ~13 months for momentum calculation

def load_config():
    """Load environment variables from .env file"""
    load_dotenv()
    
    config = {
        'ALPACA_API_KEY': os.getenv('ALPACA_API_KEY'),
        'ALPACA_SECRET_KEY': os.getenv('ALPACA_SECRET_KEY'),
        'PAPER_TRADING': True  # Always use paper trading for safety
    }
    
    # Validate required environment variables
    if not config['ALPACA_API_KEY'] or not config['ALPACA_SECRET_KEY']:
        raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env file")
        
    return config

def initialize_client(config):
    """Initialize Alpaca trading client"""
    trading_client = TradingClient(
        api_key=config['ALPACA_API_KEY'],
        secret_key=config['ALPACA_SECRET_KEY'],
        paper=config['PAPER_TRADING']
    )
    
    # Check account status
    account = trading_client.get_account()
    logger.info(f"Trading account ready: {account.account_number}")
    logger.info(f"Account cash: ${float(account.cash):.2f}")
    logger.info(f"Account equity: ${float(account.equity):.2f}")
    
    return trading_client

def get_tradable_assets(trading_client):
    """Get list of tradable assets on Alpaca"""
    assets = trading_client.get_all_assets()
    tradable_assets = {}
    
    for asset in assets:
        if asset.tradable and asset.status == AssetStatus.ACTIVE and asset.shortable:
            tradable_assets[asset.symbol] = {
                'name': asset.name,
                'exchange': asset.exchange,
                'shortable': asset.shortable
            }
    
    logger.info(f"Found {len(tradable_assets)} tradable assets")
    return tradable_assets

def fetch_momentum_data():
    """Fetch latest stock data and calculate momentum signals"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    
    # Setup data fetcher
    datafetcher = data_fetcher.DataFetcher(debug=False)
    tickers = datafetcher.get_xbi_tickers()
    
    logger.info(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")
    
    # Create temporary data directory
    temp_data_dir = os.path.join('data', f'temp_{datetime.now().strftime("%Y%m%d%H%M%S")}')
    os.makedirs(os.path.join(temp_data_dir, 'raw'), exist_ok=True)
    
    # Fetch latest data for each ticker
    valid_tickers = []
    for ticker in tickers:
        data = datafetcher.fetch_historical_data(ticker, start_date, end_date, force_refresh=True)
        if data is not None and not data.empty and len(data) >= 30:
            data_file = os.path.join(temp_data_dir, 'raw', f"{ticker}.csv")
            data.to_csv(data_file)
            valid_tickers.append(ticker)
    
    logger.info(f"Successfully fetched data for {len(valid_tickers)} tickers")
    
    # Calculate momentum and get winners/losers
    top_momentum_winners, bottom_momentum_losers = returns.get_momentum_winners_and_losers(
        data_dir=temp_data_dir, 
        debug=False,
        top_n=TOP_N
    )
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_data_dir, ignore_errors=True)
    
    return top_momentum_winners, bottom_momentum_losers

def close_all_existing_positions(trading_client):
    """Close all existing positions before rebalancing"""
    logger.info("Closing all existing positions")
    
    # Cancel all open orders first
    trading_client.cancel_orders()
    
    # Close all positions
    close_result = trading_client.close_all_positions(cancel_orders=True)
    logger.info(f"Closed all positions: {len(close_result)} positions affected")
    
    # Wait for positions to close
    time.sleep(10)

def get_current_positions(trading_client):
    """Get current positions as a dictionary mapping symbol to quantity"""
    positions = trading_client.get_all_positions()
    position_dict = {}
    
    for position in positions:
        position_dict[position.symbol] = {
            'qty': float(position.qty),
            'market_value': float(position.market_value),
            'side': 'long' if float(position.qty) > 0 else 'short'
        }
    
    return position_dict

def execute_momentum_trades(trading_client, top_winners, bottom_losers, tradable_assets):
    """Execute trades based on momentum signals"""
    # Get account details
    account = trading_client.get_account()
    buying_power = float(account.buying_power)
    logger.info(f"Account buying power: ${buying_power:.2f}")
    
    # Close existing positions
    close_all_existing_positions(trading_client)
    
    # Calculate number of stocks we can trade (long and short)
    num_long = min(len(top_winners), TOP_N)
    num_short = min(len(bottom_losers), TOP_N)
    
    logger.info(f"Trading {num_long} long positions and {num_short} short positions")
    
    # Long top performers
    for _, row in top_winners.head(num_long).iterrows():
        ticker = row['ticker']
        
        # Skip if not tradable on Alpaca
        if ticker not in tradable_assets:
            logger.warning(f"{ticker} not tradable on Alpaca, skipping")
            continue
        
        # Create market order for long position
        try:
            market_order_data = MarketOrderRequest(
                symbol=ticker,
                notional=POSITION_VALUE,  # Dollar amount to purchase
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            
            # Submit order
            order = trading_client.submit_order(order_data=market_order_data)
            logger.info(f"Submitted BUY order for {ticker}: ${POSITION_VALUE} notional")
        except Exception as e:
            logger.error(f"Error submitting BUY order for {ticker}: {e}")
    
    # Short bottom performers
    for _, row in bottom_losers.head(num_short).iterrows():
        ticker = row['ticker']
        
        # Skip if not tradable or not shortable on Alpaca
        if ticker not in tradable_assets or not tradable_assets[ticker]['shortable']:
            logger.warning(f"{ticker} not shortable on Alpaca, skipping")
            continue
        
        # Create market order for short position
        try:
            market_order_data = MarketOrderRequest(
                symbol=ticker,
                notional=POSITION_VALUE,  # Dollar amount to short
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            # Submit order
            order = trading_client.submit_order(order_data=market_order_data)
            logger.info(f"Submitted SELL (short) order for {ticker}: ${POSITION_VALUE} notional")
        except Exception as e:
            logger.error(f"Error submitting SELL order for {ticker}: {e}")

def rebalance_portfolio():
    """Main function to rebalance the portfolio"""
    try:
        logger.info("Starting portfolio rebalance")
        
        # Load configuration
        config = load_config()
        
        # Initialize Alpaca client
        trading_client = initialize_client(config)
        
        # Get tradable assets
        tradable_assets = get_tradable_assets(trading_client)
        
        # Calculate momentum signals
        top_winners, bottom_losers = fetch_momentum_data()
        
        # Log momentum winners and losers
        logger.info("Top momentum winners:")
        for _, row in top_winners.iterrows():
            logger.info(f"{row['ticker']}: {row['total_return_12m_excluding_most_recent_month']:.2f}%")
        
        logger.info("Bottom momentum losers:")
        for _, row in bottom_losers.iterrows():
            logger.info(f"{row['ticker']}: {row['total_return_12m_excluding_most_recent_month']:.2f}%")
        
        # Execute trades
        execute_momentum_trades(trading_client, top_winners, bottom_losers, tradable_assets)
        
        # Log current positions after rebalance
        time.sleep(30)  # Wait for orders to process
        positions = get_current_positions(trading_client)
        
        logger.info("Current positions after rebalance:")
        for symbol, details in positions.items():
            side = "LONG" if details['side'] == 'long' else "SHORT"
            logger.info(f"{symbol} ({side}): {abs(details['qty'])} shares, ${abs(details['market_value']):.2f}")
        
        # Log account summary
        account = trading_client.get_account()
        logger.info(f"Account equity: ${float(account.equity):.2f}")
        logger.info(f"Cash: ${float(account.cash):.2f}")
        
        logger.info("Portfolio rebalance completed successfully")
        
    except Exception as e:
        logger.error(f"Error during portfolio rebalance: {e}")

def run():
    """Run the trading strategy"""
    logger.info("Starting momentum trading strategy")
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("ALPACA_API_KEY=your_api_key_here\n")
            f.write("ALPACA_SECRET_KEY=your_secret_key_here\n")
        logger.info("Created .env file. Please update with your Alpaca API credentials.")
        return
    
    # Execute initial rebalance
    rebalance_portfolio()
    
    # Schedule daily rebalance at market open (9:30 AM Eastern)
    schedule.every().monday.at("09:35:00").do(rebalance_portfolio)
    schedule.every().tuesday.at("09:35:00").do(rebalance_portfolio)
    schedule.every().wednesday.at("09:35:00").do(rebalance_portfolio)
    schedule.every().thursday.at("09:35:00").do(rebalance_portfolio)
    schedule.every().friday.at("09:35:00").do(rebalance_portfolio)
    
    logger.info("Scheduled daily rebalancing at 9:35 AM Eastern, Monday through Friday")
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    run() 