# XBI Momentum Analysis

A tool for analyzing momentum of stocks in the SPDR S&P Biotech ETF (XBI).

## Overview

This project analyzes momentum for stocks in the XBI ETF by:

1. Loading XBI holdings from a CSV file
2. Fetching historical price data for each constituent stock
3. Calculating momentum using a 12-month lookback period (excluding the most recent month)
4. Processing and saving the results
5. Generating visualizations and statistics
6. Optionally executing a trading strategy via Alpaca

## Setup with Git and VS Code

### Clone the Repository

```bash
git clone https://github.com/yourusername/xbi_momentum.git
cd xbi_momentum
```

### Open in VS Code

```bash
code .
```

## Installation with uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create and Activate Virtual Environment

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Dependencies

```bash
uv pip install -r requirements.txt
```

## Usage

Run the momentum analysis with default settings:

```bash
python main.py
```

This will:
- Fetch historical data for XBI constituent stocks
- Calculate momentum using a 12-month lookback period
- Perform backtests for both daily and weekly rebalancing
- Generate performance metrics and plots
- Save results to the `data/results/` directory

## Alpaca Trading (Experimental)

The repository includes an untested implementation for automated trading using the Alpaca API.

> **Note**: The Alpaca trading functionality is experimental and has not been fully tested.

### Setup for Alpaca Trading

1. Create a `.env` file in the project root with your Alpaca API credentials:

```
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
```

2. Run the Alpaca trading module:

```bash
python alpaca.py
```

This will:
- Connect to your Alpaca account
- Calculate momentum signals for XBI stocks
- Execute trades based on momentum rankings
- Schedule regular portfolio rebalancing

## Command-line Options (Future Development)

- `--window DAYS`: Window size for momentum calculation in trading days
- `--top-n N`: Number of top momentum stocks to display/trade
- `--plot`: Generate plots of momentum over time
- `--debug`: Enable debug output with detailed information

## Project Structure

```
xbi_momentum/
│
├── main.py                  # Main entry point for backtesting
├── alpaca.py                # Automated trading via Alpaca API
├── requirements.txt         # Project dependencies
├── .env                     # Environment variables (create this file)
│
├── data/                    # Data directory
│   ├── holdings/            # XBI holdings file
│   ├── raw/                 # Raw price data files
│   ├── returns/             # Processed returns data
│   └── results/             # Analysis results
│
└── utils/                   # Utility modules
    ├── data_fetching/       # Data fetching utilities
    ├── calculations/        # Calculation utilities
    └── viz/                 # Visualization utilities
```

## Methodology

1. **Data Collection**: Historical price data is fetched for each stock in the XBI ETF using Yahoo Finance.
2. **Momentum Calculation**: Momentum is calculated as the total return over a 12-month period, excluding the most recent month.
3. **Strategy Implementation**: 
   - Long positions in top momentum performers
   - Short positions in bottom momentum performers
   - Regular rebalancing (daily or weekly)
4. **Performance Metrics**: Total return, annualized return, max drawdown, Sharpe ratio, transaction costs.

## License

MIT