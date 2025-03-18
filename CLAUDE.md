# CLAUDE.md - XBI Momentum Analysis

## Commands
- Run main analysis: `python main.py`
- Run with options: `python main.py --window 126 --top-n 15 --plot --debug`
- Run live trading: `python alpaca.py`
- Install dependencies: `pip install -r requirements.txt`

## Code Style
- **Imports**: Group standard library imports first, third-party packages second, local modules last
- **Formatting**: Use 4-space indentation; line length ~100 characters
- **Types**: Use docstrings with type hints in Parameters/Returns sections
- **Variables**: Use snake_case for variables and functions; ALL_CAPS for constants
- **Error handling**: Use try/except blocks with specific exception types; log errors
- **Docstrings**: Follow NumPy docstring style with Parameters and Returns sections
- **Functions**: Create small, focused functions with clear responsibilities
- **Logging**: Use the logging module instead of print statements in production code

## Notes
- yfinance already auto adjusts the close (from .cursorrules)
- Ensure error handling includes appropriate debugging info when debug=True
- Keep data processing and financial calculations in separate utility modules