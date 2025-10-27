import yfinance as yf
import pandas as pd
import numpy as np 
import sys

def calculate_rsi(series, period=14):
    """Calculates the Relative Strength Index (RSI) using Pandas.
    It uses Exponential Moving Average (EMA) smoothing for the gain/loss components, 
    which is the standard approach for RSI (Wilder's smoothing approximation)."""
    
    # 1. Calculate price changes
    delta = series.diff().dropna()

    # 2. Separate into gains (up) and losses (down)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # 3. Calculate EMA (Wilder's smoothing approximation) for gain/loss
    # com = period - 1 is equivalent to alpha = 1 / period
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    
    # 4. Calculate Relative Strength (RS) and RSI
    # Use np.divide for safe division, handling cases where avg_loss might be zero
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def fetch_and_analyze_forex_data(pairs, period='60d', interval='1d', RSI_LENGTH=14, EMA_SHORT=20, EMA_LONG=50):
    """
    Fetches historical data for specified Forex and Metal pairs, calculates
    technical indicators (RSI and EMA Crossover), and generates a screener report.
    
    Args:
        pairs (list): List of Yahoo Finance ticker symbols.
        period (str): Period over which to fetch data (e.g., '90d').
        interval (str): Data interval (e.g., '1d').
        RSI_LENGTH (int): Period for RSI calculation.
        EMA_SHORT (int): Period for the short Exponential Moving Average.
        EMA_LONG (int): Period for the long Exponential Moving Average.
    
    Returns:
        pd.DataFrame: A DataFrame containing the screener results and signals.
    """
    print(f"--- Fetching {interval} data for the last {period}... ---")

    try:
        # Fetch data for all pairs at once. 
        data = yf.download(pairs, period=period, interval=interval, progress=False)
    except Exception as e:
        # Catch unexpected API errors
        print(f"\nCritical Error during data fetching: {e}")
        return pd.DataFrame()

    # The yfinance output for multiple tickers is a MultiIndex DataFrame.
    
    # 1. Isolate the 'Close' prices
    # Use .get() defensively in case 'Close' is not in the columns
    close_prices = data.get('Close', pd.DataFrame()).copy()
    
    # If the download was for a single ticker, close_prices might be the main DataFrame
    if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
        # Rename the single column to the ticker name for consistency
        close_prices = data[['Close']].rename(columns={'Close': pairs[0]})
        
    if close_prices.empty:
        print("No historical data available for the specified pairs/period.")
        return pd.DataFrame()

    # 2. Robust Data Filtering (Crucial Fix)
    # Drop columns (pairs) that are entirely NaN (i.e., failed to download)
    close_prices = close_prices.dropna(axis=1, how='all')
    
    # Drop rows that contain NaNs in the remaining data (required for rolling calculations)
    close_prices = close_prices.dropna(axis=0)

    if close_prices.empty:
        print("After cleaning failed tickers and missing values, no complete data remains for analysis.")
        return pd.DataFrame()

    # Ensure we have enough data points for the longest calculation (EMA_LONG)
    min_data_points = max(RSI_LENGTH, EMA_LONG)
    if len(close_prices) < min_data_points:
        print(f"Insufficient data ({len(close_prices)} rows) for required indicator periods (max is {min_data_points}). Please use a longer period or interval.")
        return pd.DataFrame()


    # Initialize the results DataFrame
    results = pd.DataFrame(index=close_prices.columns)

    print(f"--- Analyzing {len(close_prices.columns)} pairs... ---")

    for pair in close_prices.columns:
        # 1. Calculate RSI 
        rsi_series = calculate_rsi(close_prices[pair], period=RSI_LENGTH)
        # We need the last non-NaN RSI value
        latest_rsi = rsi_series.iloc[-1]
        results.loc[pair, f'RSI ({RSI_LENGTH})'] = round(latest_rsi, 2)
        
        # 2. Calculate EMAs (using Pandas Exponential Weighted Moving)
        # FIX: Changed from .rolling().mean() (SMA) to .ewm(span=...).mean() (EMA)
        ema_short_series = close_prices[pair].ewm(span=EMA_SHORT, adjust=False).mean()
        ema_long_series = close_prices[pair].ewm(span=EMA_LONG, adjust=False).mean()
        
        # Get latest values for EMA (FIX: Used correct variable names)
        latest_ema_short = ema_short_series.iloc[-1]
        latest_ema_long = ema_long_series.iloc[-1]
        
        # Populate results with latest prices and EMAs
        results.loc[pair, 'Latest Price'] = round(close_prices[pair].iloc[-1], 5)
        results.loc[pair, f'EMA {EMA_SHORT}'] = round(latest_ema_short, 5)
        results.loc[pair, f'EMA {EMA_LONG}'] = round(latest_ema_long, 5)
        
        # 3. Generate Signals
        
        # --- RSI Signal ---
        if latest_rsi < 30:
            rsi_signal = 'Oversold (BUY)'
        elif latest_rsi > 70:
            rsi_signal = 'Overbought (SELL)'
        else:
            rsi_signal = 'Neutral'
            
        results.loc[pair, 'RSI Signal'] = rsi_signal
        
        # --- EMA Crossover Signal ---
        ema_signal = 'Neutral'
        # Check if Short EMA is above Long EMA (Bullish)
        if latest_ema_short > latest_ema_long:
            ema_signal = 'Bullish Trend'
        # Check if Short EMA is below Long EMA (Bearish)
        elif latest_ema_short < latest_ema_long:
            ema_signal = 'Bearish Trend'
            
        results.loc[pair, 'EMA Signal'] = ema_signal

    # Final formatting of the DataFrame for display
    results.index.name = f'Pair ({interval} over {period})'
    
    # Define the final recommendation based on a consensus
    def get_final_recommendation(row):
        rsi = row['RSI Signal']
        ema = row['EMA Signal']
        
        if rsi == 'Oversold (BUY)' and ema == 'Bullish Trend':
            return 'STRONG BUY (RSI & Trend Match)'
        elif rsi == 'Overbought (SELL)' and ema == 'Bearish Trend':
            return 'STRONG SELL (RSI & Trend Match)'
        elif rsi.endswith('(BUY)') and ema == 'Bearish Trend':
            return 'Potential BUY (Against Trend)'
        elif rsi.endswith('(SELL)') and ema == 'Bullish Trend':
            return 'Potential SELL (Against Trend)'
        elif 'Trend' in ema:
            return f'Trend Following ({ema.split(" ")[0]})'
        else:
            return 'Monitor (Mixed/Neutral)'

    results['Recommendation'] = results.apply(get_final_recommendation, axis=1)

    # Reorder columns for better readability
    final_cols = ['Latest Price', f'RSI ({RSI_LENGTH})', 'RSI Signal', f'EMA {EMA_SHORT}', f'EMA {EMA_LONG}', 'EMA Signal', 'Recommendation']
    
    return results[final_cols]

# -------------------- MAIN EXECUTION --------------------

if __name__ == "__main__":
    FOREX_PAIRS_TICKERS = [
        'GBPUSD=X',
        'GLD=X',
        'USDJPY=X',
        'EURUSD=X',
        'AUDUSD=X',
        'NZDUSD=X',
        'USDCHF=X',
        'EURGBP=X',
        'GBPJPY=X',
        'GBPCAD=X',
        'USDCAD=X'
    ]
    
    # Configuration
    DATA_PERIOD = '90d'  # 90 days of history
    DATA_INTERVAL = '1d' # Daily closing data

    # Define indicator parameters (as requested by user)
    RSI_LENGTH = 5
    EMA_SHORT = 20
    EMA_LONG = 40

    # Check if necessary libraries are present before running
    try:
        import yfinance
        import pandas
        import numpy
    except ImportError:
        print("Required libraries missing.")
        # Updated installation instructions
        print("Please install them using: pip install yfinance pandas numpy")
        sys.exit(1)

    # Run the screener - Pass the indicator parameters
    screener_df = fetch_and_analyze_forex_data(
        FOREX_PAIRS_TICKERS, 
        period=DATA_PERIOD, 
        interval=DATA_INTERVAL,
        RSI_LENGTH=RSI_LENGTH,
        EMA_SHORT=EMA_SHORT,
        EMA_LONG=EMA_LONG
    )

    if not screener_df.empty:
        print("\n" + "="*80)
        print(f"                 DAILY QUANTITATIVE FOREX VOLATILITY & TRNED SCREENER")
        print("="*80)
        # Set display options to ensure the DataFrame is fully visible
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'center')
        
        print(screener_df)
        print("="*80)
        print(f"RSI Check: <30 (Oversold/Buy), >70 (Overbought/Sell) on {RSI_LENGTH}-period RSI.")
        # This line now correctly uses the variables defined above:
        print(f"EMA Check: EMA {EMA_SHORT} ({EMA_SHORT}-period) > EMA {EMA_LONG} ({EMA_LONG}-period) for Bullish Trend.")

import yfinance as yf
import pandas as pd
import numpy as np 
import sys

def calculate_rsi(series, period=14):
    """Calculates the Relative Strength Index (RSI) using Pandas.
    It uses Exponential Moving Average (EMA) smoothing for the gain/loss components, 
    which is the standard approach for RSI (Wilder's smoothing approximation)."""
    
    # 1. Calculate price changes
    delta = series.diff().dropna()

    # 2. Separate into gains (up) and losses (down)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # 3. Calculate EMA (Wilder's smoothing approximation) for gain/loss
    # com = period - 1 is equivalent to alpha = 1 / period
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    
    # 4. Calculate Relative Strength (RS) and RSI
    # Use np.divide for safe division, handling cases where avg_loss might be zero
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def fetch_and_analyze_forex_data(pairs, period='60d', interval='1d', RSI_LENGTH=14, EMA_SHORT=20, EMA_LONG=50):
    """
    Fetches historical data for specified Forex and Metal pairs, calculates
    technical indicators (RSI and EMA Crossover), and generates a screener report.
    
    Args:
        pairs (list): List of Yahoo Finance ticker symbols.
        period (str): Period over which to fetch data (e.g., '90d').
        interval (str): Data interval (e.g., '1d').
        RSI_LENGTH (int): Period for RSI calculation.
        EMA_SHORT (int): Period for the short Exponential Moving Average.
        EMA_LONG (int): Period for the long Exponential Moving Average.
    
    Returns:
        pd.DataFrame: A DataFrame containing the screener results and signals.
    """
    print(f"--- Fetching {interval} data for the last {period}... ---")

    try:
        # Fetch data for all pairs at once. 
        data = yf.download(pairs, period=period, interval=interval, progress=False)
    except Exception as e:
        # Catch unexpected API errors
        print(f"\nCritical Error during data fetching: {e}")
        return pd.DataFrame()

    # The yfinance output for multiple tickers is a MultiIndex DataFrame.
    
    # 1. Isolate the 'Close' prices
    # Use .get() defensively in case 'Close' is not in the columns
    close_prices = data.get('Close', pd.DataFrame()).copy()
    
    # If the download was for a single ticker, close_prices might be the main DataFrame
    if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
        # Rename the single column to the ticker name for consistency
        close_prices = data[['Close']].rename(columns={'Close': pairs[0]})
        
    if close_prices.empty:
        print("No historical data available for the specified pairs/period.")
        return pd.DataFrame()

    # 2. Robust Data Filtering (Crucial Fix)
    # Drop columns (pairs) that are entirely NaN (i.e., failed to download)
    close_prices = close_prices.dropna(axis=1, how='all')
    
    # Drop rows that contain NaNs in the remaining data (required for rolling calculations)
    close_prices = close_prices.dropna(axis=0)

    if close_prices.empty:
        print("After cleaning failed tickers and missing values, no complete data remains for analysis.")
        return pd.DataFrame()

    # Ensure we have enough data points for the longest calculation (EMA_LONG)
    min_data_points = max(RSI_LENGTH, EMA_LONG)
    if len(close_prices) < min_data_points:
        print(f"Insufficient data ({len(close_prices)} rows) for required indicator periods (max is {min_data_points}). Please use a longer period or interval.")
        return pd.DataFrame()


    # Initialize the results DataFrame
    results = pd.DataFrame(index=close_prices.columns)

    print(f"--- Analyzing {len(close_prices.columns)} pairs... ---")

    for pair in close_prices.columns:
        # 1. Calculate RSI 
        rsi_series = calculate_rsi(close_prices[pair], period=RSI_LENGTH)
        # We need the last non-NaN RSI value
        latest_rsi = rsi_series.iloc[-1]
        results.loc[pair, f'RSI ({RSI_LENGTH})'] = round(latest_rsi, 2)
        
        # 2. Calculate EMAs (using Pandas Exponential Weighted Moving)
        # FIX: Changed from .rolling().mean() (SMA) to .ewm(span=...).mean() (EMA)
        ema_short_series = close_prices[pair].ewm(span=EMA_SHORT, adjust=False).mean()
        ema_long_series = close_prices[pair].ewm(span=EMA_LONG, adjust=False).mean()
        
        # Get latest values for EMA (FIX: Used correct variable names)
        latest_ema_short = ema_short_series.iloc[-1]
        latest_ema_long = ema_long_series.iloc[-1]
        
        # Populate results with latest prices and EMAs
        results.loc[pair, 'Latest Price'] = round(close_prices[pair].iloc[-1], 5)
        results.loc[pair, f'EMA {EMA_SHORT}'] = round(latest_ema_short, 5)
        results.loc[pair, f'EMA {EMA_LONG}'] = round(latest_ema_long, 5)
        
        # 3. Generate Signals
        
        # --- RSI Signal ---
        if latest_rsi < 30:
            rsi_signal = 'Oversold (BUY)'
        elif latest_rsi > 70:
            rsi_signal = 'Overbought (SELL)'
        else:
            rsi_signal = 'Neutral'
            
        results.loc[pair, 'RSI Signal'] = rsi_signal
        
        # --- EMA Crossover Signal ---
        ema_signal = 'Neutral'
        # Check if Short EMA is above Long EMA (Bullish)
        if latest_ema_short > latest_ema_long:
            ema_signal = 'Bullish Trend'
        # Check if Short EMA is below Long EMA (Bearish)
        elif latest_ema_short < latest_ema_long:
            ema_signal = 'Bearish Trend'
            
        results.loc[pair, 'EMA Signal'] = ema_signal

    # Final formatting of the DataFrame for display
    results.index.name = f'Pair ({interval} over {period})'
    
    # Define the final recommendation based on a consensus
    def get_final_recommendation(row):
        rsi = row['RSI Signal']
        ema = row['EMA Signal']
        
        if rsi == 'Oversold (BUY)' and ema == 'Bullish Trend':
            return 'STRONG BUY (RSI & Trend Match)'
        elif rsi == 'Overbought (SELL)' and ema == 'Bearish Trend':
            return 'STRONG SELL (RSI & Trend Match)'
        elif rsi.endswith('(BUY)') and ema == 'Bearish Trend':
            return 'Potential BUY (Against Trend)'
        elif rsi.endswith('(SELL)') and ema == 'Bullish Trend':
            return 'Potential SELL (Against Trend)'
        elif 'Trend' in ema:
            return f'Trend Following ({ema.split(" ")[0]})'
        else:
            return 'Monitor (Mixed/Neutral)'

    results['Recommendation'] = results.apply(get_final_recommendation, axis=1)

    # Reorder columns for better readability
    final_cols = ['Latest Price', f'RSI ({RSI_LENGTH})', 'RSI Signal', f'EMA {EMA_SHORT}', f'EMA {EMA_LONG}', 'EMA Signal', 'Recommendation']
    
    return results[final_cols]

# -------------------- MAIN EXECUTION --------------------

if __name__ == "__main__":
    FOREX_PAIRS_TICKERS = [
        'GBPUSD=X',
        'GLD=X',
        'USDJPY=X',
        'EURUSD=X',
        'AUDUSD=X',
        'NZDUSD=X',
        'USDCHF=X',
        'EURGBP=X',
        'GBPJPY=X',
        'GBPCAD=X',
        'USDCAD=X'
    ]
    
    # Configuration
    DATA_PERIOD = '90d'  # 90 days of history
    DATA_INTERVAL = '1d' # Daily closing data

    # Define indicator parameters (as requested by user)
    RSI_LENGTH = 5
    EMA_SHORT = 20
    EMA_LONG = 40

    # Check if necessary libraries are present before running
    try:
        import yfinance
        import pandas
        import numpy
    except ImportError:
        print("Required libraries missing.")
        # Updated installation instructions
        print("Please install them using: pip install yfinance pandas numpy")
        sys.exit(1)

    # Run the screener - Pass the indicator parameters
    screener_df = fetch_and_analyze_forex_data(
        FOREX_PAIRS_TICKERS, 
        period=DATA_PERIOD, 
        interval=DATA_INTERVAL,
        RSI_LENGTH=RSI_LENGTH,
        EMA_SHORT=EMA_SHORT,
        EMA_LONG=EMA_LONG
    )

    if not screener_df.empty:
        print("\n" + "="*80)
        print(f"                 DAILY QUANTITATIVE FOREX VOLATILITY & TRNED SCREENER")
        print("="*80)
        # Set display options to ensure the DataFrame is fully visible
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'center')
        
        print(screener_df)
        print("="*80)
        print(f"RSI Check: <30 (Oversold/Buy), >70 (Overbought/Sell) on {RSI_LENGTH}-period RSI.")
        # This line now correctly uses the variables defined above:
        print(f"EMA Check: EMA {EMA_SHORT} ({EMA_SHORT}-period) > EMA {EMA_LONG} ({EMA_LONG}-period) for Bullish Trend.")

