import pandas as pd
import numpy as np
from scipy.stats import norm
import os

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()

def black_scholes_greeks(S, K, T, r, sigma):
    """
    Calculates Greeks for an At-The-Money (ATM) call option.
    S: Stock price
    K: Strike price (assumed S for ATM)
    T: Time to expiration in years
    r: Risk-free rate
    sigma: Volatility
    """
    if sigma <= 0 or T <= 0 or np.isnan(sigma):
        return np.nan, np.nan, np.nan, np.nan

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    delta = norm.cdf(d1)
    
    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta (Call)
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    
    # Vega
    vega = (S * norm.pdf(d1) * np.sqrt(T)) / 100

    return delta, gamma, theta, vega

def compute_features(df, spy_df=None):
    """
    Computes technical indicators and synthetic Greeks for a given stock dataframe.
    """
    df = df.copy()
    
    # Technical Indicators
    df['RSI'] = compute_rsi(df['Close'])
    
    macd, _ = compute_macd(df['Close'])
    df['MACD'] = macd
    
    # Bollinger Bands
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BBU'] = sma20 + 2 * std20
    df['BBL'] = sma20 - 2 * std20
    df['BB_Pos'] = (df['Close'] - df['BBL']) / (df['BBU'] - df['BBL'])
    
    df['SMA_20_Ratio'] = df['Close'] / sma20
    df['SMA_50_Ratio'] = df['Close'] / df['Close'].rolling(window=50).mean()
    
    df['ATR_Norm'] = compute_atr(df) / df['Close']
    
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    df['Momentum'] = df['Close'].pct_change(20)
    
    roll_low = df['Close'].rolling(window=252).min()
    roll_high = df['Close'].rolling(window=252).max()
    df['Range_Pos'] = (df['Close'] - roll_low) / (roll_high - roll_low)
    
    # Synthetic Greeks
    df['Returns'] = df['Close'].pct_change()
    df['IV'] = df['Returns'].rolling(window=30).std() * np.sqrt(252)
    
    # IV Rank
    roll_iv_low = df['IV'].rolling(window=252).min()
    roll_iv_high = df['IV'].rolling(window=252).max()
    df['IV_Rank'] = (df['IV'] - roll_iv_low) / (roll_iv_high - roll_iv_low)
    
    # Apply Black-Scholes
    r = 0.05
    T = 30/365
    
    greeks = []
    for i in range(len(df)):
        S = df['Close'].iloc[i]
        sigma = df['IV'].iloc[i]
        if pd.isna(S) or pd.isna(sigma):
            greeks.append((np.nan, np.nan, np.nan, np.nan))
        else:
            greeks.append(black_scholes_greeks(S, S, T, r, sigma))
            
    greeks_df = pd.DataFrame(greeks, columns=['Delta', 'Gamma', 'Theta', 'Vega'], index=df.index)
    df = pd.concat([df, greeks_df], axis=1)
    
    # Target Variable: 1-week forward excess return
    if spy_df is not None:
        df['Fwd_Ret'] = df['Close'].shift(-5) / df['Close'] - 1
        spy_close = spy_df['Close'].reindex(df.index)
        spy_fwd_ret = spy_close.shift(-5) / spy_close - 1
        df['Alpha'] = df['Fwd_Ret'] - spy_fwd_ret
    
    return df

if __name__ == "__main__":
    from data_loader import get_stock_data
    
    tickers = ["AAPL", "SPY"]
    data = get_stock_data(tickers)
    
    aapl = data["AAPL"]
    spy = data["SPY"]
    
    features = compute_features(aapl, spy)
    
    mask = (features.index >= '2009-01-01') & (features.index <= '2009-12-31')
    result = features.loc[mask]
    
    print(result[['Close', 'RSI', 'MACD', 'BB_Pos', 'IV_Rank', 'Delta', 'Gamma', 'Theta', 'Vega', 'Alpha']].tail(20))
    
    mask = (features.index >= '2009-01-01') & (features.index <= '2009-12-31')
    result = features.loc[mask]
    
    print(result[['Close', 'RSI', 'MACD', 'BB_Pos', 'IV_Rank', 'Delta', 'Gamma', 'Theta', 'Vega', 'Alpha']].tail(20))
