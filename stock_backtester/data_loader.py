import yfinance as yf
import pandas as pd
import os

def get_stock_data(tickers, start_date="2009-01-01", data_dir="data"):
    """
    Pulls historical stock data for the given tickers and caches them as CSVs.
    """
    data_path = os.path.join(os.getcwd(), data_dir)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    all_data = {}
    
    for ticker in tickers:
        file_path = os.path.join(data_path, f"{ticker}.csv")
        
        if os.path.exists(file_path):
            print(f"Loading {ticker} from cache...")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            # Ensure numeric types
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Downloading {ticker} from yfinance...")
            t = yf.Ticker(ticker)
            df = t.history(start=start_date)
            if not df.empty:
                df.to_csv(file_path)
        
        if not df.empty:
            all_data[ticker] = df
            
    return all_data

if __name__ == "__main__":
    # Quick test
    tickers = ["AAPL", "SPY"]
    data = get_stock_data(tickers)
    print(f"Loaded {len(data)} tickers.")
