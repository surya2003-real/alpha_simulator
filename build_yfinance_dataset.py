"""
build_yfinance_dataset.py
Robust yfinance scraper with exponential backoff for timeouts.
"""
import yfinance as yf
import pandas as pd
import requests
from io import StringIO
import time

def get_ticker_list():
    print("Fetching S&P 1500 (Large, Mid, Small cap) ticker lists from Wikipedia...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        resp500 = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers)
        sp500 = pd.read_html(StringIO(resp500.text))[0]
        
        resp400 = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies', headers=headers)
        sp400 = pd.read_html(StringIO(resp400.text))[0]
        
        resp600 = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies', headers=headers)
        sp600 = pd.read_html(StringIO(resp600.text))[0]
        
        all_tickers = sp500['Symbol'].tolist() + sp400['Symbol'].tolist() + sp600['Symbol'].tolist()
        cleaned_tickers = [str(ticker).replace('.', '-') for ticker in all_tickers]
        unique_tickers = list(set(cleaned_tickers))
        
        print(f"Successfully fetched {len(unique_tickers)} unique tickers.")
        return unique_tickers
        
    except Exception as e:
        print(f"Error fetching tickers from Wikipedia: {e}")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "TSLA", "JNJ", "JPM"]

def calculate_factors(df):
    print("\nCalculating technical factors...")
    df = df.sort_values(['ticker', 'date'])
    
    df['returns'] = df.groupby('ticker')['price'].pct_change()
    
    df['momentum_1m'] = df.groupby('ticker')['price'].pct_change(periods=21)
    df['momentum_6m'] = df.groupby('ticker')['price'].pct_change(periods=126)
    df['momentum_12m'] = df.groupby('ticker')['price'].pct_change(periods=252)
    
    df['volatility_20d'] = df.groupby('ticker')['returns'].transform(lambda x: x.rolling(20).std())
    
    df['dollar_volume'] = df['price'] * df['volume']
    df['adv_20d'] = df.groupby('ticker')['dollar_volume'].transform(lambda x: x.rolling(20).mean())
    df['adv'] = df['adv_20d'] 
    
    delta = df.groupby('ticker')['price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.groupby(df['ticker']).transform(lambda x: x.rolling(14).mean())
    avg_loss = loss.groupby(df['ticker']).transform(lambda x: x.rolling(14).mean())
    
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    df['sector'] = "Broad Market" 
    df['market_cap_tier'] = "Large"
    
    return df.dropna(subset=['returns', 'momentum_1m'])

def download_chunk_with_retry(chunk, max_retries=4, base_sleep=5):
    """
    Downloads a list of tickers. If any fail/timeout, it isolates the failures
    and retries them with an exponentially increasing sleep timer.
    """
    pending = chunk.copy()
    chunk_records = []
    
    for attempt in range(max_retries):
        if not pending:
            break
            
        if attempt > 0:
            wait_time = base_sleep * (2 ** (attempt - 1)) # 5s, 10s, 20s...
            print(f"    [Retry {attempt}/{max_retries-1}] Waiting {wait_time}s before retrying {len(pending)} failed tickers...")
            time.sleep(wait_time)
            
        try:
            data = yf.download(
                pending, 
                start="2002-01-01", 
                end="2018-12-31", 
                group_by="ticker", 
                auto_adjust=True,
                threads=True,
                ignore_tz=True,
                timeout=25 # Higher base timeout to give YF breathing room
            )
        except Exception as e:
            print(f"    Exception during network request: {e}")
            continue # Sleep and retry the whole pending list
        
        successful = []
        
        for ticker in pending:
            # Handle yfinance's dynamic return structure based on ticker count
            if len(pending) == 1:
                tdf = data.copy()
            elif ticker in data.columns.levels[0]:
                tdf = data[ticker].copy()
            else:
                continue
                
            if tdf.empty:
                continue
                
            tdf = tdf.reset_index()
            tdf['ticker'] = ticker
            
            # Normalize casing
            col_map = {c: str(c).lower() for c in tdf.columns}
            tdf = tdf.rename(columns=col_map)
            
            if 'date' in tdf.columns and 'close' in tdf.columns and 'volume' in tdf.columns:
                tdf = tdf.rename(columns={'close': 'price'})
                chunk_records.append(tdf[['date', 'ticker', 'price', 'volume']])
                successful.append(ticker)
                
        # Remove the successful ones so we only retry the timeouts
        pending = [t for t in pending if t not in successful]
        
    if pending:
        print(f"    [!] Permanently failed on {len(pending)} tickers after {max_retries} attempts: {pending}")
        
    return chunk_records

def main():
    tickers = get_ticker_list()
    print(f"\nAttempting to download history for {len(tickers)} tickers (2002-2018)...")
    
    chunk_size = 50
    ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    all_records = []
    
    for i, chunk in enumerate(ticker_chunks):
        print(f"Downloading batch {i+1}/{len(ticker_chunks)}...")
        
        records = download_chunk_with_retry(chunk)
        all_records.extend(records)
        
        # Standard polite delay between healthy chunks
        time.sleep(1)
            
    if not all_records:
        print("Fatal Error: Failed to download any data across all batches.")
        return
        
    print(f"\nDownload complete. Compiling data for {len(all_records)} successful ticker fragments...")
    raw_df = pd.concat(all_records, ignore_index=True)
    
    final_df = calculate_factors(raw_df)
    
    out_path = "universe_yfinance.parquet"
    final_df.to_parquet(out_path, index=False)
    print(f"Success! Saved backtest dataset to {out_path}")

if __name__ == "__main__":
    main()