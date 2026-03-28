"""
data_loader.py
Loads real historical stock data from the yfinance parquet file.
"""

import pandas as pd
import yfinance as yf
from functools import lru_cache
import os

SECTORS = ["Broad Market"] # Simplified since we lack historical point-in-time sector mapping
MARKET_CAPS = ["Large", "Mid", "Small"]

DATA_PATH = os.getenv("ALPHASIM_DATA_PATH", "universe_yfinance.parquet")

@lru_cache(maxsize=1)
def generate_universe() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Data file not found at {DATA_PATH}. "
            "Please run build_yfinance_dataset.py first."
        )

    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    df["returns_lag1"] = df.groupby("ticker")["returns"].shift(1)
    return df

@lru_cache(maxsize=1)
def _fetch_spy_data() -> pd.Series:
    """Fetches real SPY data and caches it in memory."""
    try:
        # Fetch a wide enough range to cover any backtest
        spy = yf.download("SPY", start="2000-01-01", end="2020-01-01", progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy_rets = spy["Close"].pct_change().dropna()
        spy_rets.index = pd.to_datetime(spy_rets.index).tz_localize(None)
        return spy_rets
    except Exception as e:
        print(f"Warning: Could not fetch SPY benchmark ({e}).")
        return pd.Series(dtype=float)

def get_benchmark_returns(dates: pd.DatetimeIndex) -> pd.Series:
    """Aligns the cached SPY data to the requested backtest dates."""
    spy_rets = _fetch_spy_data()
    
    if spy_rets.empty:
        return pd.Series(0.0, index=dates, name="benchmark")
        
    aligned = spy_rets.reindex(dates).fillna(0)
    aligned.name = "benchmark"
    return aligned