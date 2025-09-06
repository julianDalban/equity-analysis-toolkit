'''
File built for OHLCV data collection using the Alpha Vantage API.

The goal of this file is to provide robust and convenient data collection capabilities.
Implemented features include caching, rate limiting, and fallback mechanisms to ensure relevant and 
fresh data on financial markets.
'''

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import requests
import time
from typing import Optional, Tuple, Union

class DataManager:
    def __init__(self, api_key: str, cache_dir: str = "../data_cache"):
        self.api_key = api_key
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _normalize_date_input(self, date_input: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
        """Converts various date inputs to pandas Timestamp"""
        if isinstance(date_input, str):
            if date_input.upper() == 'TODAY':
                # Get last business day (accounts for weekends automatically)
                return pd.Timestamp.today().normalize()
            else:
                return pd.to_datetime(date_input)
        elif isinstance(date_input, datetime):
            return pd.Timestamp(date_input)
        elif isinstance(date_input, pd.Timestamp):
            return date_input
        else:
            raise ValueError(f"Unsupported date format: {type(date_input)}")
    
    def _get_cache_file_path(self, symbol: str) -> str:
        """Get the file path for cached data"""
        return os.path.join(self.cache_dir, f"{symbol.upper()}_daily.csv")
    
    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load cached data for a symbol"""
        cache_file = self._get_cache_file_path(symbol)
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
            return df
        except Exception as e:
            print(f"Error loading cache for {symbol}: {e}")
            return None

    
    
    def _get_cache_date_range(self, cached_data: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get the date range of cached data"""
        if cached_data is None or cached_data.empty:
            return None, None
        
        return cached_data.index.min(), cached_data.index.max()
    
    def _needs_refresh(self, symbol: str, requested_start: Union[str, datetime, pd.Timestamp], requested_end: Union[str, datetime, pd.Timestamp]) -> bool:
        """
        Function for determining whether cached data on a specific ticker needs a refresh.
        This is determined based on the time range of the cache and the requested time frame.
        
        Returns:
            True if refresh needed, False if cache is sufficient
        """
        # NOTE: The logic here is designed to refresh under most circumstances (i.e. if req range is not within cache period),
        #       as we are assuming that more often than not time ranges will be month denominated (1mo, 3mo, 6mo), so it is typically
        #       better to refresh and have an extra few days even if the range diff is small (since a few extra days can change our outlook
        #       given the smaller time frame). The logic can easily be changed to change this consideration.
        
        # Normalize dates
        req_start = self._normalize_date_input(requested_start)
        req_end = self._normalize_date_input(requested_end)
        
        # Ensure we're dealing with business days only
        req_start = pd.bdate_range(start=req_start, periods=1)[0]
        req_end = pd.bdate_range(end=req_end, periods=1)[0]
        
        # Load cached data
        cached_data = self._load_from_cache(symbol)
        if cached_data is None:
            print(f"No cached data for {symbol} - refresh needed")
            return True
        
        cached_start, cached_end = self._get_cache_date_range(cached_data)
        print(f"Cache range: {cached_start} to {cached_end}")
        print(f"Request range: {req_start} to {req_end}")
        
        # Calculate request period length
        request_period_days = (req_end - req_start).days
        
        # Case 1: Request is fully within cached range
        if cached_start <= req_start and req_end <= cached_end:
            # For short-term requests (< 5 days), always refresh to get latest data
            if request_period_days < 5:
                print(f"Short-term request ({request_period_days} days) - refreshing for latest data")
                return True
            else:
                print("Request fully covered by cache - using cached data")
                return False
        
        # Case 2: Partial overlap scenarios
        start_within = cached_start <= req_start <= cached_end
        end_within = cached_start <= req_end <= cached_end
        
        if start_within and not end_within:
            # Need data beyond cached end date
            gap_days = (req_end - cached_end).days
            if gap_days > 30:  # More than a month gap
                print(f"Large gap beyond cache ({gap_days} days) - refresh needed")
                return True
            else:
                print(f"Small gap beyond cache ({gap_days} days) - refresh for extension")
                return True
        elif end_within and not start_within:
            # Need data before cached start date
            gap_days = (cached_start - req_start).days
            if gap_days > 30:
                print(f"Large gap before cache ({gap_days} days) - refresh needed")
                return True
            else:
                print(f"Small gap before cache ({gap_days} days) - refresh for extension")
                return True
        
        # Case 3: No overlap - completely outside cached range
        start_gap = min(abs((cached_start - req_start).days), abs((cached_end - req_start).days))
        end_gap = min(abs((cached_start - req_end).days), abs((cached_end - req_end).days))
        
        # If gaps are small (within a month), might be worth refreshing to get continuous data
        if start_gap < 30 and end_gap < 30:
            print("Request close to cached range - refresh for continuous data")
            return True
        else:
            print("Request far from cached range - refresh needed")
            return True

    def _fetch_from_api(self, symbol: str, outputsize: str = 'full') -> pd.DataFrame:
        """
        Fetch data from Alpha Vantage API and return clean DataFrame, indexed by timestamp
        """
        print(f"Fetching {symbol} from API...")
        time.sleep(1)  # Rate limiting
        
        symbol = symbol.upper()
        
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': outputsize
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            
            if 'Note' in data:
                raise ValueError(f"API Rate Limit: {data['Note']}")
            
            # Extract time series data
            time_series_key = 'Time Series (Daily)'
            if time_series_key not in data:
                available_keys = list(data.keys())
                raise ValueError(f"Expected key '{time_series_key}' not found. Available keys: {available_keys}")
            
            time_series_data = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series_data, orient='index')
            
            # Clean up column names
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Convert to proper data types
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert index to datetime and sort
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            print(f"Successfully fetched {len(df)} days of data for {symbol}")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching {symbol}: {e}")
            raise
        except Exception as e:
            print(f"Error processing {symbol} data: {e}")
            raise
    
    def _save_to_cache(self, symbol: str, df: pd.DataFrame):
        """Save DataFrame to CSV cache"""
        cache_file = self._get_cache_file_path(symbol)
        try:
            df.to_csv(cache_file, index_label='Date')
            print(f"Cached {len(df)} days of data for {symbol}")
        except Exception as e:
            print(f"Error saving cache for {symbol}: {e}")
    
    def get_stock_data(self, symbol: str, start_date: Union[str, datetime, pd.Timestamp] = None, end_date: Union[str, datetime, pd.Timestamp] = "TODAY", force_refresh: bool = False) -> pd.DataFrame:
        """
        Main method to get stock data 
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            start_date: Start date for data (default: 2 years ago)
            end_date: End date for data (default: today)
            force_refresh: If True, always fetch from API regardless of cache
            
        Returns:
            DataFrame with OHLCV data
        """
        symbol = symbol.upper()
        
        # Set default start date
        if start_date is None:
            start_date = pd.Timestamp.today() - pd.DateOffset(years=2)
        
        # Normalize dates
        start_date = self._normalize_date_input(start_date)
        end_date = self._normalize_date_input(end_date)
        
        print(f"\n=== Getting {symbol} data from {start_date.date()} to {end_date.date()} ===")
        
        # Check if refresh is needed (unless forced)
        if not force_refresh and not self._needs_refresh(symbol, start_date, end_date):
            print("Using cached data")
            cached_data = self._load_from_cache(symbol)
            # Filter to requested date range
            mask = (cached_data.index >= start_date) & (cached_data.index <= end_date)
            return cached_data.loc[mask]
        
        # Fetch fresh data from API
        try:
            # For requests within last 100 days, use compact to save API calls
            days_requested = (end_date - start_date).days
            outputsize = 'compact' if days_requested <= 100 else 'full'
            
            fresh_data = self._fetch_from_api(symbol, outputsize=outputsize)
            
            # Save to cache
            self._save_to_cache(symbol, fresh_data)
            
            # Filter to requested date range
            mask = (fresh_data.index >= start_date) & (fresh_data.index <= end_date)
            filtered_data = fresh_data.loc[mask]
            
            print(f"Returning {len(filtered_data)} days of data")
            return filtered_data
            
        except Exception as e:
            print(f"API fetch failed: {e}")
            # Try to fall back to cache
            print("Attempting to use cached data as fallback...")
            cached_data = self._load_from_cache(symbol)
            if cached_data is not None:
                print("Using cached data as fallback")
                mask = (cached_data.index >= start_date) & (cached_data.index <= end_date)
                return cached_data.loc[mask]
            else:
                print("No cached data available")
                raise
