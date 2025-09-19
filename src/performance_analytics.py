'''
Performance Analytics Module for Equity Analysis Toolkit

This class provides comprehensive performance analysis capabalities including return calculations, risk metrics, comparative analysis, and visualisation.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
from .data_collection import DataManager

class PerformanceAnalyzer:
    def __init__(self, api_key: str, cache_dir: str = '../data_cache'):
        self.dm = DataManager(api_key=api_key, cache_dir=cache_dir)
        self.api_key = api_key
        self.cache_dir = cache_dir
    
    def calculate_simple_returns(self, prices: pd.DataFrame):
        copy_df = prices.copy()
        copy_df['simple_returns'] = copy_df['Close'].pct_change()
        return copy_df
    
    def calculate_log_returns(self, prices):
        copy_df = prices.copy()
        copy_df['log_returns'] = np.log(1 + copy_df['Close'].pct_change())
        return copy_df
    
    def calculate_rolling_volatility(self, df, window=30, annualization_factor = 252):
        # window should be either in specific time frames allowed (i.e. 7, 14, 30, 60)
        # after research it seems it is standard to use log returns to calculate volatility
        # annualization factor defaults to 252 for trading days in a year given daily data set
        copy_df = df.copy()
        copy_df[f'vol_{window}'] = copy_df['log_returns'].rolling(window=window).std()
        copy_df[f'vol_{window}_annualized'] = copy_df[f'vol_{window}'] * np.sqrt(annualization_factor)
        return copy_df
    
    def calculate_sharpe_ratio(self, df, window=30, risk_free_rate=0.04025):
        # For now we will implement using a risk_free_rate var that we can keep constant, but might worth it
        # to eventually implement a new data pipeline to simply pull and download a data set from FRED for historical 3month Tbill rates
        # And additionally once that pipeline is established we can implement different types of risk-free rate implementations based
        # on time horizon aimed: short use FFR, common 3month Tbill, 10y Treasury bonds for longer term
        copy_df = df.copy()

        # Convert annual risk-free rate to period rate (so data point rate)
        period_risk_free_rate = risk_free_rate/252 # daily data

        # point here is that we are taking the avg log return over the window period and subtracting teh risk free rate
        # then we take the std_dev for the given the period, thus getting our period sharpe ratio
        # naming of columns subject to change  
        rolling_excess_return = copy_df['log_returns'].rolling(window=window).mean() - period_risk_free_rate
        rolling_volatility = copy_df['log_returns'].rolling(window=window).std()
        copy_df['sharpe_ratio'] = rolling_excess_return / rolling_volatility 
        
        return copy_df
    
    def calculate_performance_metrics(self, df, window=30, simple_returns=False):
        '''
        Complete transformation pipeline.
        '''
        # ideally we want the user to choose a window that is appropriate with the time frame they perform their analysis on. 
        # NOTE: should add options to give user option to customise the funcs called even more.
        transformed_df = df.copy()

        if simple_returns:
            transformed_df = self.calculate_simple_returns(transformed_df)
        
        transformed_df = self.calculate_log_returns(transformed_df)
        transformed_df = self.calculate_rolling_volatility(transformed_df, window=window)
        transformed_df = self.calculate_sharpe_ratio(transformed_df, window=window)

        return transformed_df
    
    def compare_performance(self, symbols: list, start_date, end_date, window=30, visualization=False):
        '''
        Compare multiple symbols across key performance metrics.
        I am assuming this will be in a class for now.
        
        Returns: a summary DataFrame with symbols as rows, metrics as columns
        '''
        # data collection
        ticker_data = {}
        for ticker in symbols:
            data = self.dm.get_stock_data(ticker, start_date=start_date, end_date=end_date)
            ticker_data[ticker] = self.calculate_performance_metrics(data, window=window)

        # summary metrics
        summary_metrics = []
        for ticker, data in ticker_data.items():
            clean_data = data.dropna() # subject to change, this forces larger periods

            if len(clean_data) == 0:
                raise Exception('Period was too small, please expand period')

            total_return = (clean_data['Close'].iloc[-1]/clean_data['Close'].loc[0] -1) * 100
            latest_vol = clean_data[f'vol_{window}_annualised'].iloc[-1] * 100
            latest_sharpe = clean_data['sharpe_ratio'].iloc[-1]
            max_drawdown = self.calculate_max_drawdown(clean_data['Close'])
            latest_price = clean_data['Close'].iloc[-1]

            summary_metrics.append({
                'Symbol': ticker,
                'Total_Return_%': round(total_return, 2),
                'Annualized_Vol_%': round(latest_vol, 2),
                'Current_Sharpe': round(latest_sharpe, 2),
                'Max_Drawdown_%': round(max_drawdown, 2),
                'Latest_Close': round(latest_price, 2)
            })

        comparison_df = pd.DataFrame(summary_metrics)
        comparison_df = comparison_df.set_index('Symbol')
        comparison_df = comparison_df.sort_values('Current_Sharpe', ascending=False)

        if visualization:
            self.create_comparison_charts(ticker_data, comparison_df)
        
        return comparison_df, ticker_data
