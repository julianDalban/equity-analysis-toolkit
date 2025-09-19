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
    
    def calculate_max_drawdown(self, prices):
        '''
        Calculating maximum consecutive peak-to-trough decline.
        '''

        if len(prices) == 0:
            return 0

        peak = prices.iloc[0]
        max_drawdown = 0

        for price in prices:
            if price > peak:
                peak = price

        current_drawdown = (price - peak) / peak

        if current_drawdown < max_drawdown:
            max_drawdown = current_drawdown

        return max_drawdown * 100
    
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
    
    # Visualisation below
    def create_comparison_charts(self, ticker_data, summary_df):
        """
        Create comprehensive visualization comparing multiple stocks
        
        Args:
            ticker_data: Dict with ticker -> DataFrame of performance metrics
            summary_df: Summary comparison DataFrame
        """
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Stock Performance Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Chart 1: Normalized Price Performance (Top Left)
        self.create_normalized_performance_chart(ticker_data, axes[0, 0])
        
        # Chart 2: Risk-Return Scatter Plot (Top Right)
        self.create_risk_return_scatter(summary_df, axes[0, 1])
        
        # Chart 3: Rolling Sharpe Ratios (Bottom Left)
        self.create_rolling_sharpe_chart(ticker_data, axes[1, 0])
        
        # Chart 4: Performance Metrics Comparison (Bottom Right)
        self.create_metrics_comparison_chart(summary_df, axes[1, 1])
        
        plt.tight_layout()
        plt.show()
    
    def create_normalized_performance_chart(self, ticker_data, ax):
        """
        Chart 1: Normalized price performance (all start at 100)
        Shows which stock had the best overall performance
        """
        for ticker, data in ticker_data.items():
            clean_data = data.dropna()
            if len(clean_data) == 0:
                continue
                
            # Normalize prices to start at 100
            normalized_prices = (clean_data['Close'] / clean_data['Close'].iloc[0]) * 100
            
            ax.plot(normalized_prices.index, normalized_prices, 
                    linewidth=2, label=ticker, alpha=0.8)
        
        ax.set_title('Normalized Price Performance\n(Starting Value = 100)', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at 100 (break-even)
        ax.axhline(y=100, color='black', linestyle='--', alpha=0.5)
    
    def create_risk_return_scatter(self, summary_df, ax):
        """
        Chart 2: Risk vs Return scatter plot
        Shows risk-adjusted performance positioning
        """
        x = summary_df['Annualized_Vol_%']
        y = summary_df['Total_Return_%']
        
        # Create scatter plot
        scatter = ax.scatter(x, y, s=100, alpha=0.7, c=summary_df['Current_Sharpe'], 
                            cmap='RdYlGn', edgecolors='black', linewidth=1)
        
        # Add labels for each point
        for i, ticker in enumerate(summary_df.index):
            ax.annotate(ticker, (x.iloc[i], y.iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Annualized Volatility (%)')
        ax.set_ylabel('Total Return (%)')
        ax.set_title('Risk vs Return Analysis\n(Color = Sharpe Ratio)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe Ratio', rotation=270, labelpad=15)
    
    def create_rolling_sharpe_chart(self, ticker_data, ax):
        """
        Chart 3: Rolling Sharpe ratios over time
        Shows how risk-adjusted performance evolved
        """
        for ticker, data in ticker_data.items():
            clean_data = data.dropna()
            if len(clean_data) == 0 or 'sharpe_ratio' not in clean_data.columns:
                continue
                
            ax.plot(clean_data.index, clean_data['sharpe_ratio'], 
                    linewidth=2, label=ticker, alpha=0.8)
        
        ax.set_title('Rolling Sharpe Ratios Over Time', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at 0 (break-even risk-adjusted return)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
    
    def create_metrics_comparison_chart(self, summary_df, ax):
        """
        Chart 4: Bar chart comparing key metrics
        Easy comparison of summary statistics
        """
        # Select key metrics for comparison
        metrics_to_plot = ['Total_Return_%', 'Current_Sharpe']
        
        # Create grouped bar chart
        x_pos = np.arange(len(summary_df.index))
        width = 0.35
        
        # Plot Total Return bars
        bars1 = ax.bar(x_pos - width/2, summary_df['Total_Return_%'], 
                    width, label='Total Return (%)', alpha=0.8, color='skyblue')
        
        # Create second y-axis for Sharpe ratio
        ax2 = ax.twinx()
        bars2 = ax2.bar(x_pos + width/2, summary_df['Current_Sharpe'], 
                        width, label='Sharpe Ratio', alpha=0.8, color='lightcoral')
        
        # Formatting
        ax.set_xlabel('Stocks')
        ax.set_ylabel('Total Return (%)', color='skyblue')
        ax2.set_ylabel('Sharpe Ratio', color='lightcoral')
        ax.set_title('Performance Metrics Comparison', fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticks_labels = summary_df.index
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Add legends
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    def plot_individual_stock_analysis(self, ticker, data, window=30):
        """
        Create detailed analysis for a single stock
        Useful for deep-dive analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{ticker} Detailed Performance Analysis', fontsize=16, fontweight='bold')
        
        clean_data = data.dropna()
        
        # Price and Volume
        ax1 = axes[0, 0]
        ax1_vol = ax1.twinx()
        
        ax1.plot(clean_data.index, clean_data['Close'], color='blue', linewidth=2, label='Close Price')
        ax1_vol.bar(clean_data.index, clean_data['Volume'], alpha=0.3, color='gray', label='Volume')
        
        ax1.set_title('Price and Volume')
        ax1.set_ylabel('Price ($)', color='blue')
        ax1_vol.set_ylabel('Volume', color='gray')
        ax1.legend(loc='upper left')
        ax1_vol.legend(loc='upper right')
        
        # Returns Distribution
        axes[0, 1].hist(clean_data['log_returns'] * 100, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Log Returns Distribution')
        axes[0, 1].set_xlabel('Daily Log Returns (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # Rolling Volatility
        vol_col = f'vol_{window}_annualized'
        if vol_col in clean_data.columns:
            axes[1, 0].plot(clean_data.index, clean_data[vol_col] * 100, 
                        color='red', linewidth=2, label=f'{window}-day Volatility')
            axes[1, 0].set_title('Rolling Annualized Volatility')
            axes[1, 0].set_ylabel('Volatility (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Sharpe Ratio Over Time
        if 'sharpe_ratio' in clean_data.columns:
            axes[1, 1].plot(clean_data.index, clean_data['sharpe_ratio'], 
                        color='green', linewidth=2, label='Sharpe Ratio')
            axes[1, 1].set_title('Rolling Sharpe Ratio')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()