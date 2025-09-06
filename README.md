# Equity Analysis Toolkit

The goal of this project is to gain some preliminary exposure to using OHLCV data to assess and understanding the movements of a specific stock's price given market conditions, its industry, relative position, and general macro environment. In this project I'm aiming to analyse the following areas of the market:
- stocks related to the digital asset ecosystem and their movements with relation to  traditional tech stocks
- macro-sensitive sectors such as financial services, tech, and interest rate sensitive areas to study and understand how monetary policy and fiscal policy can affect stock prices

Given this context, I hope to learn how correlation structures change during market stress and why certain sectors rotate based on economic cycles, how technical indicators can inform trading decisions, and how macro events impact sector performance.

## Module 1: Data Collection Infrastructure
Built a robust data collection system with intelligent caching and API management for market data analysis.

## Key Features
- **Smart Caching**: Context-aware refresh logic that distinguishes short-term and historical requests
- **API Management**: Alpha Vantage integration with rate limiting and automatic fallback to cache
- **Data Quality**: Automatic type conversion, date normalization, and business day handling
- **Error Handling**: Comprehensive validation and graceful degradation when APIs fail

### Architecture
- **DataManager**: Main class orchestrating data operations with CSV-based caching
- **Date Handling**: Robust pandas-based date processing supporting multiple input formats
- **Performance Optimization**: Uses compact/full API modes based on request size

### Usage Example
```python
from src.data_collection import DataManager

dm = DataManager("your_api_key")
coin_data = dm.get_stock_data("COIN", start_date="2024-10-01", end_date="TODAY")
```

