# src/data_ingestion.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import streamlit as st
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialDataIngestion:
    """
    Professional grade financial data ingestion system for real-time market analysis.
    Designed for institutional-quality financial applications.
    """
    
    def __init__(self, cache_ttl: int = 300):
        """
        Initialize the data ingestion system.
        Args:
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
        """
        self.cache_ttl = cache_ttl
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.trading_days = 252
        
    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_stock_data(_self, tickers: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical stock data with comprehensive error handling and data validation.
        
        Args:
            tickers: List of stock symbols to fetch
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            
        Returns:
            Dictionary mapping ticker symbols to historical data DataFrames
        """
        logger.info(f"Fetching data for {len(tickers)} tickers over {period} period")
        
        stock_data = {}
        failed_tickers = []
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(tickers):
            try:
                status_text.text(f"Fetching data for {ticker}...")
                
                # Fetch stock data
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period, auto_adjust=True, prepost=True)
                
                if hist.empty:
                    logger.warning(f"No data available for {ticker}")
                    failed_tickers.append(ticker)
                    continue
                
                # Clean and validate data
                hist = _self._clean_data(hist, ticker)
                
                if hist is not None and len(hist) > 20:  # Minimum 20 days of data
                    # Calculate derived metrics
                    hist = _self._calculate_derived_metrics(hist)
                    stock_data[ticker] = hist
                    logger.info(f"Successfully fetched {len(hist)} days of data for {ticker}")
                else:
                    failed_tickers.append(ticker)
                    logger.warning(f"Insufficient data for {ticker}")
                    
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {str(e)}")
                failed_tickers.append(ticker)
            
            # Update progress
            progress_bar.progress((i + 1) / len(tickers))
        
        progress_bar.empty()
        status_text.empty()
        
        if failed_tickers:
            st.warning(f"Failed to fetch data for: {', '.join(failed_tickers)}")
        
        logger.info(f"Successfully fetched data for {len(stock_data)} out of {len(tickers)} tickers")
        return stock_data
    
    def _clean_data(self, data: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """
        Clean and validate financial data with institutional-grade quality checks.
        
        Args:
            data: Raw price data from yfinance
            ticker: Stock symbol for logging
            
        Returns:
            Cleaned DataFrame or None if data quality is insufficient
        """
        try:
            # Remove rows with missing critical data
            data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
            
            # Validate price relationships
            invalid_prices = (
                (data['High'] < data['Low']) | 
                (data['High'] < data['Close']) | 
                (data['Low'] > data['Close']) |
                (data['Close'] <= 0) |
                (data['Volume'] < 0)
            )
            
            if invalid_prices.any():
                logger.warning(f"Found {invalid_prices.sum()} invalid price records for {ticker}")
                data = data[~invalid_prices]
            
            # Remove extreme outliers (price changes > 50% in a day, likely stock splits not adjusted)
            data['price_change'] = data['Close'].pct_change().abs()
            extreme_changes = data['price_change'] > 0.5
            
            if extreme_changes.any():
                logger.warning(f"Removing {extreme_changes.sum()} extreme price changes for {ticker}")
                data = data[~extreme_changes]
            
            # Ensure minimum data requirements
            if len(data) < 20:
                logger.warning(f"Insufficient data points for {ticker}: {len(data)} days")
                return None
            
            # Sort by date and reset index
            data = data.sort_index().drop('price_change', axis=1)
            
            return data
            
        except Exception as e:
            logger.error(f"Error cleaning data for {ticker}: {str(e)}")
            return None
    
    def _calculate_derived_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive derived metrics for financial analysis.
        
        Args:
            data: Clean historical price data
            
        Returns:
            DataFrame with additional calculated metrics
        """
        # Basic returns
        data['Daily_Return'] = data['Close'].pct_change()
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Moving averages
        data['SMA_10'] = data['Close'].rolling(window=10, min_periods=1).mean()
        data['SMA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
        data['SMA_50'] = data['Close'].rolling(window=50, min_periods=1).mean()
        data['SMA_200'] = data['Close'].rolling(window=200, min_periods=1).mean()
        
        # Exponential moving averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # Volatility measures
        data['Rolling_Vol_20'] = data['Daily_Return'].rolling(window=20, min_periods=1).std() * np.sqrt(self.trading_days)
        data['Rolling_Vol_60'] = data['Daily_Return'].rolling(window=60, min_periods=1).std() * np.sqrt(self.trading_days)
        
        # Technical indicators
        data['RSI'] = self._calculate_rsi(data['Close'])
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        
        # Price position indicators
        data['Price_vs_SMA20'] = (data['Close'] - data['SMA_20']) / data['SMA_20']
        data['Price_vs_SMA50'] = (data['Close'] - data['SMA_50']) / data['SMA_50']
        data['Price_vs_SMA200'] = (data['Close'] - data['SMA_200']) / data['SMA_200']
        
        # Momentum indicators
        data['Price_Change_1W'] = data['Close'].pct_change(5)  # 1 week
        data['Price_Change_1M'] = data['Close'].pct_change(22)  # 1 month
        data['Price_Change_3M'] = data['Close'].pct_change(66)  # 3 months
        data['Price_Change_6M'] = data['Close'].pct_change(132)  # 6 months
        
        # Volume indicators
        data['Volume_SMA_20'] = data['Volume'].rolling(window=20, min_periods=1).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20']
        
        # Price range indicators
        data['Price_Range'] = (data['High'] - data['Low']) / data['Close']
        data['True_Range'] = np.maximum(
            data['High'] - data['Low'],
            np.maximum(
                abs(data['High'] - data['Close'].shift(1)),
                abs(data['Low'] - data['Close'].shift(1))
            )
        )
        data['ATR'] = data['True_Range'].rolling(window=14, min_periods=1).mean()
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Price series
            window: RSI calculation window
            
        Returns:
            RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
    def fetch_market_indices(_self) -> Dict[str, pd.DataFrame]:
        """
        Fetch major market indices for benchmark comparison.
        
        Returns:
            Dictionary of market index data
        """
        indices = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC', 
            'Dow Jones': '^DJI',
            'Russell 2000': '^RUT',
            'VIX': '^VIX'
        }
        
        index_data = {}
        for name, symbol in indices.items():
            try:
                data = yf.Ticker(symbol).history(period="1y")
                if not data.empty:
                    data['Daily_Return'] = data['Close'].pct_change()
                    index_data[name] = data
            except Exception as e:
                logger.warning(f"Could not fetch {name} data: {str(e)}")
        
        return index_data
    
    def get_stock_info(self, ticker: str) -> Dict:
        """
        Get detailed stock information and company fundamentals.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key information
            stock_info = {
                'company_name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'dividend_yield': info.get('dividendYield', None),
                'beta': info.get('beta', None),
                'description': info.get('longBusinessSummary', 'N/A')[:200] + '...' if info.get('longBusinessSummary') else 'N/A'
            }
            
            return stock_info
            
        except Exception as e:
            logger.error(f"Error fetching info for {ticker}: {str(e)}")
            return {'company_name': ticker, 'sector': 'N/A', 'industry': 'N/A'}
    
    def save_data(self, data: Dict[str, pd.DataFrame], filename: str = None) -> str:
        """
        Save processed data to CSV files.
        
        Args:
            data: Dictionary of stock data
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_data_{timestamp}.csv"
        
        filepath = f"data/processed/{filename}"
        
        # Combine all stock data
        combined_data = []
        for ticker, df in data.items():
            df_copy = df.copy()
            df_copy['Ticker'] = ticker
            combined_data.append(df_copy)
        
        if combined_data:
            final_df = pd.concat(combined_data, ignore_index=True)
            final_df.to_csv(filepath, index=False)
            logger.info(f"Data saved to {filepath}")
        
        return filepath