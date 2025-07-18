# src/risk_analyzer.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedRiskAnalyzer:
    """
    Institutional-grade risk analysis system for financial portfolio management.
    Implements industry-standard risk metrics used by major financial institutions.
    """
    
    def __init__(self, risk_free_rate: float = 0.02, trading_days: int = 252):
        """
        Initialize the risk analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
            trading_days: Number of trading days per year (default: 252)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.confidence_levels = [0.01, 0.05, 0.10]  # 99%, 95%, 90% confidence
        
    def calculate_comprehensive_metrics(self, stock_data: Dict[str, pd.DataFrame], 
                                      market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate comprehensive risk and performance metrics for all stocks.
        
        Args:
            stock_data: Dictionary mapping ticker symbols to price data
            market_data: Market benchmark data (optional)
            
        Returns:
            DataFrame with comprehensive risk metrics for each stock
        """
        logger.info(f"Calculating risk metrics for {len(stock_data)} securities")
        
        metrics_list = []
        
        for ticker, data in stock_data.items():
            try:
                metrics = self._calculate_single_stock_metrics(ticker, data, market_data)
                if metrics:
                    metrics_list.append(metrics)
            except Exception as e:
                logger.error(f"Error calculating metrics for {ticker}: {str(e)}")
                continue
        
        if not metrics_list:
            logger.warning("No metrics calculated for any stocks")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(metrics_list)
        results_df.set_index('Ticker', inplace=True)
        
        # Add portfolio-level rankings
        results_df = self._add_rankings(results_df)
        
        logger.info(f"Successfully calculated metrics for {len(results_df)} securities")
        return results_df
    
    def _calculate_single_stock_metrics(self, ticker: str, data: pd.DataFrame, 
                                      market_data: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        """
        Calculate comprehensive metrics for a single stock.
        
        Args:
            ticker: Stock symbol
            data: Historical price data
            market_data: Market benchmark data
            
        Returns:
            Dictionary of calculated metrics
        """
        if data.empty or 'Daily_Return' not in data.columns:
            return None
        
        returns = data['Daily_Return'].dropna()
        prices = data['Close'].dropna()
        
        if len(returns) < 20:  # Minimum data requirement
            return None
        
        # Basic return metrics
        annual_return = returns.mean() * self.trading_days
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(self.trading_days)
        downside_deviation = self._calculate_downside_deviation(returns)
        
        # Risk-adjusted returns
        sharpe_ratio = (annual_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Value at Risk calculations
        var_metrics = self._calculate_var_metrics(returns)
        
        # Drawdown analysis
        drawdown_metrics = self._calculate_drawdown_metrics(prices)
        
        # Technical analysis metrics
        technical_metrics = self._calculate_technical_metrics(data)
        
        # Market comparison metrics
        market_metrics = self._calculate_market_metrics(returns, market_data) if market_data is not None else {}
        
        # Distribution analysis
        distribution_metrics = self._calculate_distribution_metrics(returns)
        
        # Combine all metrics
        metrics = {
            'Ticker': ticker,
            'Current_Price': prices.iloc[-1],
            'Total_Return': total_return,
            'Annual_Return': annual_return,
            'Volatility': volatility,
            'Downside_Deviation': downside_deviation,
            'Sharpe_Ratio': sharpe_ratio,
            'Sortino_Ratio': sortino_ratio,
            **var_metrics,
            **drawdown_metrics,
            **technical_metrics,
            **market_metrics,
            **distribution_metrics
        }
        
        return metrics
    
    def _calculate_downside_deviation(self, returns: pd.Series, target_return: float = 0) -> float:
        """Calculate downside deviation (semi-deviation)."""
        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            return 0
        return downside_returns.std() * np.sqrt(self.trading_days)
    
    def _calculate_var_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate Value at Risk and Expected Shortfall metrics."""
        var_metrics = {}
        
        for confidence in self.confidence_levels:
            confidence_pct = int((1 - confidence) * 100)
            
            # Parametric VaR (assuming normal distribution)
            var_parametric = stats.norm.ppf(confidence, returns.mean(), returns.std())
            
            # Historical VaR
            var_historical = np.percentile(returns, confidence * 100)
            
            # Expected Shortfall (Conditional VaR)
            tail_returns = returns[returns <= var_historical]
            expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var_historical
            
            var_metrics.update({
                f'VaR_{confidence_pct}%_Parametric': var_parametric,
                f'VaR_{confidence_pct}%_Historical': var_historical,
                f'ES_{confidence_pct}%': expected_shortfall
            })
        
        return var_metrics
    
    def _calculate_drawdown_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive drawdown analysis."""
        # Calculate running maximum
        rolling_max = prices.expanding().max()
        
        # Calculate drawdowns
        drawdowns = (prices - rolling_max) / rolling_max
        
        # Maximum drawdown
        max_drawdown = drawdowns.min()
        
        # Average drawdown
        negative_drawdowns = drawdowns[drawdowns < 0]
        avg_drawdown = negative_drawdowns.mean() if len(negative_drawdowns) > 0 else 0
        
        # Drawdown duration analysis
        in_drawdown = drawdowns < -0.01  # More than 1% drawdown
        drawdown_periods = self._find_drawdown_periods(in_drawdown)
        
        max_drawdown_duration = max([period['duration'] for period in drawdown_periods]) if drawdown_periods else 0
        avg_drawdown_duration = np.mean([period['duration'] for period in drawdown_periods]) if drawdown_periods else 0
        
        # Current drawdown
        current_drawdown = drawdowns.iloc[-1]
        
        return {
            'Max_Drawdown': max_drawdown,
            'Avg_Drawdown': avg_drawdown,
            'Current_Drawdown': current_drawdown,
            'Max_Drawdown_Duration': max_drawdown_duration,
            'Avg_Drawdown_Duration': avg_drawdown_duration,
            'Drawdown_Periods': len(drawdown_periods)
        }
    
    def _find_drawdown_periods(self, in_drawdown: pd.Series) -> List[Dict]:
        """Identify discrete drawdown periods."""
        periods = []
        start_idx = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                periods.append({
                    'start': start_idx,
                    'end': i - 1,
                    'duration': i - start_idx
                })
                start_idx = None
        
        # Handle case where drawdown continues to end
        if start_idx is not None:
            periods.append({
                'start': start_idx,
                'end': len(in_drawdown) - 1,
                'duration': len(in_drawdown) - start_idx
            })
        
        return periods
    
    def _calculate_technical_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical analysis indicators."""
        try:
            current_price = data['Close'].iloc[-1]
            
            # Moving average positions
            sma_20 = data['SMA_20'].iloc[-1] if 'SMA_20' in data.columns else current_price
            sma_50 = data['SMA_50'].iloc[-1] if 'SMA_50' in data.columns else current_price
            sma_200 = data['SMA_200'].iloc[-1] if 'SMA_200' in data.columns else current_price
            
            # RSI
            rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50
            
            # MACD
            macd = data['MACD'].iloc[-1] if 'MACD' in data.columns else 0
            macd_signal = data['MACD_Signal'].iloc[-1] if 'MACD_Signal' in data.columns else 0
            
            # Volume analysis
            volume_ratio = data['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in data.columns else 1
            
            # Price momentum
            momentum_1m = data['Price_Change_1M'].iloc[-1] if 'Price_Change_1M' in data.columns else 0
            momentum_3m = data['Price_Change_3M'].iloc[-1] if 'Price_Change_3M' in data.columns else 0
            
            return {
                'Price_vs_SMA20': (current_price - sma_20) / sma_20,
                'Price_vs_SMA50': (current_price - sma_50) / sma_50,
                'Price_vs_SMA200': (current_price - sma_200) / sma_200,
                'RSI': rsi,
                'MACD': macd,
                'MACD_Signal': macd_signal,
                'MACD_Histogram': macd - macd_signal,
                'Volume_Ratio': volume_ratio,
                'Momentum_1M': momentum_1m,
                'Momentum_3M': momentum_3m
            }
            
        except Exception as e:
            logger.warning(f"Error calculating technical metrics: {str(e)}")
            return {
                'Price_vs_SMA20': 0, 'Price_vs_SMA50': 0, 'Price_vs_SMA200': 0,
                'RSI': 50, 'MACD': 0, 'MACD_Signal': 0, 'MACD_Histogram': 0,
                'Volume_Ratio': 1, 'Momentum_1M': 0, 'Momentum_3M': 0
            }
    
    def _calculate_market_metrics(self, returns: pd.Series, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate market-relative metrics including beta and correlation."""
        try:
            if market_data.empty or 'Daily_Return' not in market_data.columns:
                return {'Beta': 1.0, 'Alpha': 0.0, 'Correlation': 0.0, 'Information_Ratio': 0.0}
            
            market_returns = market_data['Daily_Return'].dropna()
            
            # Align dates
            common_dates = returns.index.intersection(market_returns.index)
            if len(common_dates) < 20:
                return {'Beta': 1.0, 'Alpha': 0.0, 'Correlation': 0.0, 'Information_Ratio': 0.0}
            
            aligned_returns = returns[common_dates]
            aligned_market = market_returns[common_dates]
            
            # Calculate beta and alpha
            covariance = np.cov(aligned_returns, aligned_market)[0, 1]
            market_variance = np.var(aligned_market)
            beta = covariance / market_variance if market_variance > 0 else 1.0
            
            alpha = (aligned_returns.mean() - self.risk_free_rate / self.trading_days) - \
                   beta * (aligned_market.mean() - self.risk_free_rate / self.trading_days)
            alpha_annualized = alpha * self.trading_days
            
            # Correlation
            correlation = np.corrcoef(aligned_returns, aligned_market)[0, 1]
            
            # Information ratio (active return / tracking error)
            excess_returns = aligned_returns - aligned_market
            tracking_error = excess_returns.std() * np.sqrt(self.trading_days)
            information_ratio = (excess_returns.mean() * self.trading_days) / tracking_error if tracking_error > 0 else 0
            
            return {
                'Beta': beta,
                'Alpha': alpha_annualized,
                'Correlation': correlation,
                'Information_Ratio': information_ratio,
                'Tracking_Error': tracking_error
            }
            
        except Exception as e:
            logger.warning(f"Error calculating market metrics: {str(e)}")
            return {'Beta': 1.0, 'Alpha': 0.0, 'Correlation': 0.0, 'Information_Ratio': 0.0}
    
    def _calculate_distribution_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate return distribution characteristics."""
        try:
            # Basic distribution metrics
            skewness = stats.skew(returns.dropna())
            kurtosis = stats.kurtosis(returns.dropna())
            
            # Jarque-Bera test for normality
            jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna())
            
            # Tail ratios
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            tail_ratio = len(positive_returns) / len(negative_returns) if len(negative_returns) > 0 else np.inf
            
            # Win rate
            win_rate = len(positive_returns) / len(returns)
            
            # Average win/loss
            avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
            avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
            win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
            
            return {
                'Skewness': skewness,
                'Kurtosis': kurtosis,
                'JB_Statistic': jb_stat,
                'JB_PValue': jb_pvalue,
                'Tail_Ratio': tail_ratio,
                'Win_Rate': win_rate,
                'Avg_Win': avg_win,
                'Avg_Loss': avg_loss,
                'Win_Loss_Ratio': win_loss_ratio
            }
            
        except Exception as e:
            logger.warning(f"Error calculating distribution metrics: {str(e)}")
            return {
                'Skewness': 0, 'Kurtosis': 0, 'JB_Statistic': 0, 'JB_PValue': 1,
                'Tail_Ratio': 1, 'Win_Rate': 0.5, 'Avg_Win': 0, 'Avg_Loss': 0, 'Win_Loss_Ratio': 1
            }
    
    def _add_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add percentile rankings for key metrics."""
        ranking_metrics = [
            'Sharpe_Ratio', 'Sortino_Ratio', 'Annual_Return', 'Total_Return',
            'Information_Ratio', 'Win_Rate', 'Alpha'
        ]
        
        for metric in ranking_metrics:
            if metric in df.columns:
                df[f'{metric}_Rank'] = df[metric].rank(pct=True) * 100
        
        # Overall score combining multiple factors
        df['Overall_Score'] = (
            df.get('Sharpe_Ratio_Rank', 50) * 0.3 +
            df.get('Annual_Return_Rank', 50) * 0.2 +
            df.get('Information_Ratio_Rank', 50) * 0.2 +
            df.get('Win_Rate_Rank', 50) * 0.15 +
            df.get('Alpha_Rank', 50) * 0.15
        )
        
        return df
    
    def calculate_portfolio_metrics(self, stock_data: Dict[str, pd.DataFrame], 
                                  weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate portfolio-level risk metrics.
        
        Args:
            stock_data: Dictionary of stock price data
            weights: Portfolio weights (equal weight if None)
            
        Returns:
            Dictionary of portfolio metrics
        """
        if not stock_data:
            return {}
        
        # Equal weights if not provided
        if weights is None:
            weights = {ticker: 1/len(stock_data) for ticker in stock_data.keys()}
        
        # Align all returns
        returns_data = {}
        for ticker, data in stock_data.items():
            if 'Daily_Return' in data.columns:
                returns_data[ticker] = data['Daily_Return'].dropna()
        
        if not returns_data:
            return {}
        
        # Create returns matrix
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if returns_df.empty:
            return {}
        
        # Calculate portfolio returns
        weight_vector = np.array([weights.get(ticker, 0) for ticker in returns_df.columns])
        portfolio_returns = returns_df.dot(weight_vector)
        
        # Portfolio metrics
        portfolio_annual_return = portfolio_returns.mean() * self.trading_days
        portfolio_volatility = portfolio_returns.std() * np.sqrt(self.trading_days)
        portfolio_sharpe = (portfolio_annual_return - self.risk_free_rate) / portfolio_volatility
        
        # Diversification metrics
        correlation_matrix = returns_df.corr()
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        
        # Portfolio VaR
        portfolio_var_5 = np.percentile(portfolio_returns, 5)
        
        return {
            'Portfolio_Annual_Return': portfolio_annual_return,
            'Portfolio_Volatility': portfolio_volatility,
            'Portfolio_Sharpe_Ratio': portfolio_sharpe,
            'Portfolio_VaR_5%': portfolio_var_5,
            'Average_Correlation': avg_correlation,
            'Diversification_Ratio': 1 - avg_correlation
        }
    
    def stress_test_analysis(self, stock_data: Dict[str, pd.DataFrame], 
                           scenarios: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Perform stress testing analysis on portfolio.
        
        Args:
            stock_data: Dictionary of stock data
            scenarios: Custom stress scenarios
            
        Returns:
            DataFrame with stress test results
        """
        if scenarios is None:
            scenarios = {
                'Market Crash (-30%)': -0.30,
                'Severe Correction (-20%)': -0.20,
                'Moderate Correction (-10%)': -0.10,
                'Market Rally (+20%)': 0.20,
                'Bull Market (+30%)': 0.30
            }
        
        stress_results = []
        
        for ticker, data in stock_data.items():
            current_price = data['Close'].iloc[-1]
            
            for scenario_name, shock in scenarios.items():
                shocked_price = current_price * (1 + shock)
                price_impact = (shocked_price - current_price) / current_price
                
                stress_results.append({
                    'Ticker': ticker,
                    'Scenario': scenario_name,
                    'Current_Price': current_price,
                    'Stressed_Price': shocked_price,
                    'Price_Impact': price_impact
                })
        
        return pd.DataFrame(stress_results)