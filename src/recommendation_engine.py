# src/recommendation_engine.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RecommendationType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class InvestmentRecommendation:
    ticker: str
    recommendation: RecommendationType
    confidence: float
    target_price: Optional[float]
    current_price: float
    reasons: List[str]
    risk_factors: List[str]
    score: float
    time_horizon: str

class AdvancedRecommendationEngine:
    """
    Sophisticated investment recommendation system that combines multiple analytical frameworks.
    Designed for institutional-grade investment decision support.
    """
    
    def __init__(self, investment_style: str = "Balanced", risk_tolerance: str = "Moderate"):
        """
        Initialize the recommendation engine.
        
        Args:
            investment_style: Investment approach (Value, Growth, Momentum, Quality, Balanced)
            risk_tolerance: Risk level (Conservative, Moderate, Aggressive)
        """
        self.investment_style = investment_style
        self.risk_tolerance = risk_tolerance
        
        # Scoring weights based on investment style
        self.style_weights = self._get_style_weights()
        
        # Risk tolerance thresholds
        self.risk_thresholds = self._get_risk_thresholds()
        
        # Recommendation thresholds
        self.rec_thresholds = {
            'strong_buy': 80,
            'buy': 65,
            'hold_upper': 55,
            'hold_lower': 45,
            'sell': 30,
            'strong_sell': 15
        }
    
    def _get_style_weights(self) -> Dict[str, float]:
        """Get scoring weights based on investment style."""
        style_configs = {
            "Value": {
                "valuation": 0.35,
                "quality": 0.25,
                "momentum": 0.10,
                "technical": 0.15,
                "risk": 0.15
            },
            "Growth": {
                "valuation": 0.15,
                "quality": 0.20,
                "momentum": 0.35,
                "technical": 0.20,
                "risk": 0.10
            },
            "Momentum": {
                "valuation": 0.10,
                "quality": 0.15,
                "momentum": 0.45,
                "technical": 0.25,
                "risk": 0.05
            },
            "Quality": {
                "valuation": 0.25,
                "quality": 0.40,
                "momentum": 0.15,
                "technical": 0.10,
                "risk": 0.10
            },
            "Balanced": {
                "valuation": 0.25,
                "quality": 0.25,
                "momentum": 0.20,
                "technical": 0.15,
                "risk": 0.15
            }
        }
        return style_configs.get(self.investment_style, style_configs["Balanced"])
    
    def _get_risk_thresholds(self) -> Dict[str, float]:
        """Get risk thresholds based on risk tolerance."""
        risk_configs = {
            "Conservative": {
                "max_volatility": 0.20,
                "min_sharpe": 0.75,
                "max_drawdown": -0.10,
                "min_win_rate": 0.60
            },
            "Moderate": {
                "max_volatility": 0.35,
                "min_sharpe": 0.50,
                "max_drawdown": -0.20,
                "min_win_rate": 0.55
            },
            "Aggressive": {
                "max_volatility": 0.60,
                "min_sharpe": 0.30,
                "max_drawdown": -0.35,
                "min_win_rate": 0.50
            }
        }
        return risk_configs.get(self.risk_tolerance, risk_configs["Moderate"])
    
    def generate_recommendations(self, risk_metrics: pd.DataFrame, 
                               ml_predictions: Optional[Dict[str, Dict]] = None,
                               market_conditions: Optional[Dict] = None) -> List[InvestmentRecommendation]:
        """
        Generate comprehensive investment recommendations for all stocks.
        
        Args:
            risk_metrics: DataFrame with risk and performance metrics
            ml_predictions: ML model predictions (optional)
            market_conditions: Current market conditions (optional)
            
        Returns:
            List of investment recommendations
        """
        logger.info(f"Generating recommendations for {len(risk_metrics)} securities")
        
        recommendations = []
        
        for ticker in risk_metrics.index:
            try:
                stock_metrics = risk_metrics.loc[ticker]
                stock_predictions = ml_predictions.get(ticker, {}) if ml_predictions else {}
                
                recommendation = self._analyze_single_stock(
                    ticker, stock_metrics, stock_predictions, market_conditions
                )
                
                if recommendation:
                    recommendations.append(recommendation)
                    
            except Exception as e:
                logger.error(f"Error generating recommendation for {ticker}: {str(e)}")
                continue
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def _analyze_single_stock(self, ticker: str, metrics: pd.Series, 
                            predictions: Dict, market_conditions: Optional[Dict]) -> Optional[InvestmentRecommendation]:
        """
        Analyze a single stock and generate recommendation.
        
        Args:
            ticker: Stock symbol
            metrics: Risk and performance metrics
            predictions: ML predictions
            market_conditions: Market conditions
            
        Returns:
            Investment recommendation
        """
        # Calculate component scores
        valuation_score = self._calculate_valuation_score(metrics)
        quality_score = self._calculate_quality_score(metrics)
        momentum_score = self._calculate_momentum_score(metrics)
        technical_score = self._calculate_technical_score(metrics)
        risk_score = self._calculate_risk_score(metrics)
        
        # Weight scores by investment style
        weighted_score = (
            valuation_score * self.style_weights["valuation"] +
            quality_score * self.style_weights["quality"] +
            momentum_score * self.style_weights["momentum"] +
            technical_score * self.style_weights["technical"] +
            risk_score * self.style_weights["risk"]
        ) * 100
        
        # Adjust for ML predictions
        if predictions:
            ml_adjustment = self._calculate_ml_adjustment(predictions)
            weighted_score += ml_adjustment
        
        # Adjust for market conditions
        if market_conditions:
            market_adjustment = self._calculate_market_adjustment(metrics, market_conditions)
            weighted_score += market_adjustment
        
        # Ensure score is within bounds
        weighted_score = np.clip(weighted_score, 0, 100)
        
        # Generate recommendation
        recommendation_type = self._score_to_recommendation(weighted_score)
        confidence = self._calculate_confidence(metrics, weighted_score)
        
        # Generate reasons and risk factors
        reasons = self._generate_reasons(metrics, predictions, weighted_score)
        risk_factors = self._identify_risk_factors(metrics)
        
        # Calculate target price (simplified)
        target_price = self._calculate_target_price(metrics, recommendation_type)
        
        return InvestmentRecommendation(
            ticker=ticker,
            recommendation=recommendation_type,
            confidence=confidence,
            target_price=target_price,
            current_price=metrics.get('Current_Price', 0),
            reasons=reasons,
            risk_factors=risk_factors,
            score=weighted_score,
            time_horizon="3-6 months"
        )
    
    def _calculate_valuation_score(self, metrics: pd.Series) -> float:
        """Calculate valuation attractiveness score."""
        score = 0.5  # Base score
        
        # P/E ratio analysis (if available)
        if 'PE_Ratio' in metrics and pd.notna(metrics['PE_Ratio']):
            pe_ratio = metrics['PE_Ratio']
            if pe_ratio < 15:
                score += 0.3
            elif pe_ratio < 25:
                score += 0.1
            elif pe_ratio > 40:
                score -= 0.2
        
        # Price vs moving averages
        if 'Price_vs_SMA50' in metrics:
            price_vs_sma50 = metrics['Price_vs_SMA50']
            if price_vs_sma50 < -0.05:  # 5% below SMA50
                score += 0.2  # Potential value opportunity
            elif price_vs_sma50 > 0.20:  # 20% above SMA50
                score -= 0.2  # Potentially overvalued
        
        # Dividend yield (if available)
        if 'Dividend_Yield' in metrics and pd.notna(metrics['Dividend_Yield']):
            div_yield = metrics['Dividend_Yield']
            if div_yield > 0.03:  # >3% yield
                score += 0.1
        
        return np.clip(score, 0, 1)
    
    def _calculate_quality_score(self, metrics: pd.Series) -> float:
        """Calculate business quality score."""
        score = 0.5  # Base score
        
        # Sharpe ratio - risk-adjusted returns
        sharpe = metrics.get('Sharpe_Ratio', 0)
        if sharpe > 1.5:
            score += 0.3
        elif sharpe > 1.0:
            score += 0.2
        elif sharpe > 0.5:
            score += 0.1
        elif sharpe < 0:
            score -= 0.3
        
        # Sortino ratio
        sortino = metrics.get('Sortino_Ratio', 0)
        if sortino > sharpe * 1.2:  # Better downside protection
            score += 0.1
        
        # Win rate
        win_rate = metrics.get('Win_Rate', 0.5)
        if win_rate > 0.60:
            score += 0.2
        elif win_rate < 0.45:
            score -= 0.2
        
        # Information ratio (vs market)
        info_ratio = metrics.get('Information_Ratio', 0)
        if info_ratio > 0.5:
            score += 0.1
        elif info_ratio < -0.5:
            score -= 0.1
        
        return np.clip(score, 0, 1)
    
    def _calculate_momentum_score(self, metrics: pd.Series) -> float:
        """Calculate price and earnings momentum score."""
        score = 0.5  # Base score
        
        # Recent price momentum
        momentum_1m = metrics.get('Momentum_1M', 0)
        momentum_3m = metrics.get('Momentum_3M', 0)
        
        if momentum_1m > 0.05:  # >5% in 1 month
            score += 0.2
        elif momentum_1m < -0.05:
            score -= 0.2
        
        if momentum_3m > 0.15:  # >15% in 3 months
            score += 0.2
        elif momentum_3m < -0.15:
            score -= 0.2
        
        # Annual return
        annual_return = metrics.get('Annual_Return', 0)
        if annual_return > 0.20:  # >20% annual return
            score += 0.2
        elif annual_return < -0.10:  # <-10% annual return
            score -= 0.3
        
        # Volatility-adjusted momentum
        volatility = metrics.get('Volatility', 0.20)
        if annual_return > 0 and volatility > 0:
            momentum_ratio = annual_return / volatility
            if momentum_ratio > 1.0:
                score += 0.1
        
        return np.clip(score, 0, 1)
    
    def _calculate_technical_score(self, metrics: pd.Series) -> float:
        """Calculate technical analysis score."""
        score = 0.5  # Base score
        
        # RSI analysis
        rsi = metrics.get('RSI', 50)
        if 30 < rsi < 70:  # Neutral zone
            score += 0.1
        elif rsi < 30:  # Oversold
            score += 0.2
        elif rsi > 70:  # Overbought
            score -= 0.2
        
        # MACD analysis
        macd_histogram = metrics.get('MACD_Histogram', 0)
        if macd_histogram > 0:
            score += 0.1
        elif macd_histogram < 0:
            score -= 0.1
        
        # Price position vs SMAs
        price_vs_sma20 = metrics.get('Price_vs_SMA20', 0)
        price_vs_sma50 = metrics.get('Price_vs_SMA50', 0)
        
        # Bullish: price above both SMAs
        if price_vs_sma20 > 0 and price_vs_sma50 > 0:
            score += 0.2
        # Bearish: price below both SMAs
        elif price_vs_sma20 < 0 and price_vs_sma50 < 0:
            score -= 0.2
        
        # Volume confirmation
        volume_ratio = metrics.get('Volume_Ratio', 1)
        if volume_ratio > 1.5:  # High volume
            if price_vs_sma20 > 0:  # High volume with uptrend
                score += 0.1
            else:  # High volume with downtrend
                score -= 0.1
        
        return np.clip(score, 0, 1)
    
    def _calculate_risk_score(self, metrics: pd.Series) -> float:
        """Calculate risk assessment score (higher is better/safer)."""
        score = 0.5  # Base score
        
        # Volatility assessment
        volatility = metrics.get('Volatility', 0.25)
        risk_threshold = self.risk_thresholds['max_volatility']
        
        if volatility < risk_threshold * 0.7:
            score += 0.3  # Low volatility is good
        elif volatility > risk_threshold:
            score -= 0.3  # High volatility is concerning
        
        # Maximum drawdown
        max_drawdown = metrics.get('Max_Drawdown', -0.20)
        dd_threshold = self.risk_thresholds['max_drawdown']
        
        if max_drawdown > dd_threshold * 0.5:  # Less severe drawdown
            score += 0.2
        elif max_drawdown < dd_threshold:  # More severe than threshold
            score -= 0.3
        
        # VaR analysis
        var_5 = metrics.get('VaR_5%_Historical', -0.05)
        if var_5 > -0.03:  # Low VaR (less risky)
            score += 0.2
        elif var_5 < -0.08:  # High VaR (more risky)
            score -= 0.2
        
        # Beta (market sensitivity)
        beta = metrics.get('Beta', 1.0)
        if self.risk_tolerance == "Conservative" and beta < 0.8:
            score += 0.1
        elif self.risk_tolerance == "Aggressive" and beta > 1.2:
            score += 0.1
        
        return np.clip(score, 0, 1)
    
    def _calculate_ml_adjustment(self, predictions: Dict) -> float:
        """Calculate adjustment based on ML predictions."""
        if not predictions:
            return 0
        
        adjustment = 0
        
        # Average prediction across models
        pred_values = [pred for pred in predictions.values() if isinstance(pred, (int, float))]
        
        if pred_values:
            avg_prediction = np.mean(pred_values)
            
            # Convert prediction to score adjustment
            if avg_prediction > 0.05:  # Predicted >5% return
                adjustment += 10
            elif avg_prediction > 0.02:  # Predicted >2% return
                adjustment += 5
            elif avg_prediction < -0.05:  # Predicted <-5% return
                adjustment -= 10
            elif avg_prediction < -0.02:  # Predicted <-2% return
                adjustment -= 5
        
        return adjustment
    
    def _calculate_market_adjustment(self, metrics: pd.Series, market_conditions: Dict) -> float:
        """Calculate adjustment based on current market conditions."""
        adjustment = 0
        
        # Market volatility adjustment
        market_vol = market_conditions.get('market_volatility', 0.20)
        if market_vol > 0.30:  # High market volatility
            # Favor low-beta, high-quality stocks
            beta = metrics.get('Beta', 1.0)
            sharpe = metrics.get('Sharpe_Ratio', 0)
            
            if beta < 0.8 and sharpe > 1.0:
                adjustment += 5  # Defensive quality stock
            elif beta > 1.2:
                adjustment -= 5  # High-beta stock in volatile market
        
        # Market trend adjustment
        market_trend = market_conditions.get('market_trend', 'neutral')
        correlation = metrics.get('Correlation', 0.7)
        
        if market_trend == 'bullish' and correlation > 0.8:
            adjustment += 3  # High correlation in bull market
        elif market_trend == 'bearish' and correlation > 0.8:
            adjustment -= 3  # High correlation in bear market
        
        return adjustment
    
    def _score_to_recommendation(self, score: float) -> RecommendationType:
        """Convert numerical score to recommendation type."""
        if score >= self.rec_thresholds['strong_buy']:
            return RecommendationType.STRONG_BUY
        elif score >= self.rec_thresholds['buy']:
            return RecommendationType.BUY
        elif score >= self.rec_thresholds['hold_lower']:
            return RecommendationType.HOLD
        elif score >= self.rec_thresholds['sell']:
            return RecommendationType.SELL
        else:
            return RecommendationType.STRONG_SELL
    
    def _calculate_confidence(self, metrics: pd.Series, score: float) -> float:
        """Calculate confidence level for the recommendation."""
        base_confidence = 0.7
        
        # Higher confidence for more extreme scores
        score_distance = abs(score - 50) / 50
        confidence_boost = score_distance * 0.3
        
        # Adjust based on data quality
        data_quality = 1.0
        
        # Reduce confidence if key metrics are missing
        key_metrics = ['Sharpe_Ratio', 'Volatility', 'Annual_Return', 'Max_Drawdown']
        missing_metrics = sum(1 for metric in key_metrics if pd.isna(metrics.get(metric, np.nan)))
        data_quality -= missing_metrics * 0.1
        
        # Reduce confidence for highly volatile stocks
        volatility = metrics.get('Volatility', 0.25)
        if volatility > 0.5:
            data_quality -= 0.2
        
        final_confidence = (base_confidence + confidence_boost) * data_quality
        return np.clip(final_confidence, 0.1, 0.95)
    
    def _generate_reasons(self, metrics: pd.Series, predictions: Dict, score: float) -> List[str]:
        """Generate human-readable reasons for the recommendation."""
        reasons = []
        
        # Performance reasons
        sharpe = metrics.get('Sharpe_Ratio', 0)
        if sharpe > 1.5:
            reasons.append(f"Excellent risk-adjusted returns (Sharpe: {sharpe:.2f})")
        elif sharpe > 1.0:
            reasons.append(f"Strong risk-adjusted returns (Sharpe: {sharpe:.2f})")
        elif sharpe < 0:
            reasons.append(f"Poor risk-adjusted returns (Sharpe: {sharpe:.2f})")
        
        # Momentum reasons
        annual_return = metrics.get('Annual_Return', 0)
        if annual_return > 0.20:
            reasons.append(f"Strong annual performance ({annual_return:.1%})")
        elif annual_return < -0.10:
            reasons.append(f"Weak annual performance ({annual_return:.1%})")
        
        # Technical reasons
        price_vs_sma20 = metrics.get('Price_vs_SMA20', 0)
        if price_vs_sma20 > 0.05:
            reasons.append("Price trending above 20-day moving average")
        elif price_vs_sma20 < -0.05:
            reasons.append("Price trending below 20-day moving average")
        
        rsi = metrics.get('RSI', 50)
        if rsi < 30:
            reasons.append("Technically oversold (RSI < 30)")
        elif rsi > 70:
            reasons.append("Technically overbought (RSI > 70)")
        
        # Risk reasons
        volatility = metrics.get('Volatility', 0.25)
        if volatility < 0.15:
            reasons.append("Low volatility provides stability")
        elif volatility > 0.40:
            reasons.append("High volatility indicates elevated risk")
        
        # ML prediction reasons
        if predictions:
            pred_values = [pred for pred in predictions.values() if isinstance(pred, (int, float))]
            if pred_values:
                avg_pred = np.mean(pred_values)
                if avg_pred > 0.05:
                    reasons.append("ML models predict positive returns")
                elif avg_pred < -0.05:
                    reasons.append("ML models predict negative returns")
        
        return reasons[:5]  # Limit to top 5 reasons
    
    def _identify_risk_factors(self, metrics: pd.Series) -> List[str]:
        """Identify key risk factors for the investment."""
        risk_factors = []
        
        # Volatility risk
        volatility = metrics.get('Volatility', 0.25)
        if volatility > 0.40:
            risk_factors.append(f"High volatility ({volatility:.1%} annually)")
        
        # Drawdown risk
        max_drawdown = metrics.get('Max_Drawdown', -0.20)
        if max_drawdown < -0.25:
            risk_factors.append(f"Significant historical drawdowns ({max_drawdown:.1%})")
        
        # Concentration risk
        correlation = metrics.get('Correlation', 0.7)
        if correlation > 0.9:
            risk_factors.append("High correlation with market movements")
        
        # Momentum risk
        momentum_3m = metrics.get('Momentum_3M', 0)
        if momentum_3m < -0.20:
            risk_factors.append("Recent negative momentum trend")
        
        # Quality risk
        win_rate = metrics.get('Win_Rate', 0.5)
        if win_rate < 0.45:
            risk_factors.append("Low historical win rate")
        
        # VaR risk
        var_5 = metrics.get('VaR_5%_Historical', -0.05)
        if var_5 < -0.10:
            risk_factors.append(f"High Value at Risk exposure ({var_5:.1%})")
        
        return risk_factors[:4]  # Limit to top 4 risk factors
    
    def _calculate_target_price(self, metrics: pd.Series, recommendation: RecommendationType) -> Optional[float]:
        """Calculate a simple target price based on recommendation."""
        current_price = metrics.get('Current_Price', 0)
        
        if current_price <= 0:
            return None
        
        # Simple target price calculation based on recommendation
        if recommendation == RecommendationType.STRONG_BUY:
            return current_price * 1.20  # 20% upside
        elif recommendation == RecommendationType.BUY:
            return current_price * 1.10  # 10% upside
        elif recommendation == RecommendationType.HOLD:
            return current_price * 1.02  # 2% upside
        elif recommendation == RecommendationType.SELL:
            return current_price * 0.95  # 5% downside
        else:  # STRONG_SELL
            return current_price * 0.85  # 15% downside