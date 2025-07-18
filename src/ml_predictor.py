# src/ml_predictor.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import joblib
import os
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedMLPredictor:
    """
    Institutional-grade machine learning prediction system for financial markets.
    Implements ensemble methods, feature engineering, and proper time series validation.
    """
    
    def __init__(self, prediction_horizons: List[int] = [1, 5, 10, 22]):
        """
        Initialize the ML prediction system.
        
        Args:
            prediction_horizons: List of prediction horizons in days
        """
        self.prediction_horizons = prediction_horizons
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Model configuration
        self.model_configs = {
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            },
            'ExtraTrees': {
                'model': ExtraTreesRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5]
                }
            },
            'Ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'ElasticNet': {
                'model': ElasticNet(random_state=42),
                'params': {
                    'alpha': [0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                }
            }
        }
    
    def engineer_features(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Engineer comprehensive features for ML prediction.
        
        Args:
            data: Historical stock data
            ticker: Stock symbol
            
        Returns:
            DataFrame with engineered features
        """
        df = data.copy()
        
        # Price-based features
        df['Price_MA_Ratio_5'] = df['Close'] / df['Close'].rolling(5).mean()
        df['Price_MA_Ratio_10'] = df['Close'] / df['Close'].rolling(10).mean()
        df['Price_MA_Ratio_20'] = df['Close'] / df['Close'].rolling(20).mean()
        
        # Volatility features
        df['Volatility_5d'] = df['Daily_Return'].rolling(5).std()
        df['Volatility_10d'] = df['Daily_Return'].rolling(10).std()
        df['Volatility_20d'] = df['Daily_Return'].rolling(20).std()
        df['Volatility_Ratio'] = df['Volatility_5d'] / df['Volatility_20d']
        
        # Volume features
        df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
        df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio_Current'] = df['Volume'] / df['Volume_MA_20']
        df['Volume_Price_Trend'] = df['Volume'] * df['Daily_Return']
        
        # Price pattern features
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_High_Ratio'] = df['Close'] / df['High']
        df['Close_Low_Ratio'] = df['Close'] / df['Low']
        df['OHLC_Average'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        
        # Momentum features
        for period in [3, 5, 10, 20]:
            df[f'Returns_{period}d'] = df['Close'].pct_change(period)
            df[f'Returns_Std_{period}d'] = df['Daily_Return'].rolling(period).std()
        
        # Technical indicator derivatives
        if 'RSI' in df.columns:
            df['RSI_Momentum'] = df['RSI'].diff()
            df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
            df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
        
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            df['MACD_Crossover'] = ((df['MACD'] > df['MACD_Signal']) & 
                                   (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))).astype(int)
            df['MACD_Momentum'] = df['MACD'].diff()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        df['BB_Std'] = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Support and Resistance levels
        df['Support_Level'] = df['Low'].rolling(20).min()
        df['Resistance_Level'] = df['High'].rolling(20).max()
        df['Support_Distance'] = (df['Close'] - df['Support_Level']) / df['Close']
        df['Resistance_Distance'] = (df['Resistance_Level'] - df['Close']) / df['Close']
        
        # Trend strength
        df['Trend_Strength'] = abs(df['Close'].rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        ))
        
        # Market microstructure
        df['Bid_Ask_Spread_Proxy'] = (df['High'] - df['Low']) / df['Close']
        df['Price_Efficiency'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'])
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Volatility_Lag_{lag}'] = df['Volatility_5d'].shift(lag)
        
        # Interaction features
        df['Volume_Volatility'] = df['Volume_Ratio_Current'] * df['Volatility_5d']
        df['Price_Volume_Momentum'] = df['Returns_5d'] * df['Volume_Ratio_Current']
        df['RSI_Volume'] = df.get('RSI', 50) * df['Volume_Ratio_Current']
        
        return df
    
    def create_target_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for different prediction horizons.
        
        Args:
            data: Feature-engineered data
            
        Returns:
            DataFrame with target variables
        """
        df = data.copy()
        
        for horizon in self.prediction_horizons:
            # Price return prediction
            df[f'Target_Return_{horizon}d'] = df['Close'].pct_change(horizon).shift(-horizon)
            
            # Price direction prediction (classification target)
            df[f'Target_Direction_{horizon}d'] = (df[f'Target_Return_{horizon}d'] > 0).astype(int)
            
            # Volatility prediction
            df[f'Target_Volatility_{horizon}d'] = df['Daily_Return'].rolling(horizon).std().shift(-horizon)
            
            # Risk-adjusted return prediction
            returns = df[f'Target_Return_{horizon}d']
            volatility = df[f'Target_Volatility_{horizon}d']
            df[f'Target_Sharpe_{horizon}d'] = returns / volatility
        
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'correlation') -> List[str]:
        """
        Select relevant features using various methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Feature selection method
            
        Returns:
            List of selected feature names
        """
        if method == 'correlation':
            # Remove highly correlated features
            corr_matrix = X.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
            selected_features = [col for col in X.columns if col not in to_drop]
            
        elif method == 'statistical':
            # Statistical feature selection
            selector = SelectKBest(score_func=f_regression, k=min(50, len(X.columns)))
            selector.fit(X.fillna(0), y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'rfe':
            # Recursive feature elimination
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=min(30, len(X.columns)))
            selector.fit(X.fillna(0), y.fillna(0))
            selected_features = X.columns[selector.support_].tolist()
            
        else:  # 'all'
            selected_features = X.columns.tolist()
        
        # Remove features with too many NaN values
        selected_features = [
            col for col in selected_features 
            if X[col].notna().sum() > len(X) * 0.5
        ]
        
        logger.info(f"Selected {len(selected_features)} features using {method} method")
        return selected_features
    
    def train_ensemble_models(self, data: pd.DataFrame, ticker: str, 
                            target_horizon: int = 5) -> Dict[str, Dict]:
        """
        Train ensemble of ML models for a specific stock and prediction horizon.
        
        Args:
            data: Prepared training data
            ticker: Stock symbol
            target_horizon: Prediction horizon in days
            
        Returns:
            Dictionary containing trained models and performance metrics
        """
        logger.info(f"Training models for {ticker} with {target_horizon}d horizon")
        
        # Prepare features and targets
        feature_columns = [col for col in data.columns if not col.startswith('Target_')]
        target_column = f'Target_Return_{target_horizon}d'
        
        if target_column not in data.columns:
            logger.error(f"Target column {target_column} not found")
            return {}
        
        # Clean data
        clean_data = data[feature_columns + [target_column]].dropna()
        
        if len(clean_data) < 100:
            logger.warning(f"Insufficient data for {ticker}: {len(clean_data)} samples")
            return {}
        
        X = clean_data[feature_columns]
        y = clean_data[target_column]
        
        # Feature selection
        selected_features = self.select_features(X, y, method='correlation')
        X_selected = X[selected_features]
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train models
        model_results = {}
        
        for model_name, config in self.model_configs.items():
            try:
                logger.info(f"Training {model_name}...")
                
                # Scale features for linear models
                if model_name in ['Ridge', 'ElasticNet', 'SVR']:
                    scaler = RobustScaler()
                    X_scaled = pd.DataFrame(
                        scaler.fit_transform(X_selected.fillna(0)),
                        columns=X_selected.columns,
                        index=X_selected.index
                    )
                else:
                    scaler = None
                    X_scaled = X_selected.fillna(0)
                
                # Grid search with time series cross-validation
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_scaled, y)
                best_model = grid_search.best_estimator_
                
                # Final training on full dataset
                best_model.fit(X_scaled, y)
                
                # Performance evaluation
                y_pred = best_model.predict(X_scaled)
                
                performance = {
                    'mse': mean_squared_error(y, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                    'mae': mean_absolute_error(y, y_pred),
                    'r2': r2_score(y, y_pred),
                    'mape': mean_absolute_percentage_error(y, y_pred) if np.all(y != 0) else np.inf
                }
                
                # Feature importance (for tree-based models)
                if hasattr(best_model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': X_scaled.columns,
                        'importance': best_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                else:
                    feature_importance = None
                
                model_results[model_name] = {
                    'model': best_model,
                    'scaler': scaler,
                    'performance': performance,
                    'feature_importance': feature_importance,
                    'best_params': grid_search.best_params_,
                    'selected_features': selected_features
                }
                
                logger.info(f"{model_name} R² score: {performance['r2']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Create ensemble prediction
        if len(model_results) > 1:
            ensemble_prediction = self._create_ensemble_prediction(model_results, X_scaled, y)
            model_results['Ensemble'] = ensemble_prediction
        
        # Save models
        self._save_models(model_results, ticker, target_horizon)
        
        return model_results
    
    def _create_ensemble_prediction(self, model_results: Dict, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Create ensemble prediction using weighted average based on performance."""
        predictions = {}
        weights = {}
        
        # Calculate weights based on R² scores
        total_r2 = sum([result['performance']['r2'] for result in model_results.values() 
                       if result['performance']['r2'] > 0])
        
        if total_r2 <= 0:
            # Equal weights if no positive R² scores
            weights = {name: 1/len(model_results) for name in model_results.keys()}
        else:
            weights = {name: max(0, result['performance']['r2']) / total_r2 
                      for name, result in model_results.items()}
        
        # Generate ensemble prediction
        ensemble_pred = np.zeros(len(X))
        
        for name, result in model_results.items():
            model = result['model']
            scaler = result.get('scaler')
            
            if scaler:
                X_scaled = scaler.transform(X.fillna(0))
                pred = model.predict(X_scaled)
            else:
                pred = model.predict(X.fillna(0))
            
            ensemble_pred += weights[name] * pred
        
        # Calculate ensemble performance
        ensemble_performance = {
            'mse': mean_squared_error(y, ensemble_pred),
            'rmse': np.sqrt(mean_squared_error(y, ensemble_pred)),
            'mae': mean_absolute_error(y, ensemble_pred),
            'r2': r2_score(y, ensemble_pred),
            'mape': mean_absolute_percentage_error(y, ensemble_pred) if np.all(y != 0) else np.inf
        }
        
        return {
            'weights': weights,
            'performance': ensemble_performance,
            'predictions': ensemble_pred
        }
    
    def _save_models(self, model_results: Dict, ticker: str, horizon: int):
        """Save trained models to disk."""
        save_dir = f"data/models/{ticker}"
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, result in model_results.items():
            if model_name != 'Ensemble':
                model_path = f"{save_dir}/{model_name}_{horizon}d.pkl"
                joblib.dump({
                    'model': result['model'],
                    'scaler': result.get('scaler'),
                    'selected_features': result['selected_features'],
                    'performance': result['performance']
                }, model_path)
        
        logger.info(f"Models saved for {ticker} ({horizon}d horizon)")
    
    def predict_future_returns(self, data: pd.DataFrame, ticker: str, 
                             horizon: int = 5) -> Dict[str, float]:
        """
        Generate predictions for future returns using trained models.
        
        Args:
            data: Current stock data
            ticker: Stock symbol
            horizon: Prediction horizon
            
        Returns:
            Dictionary with predictions from different models
        """
        model_dir = f"data/models/{ticker}"
        
        if not os.path.exists(model_dir):
            logger.warning(f"No trained models found for {ticker}")
            return {}
        
        # Engineer features for the latest data point
        featured_data = self.engineer_features(data, ticker)
        latest_data = featured_data.iloc[[-1]]  # Get most recent row
        
        predictions = {}
        
        for model_file in os.listdir(model_dir):
            if model_file.endswith(f'{horizon}d.pkl'):
                model_name = model_file.replace(f'_{horizon}d.pkl', '')
                
                try:
                    model_path = os.path.join(model_dir, model_file)
                    saved_model = joblib.load(model_path)
                    
                    model = saved_model['model']
                    scaler = saved_model.get('scaler')
                    selected_features = saved_model['selected_features']
                    
                    # Prepare features
                    X = latest_data[selected_features].fillna(0)
                    
                    if scaler:
                        X_scaled = scaler.transform(X)
                        prediction = model.predict(X_scaled)[0]
                    else:
                        prediction = model.predict(X)[0]
                    
                    predictions[model_name] = prediction
                    
                except Exception as e:
                    logger.error(f"Error making prediction with {model_name}: {str(e)}")
                    continue
        
        return predictions
    
    def backtest_strategy(self, data: pd.DataFrame, ticker: str, 
                         horizon: int = 5, threshold: float = 0.02) -> Dict[str, float]:
        """
        Backtest trading strategy based on ML predictions.
        
        Args:
            data: Historical stock data with predictions
            ticker: Stock symbol
            horizon: Prediction horizon
            threshold: Minimum predicted return to trigger trade
            
        Returns:
            Dictionary with backtesting results
        """
        # Create simple trading strategy based on predictions
        # This is a simplified example - real strategies would be more sophisticated
        
        if len(data) < 100:
            return {'error': 'Insufficient data for backtesting'}
        
        # Generate predictions for historical data
        featured_data = self.engineer_features(data, ticker)
        featured_data = self.create_target_variables(featured_data)
        
        # Simulate trading signals
        signals = []
        returns = []
        
        for i in range(len(featured_data) - horizon):
            # Use features up to current point to predict future return
            current_features = featured_data.iloc[i:i+50]  # Use 50-day window
            
            if len(current_features) < 50:
                continue
            
            # Simplified prediction (in practice, would use trained model)
            recent_momentum = current_features['Daily_Return'].tail(5).mean()
            technical_signal = current_features['Price_vs_SMA20'].iloc[-1]
            
            predicted_return = recent_momentum * 0.5 + technical_signal * 0.3
            
            # Generate trading signal
            if predicted_return > threshold:
                signal = 1  # Buy
            elif predicted_return < -threshold:
                signal = -1  # Sell
            else:
                signal = 0  # Hold
            
            signals.append(signal)
            
            # Calculate actual return over horizon
            if i + horizon < len(featured_data):
                actual_return = featured_data['Daily_Return'].iloc[i+1:i+1+horizon].sum()
                returns.append(actual_return * signal)  # Strategy return
        
        if not returns:
            return {'error': 'No valid trading signals generated'}
        
        # Calculate strategy performance
        strategy_returns = pd.Series(returns)
        
        total_return = (1 + strategy_returns).prod() - 1
        annual_return = strategy_returns.mean() * 252
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().expanding().max()).min()
        
        win_rate = len(strategy_returns[strategy_returns > 0]) / len(strategy_returns)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'number_of_trades': len([s for s in signals if s != 0])
        }