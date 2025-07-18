# config/settings.py
import os
from datetime import datetime, timedelta

# Application Configuration
APP_CONFIG = {
    "title": "ION Financial Risk Analyzer",
    "subtitle": "Professional Investment Decision Support System",
    "icon": "📈",
    "layout": "wide",
    "version": "1.0.0"
}

# Data Configuration
DATA_CONFIG = {
    "default_period": "2y",
    "cache_ttl": 300,  # 5 minutes
    "risk_free_rate": 0.02,  # 2% annual
    "trading_days": 252
}

# Stock Universes
STOCK_UNIVERSES = {
    "Technology": {
        "stocks": ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'CRM', 'ORCL'],
        "description": "Large-cap technology companies"
    },
    "Financial Services": {
        "stocks": ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'BK', 'AXP'],
        "description": "Major banks and financial institutions"
    },
    "Market ETFs": {
        "stocks": ['SPY', 'QQQ', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLU', 'VTI', 'ARKK'],
        "description": "Sector and market ETFs"
    },
    "Energy": {
        "stocks": ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI'],
        "description": "Energy sector companies"
    },
    "Healthcare": {
        "stocks": ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'DHR', 'ABT', 'BMY', 'MRK', 'LLY'],
        "description": "Healthcare and pharmaceutical companies"
    },
    "Consumer": {
        "stocks": ['AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'WMT', 'COST', 'LOW', 'DIS'],
        "description": "Consumer discretionary and staples"
    }
}

# Risk Profiles
RISK_PROFILES = {
    "Conservative": {
        "max_volatility": 0.20,
        "min_sharpe": 0.5,
        "max_drawdown": -0.15,
        "description": "Low risk, stable returns",
        "color": "#28a745"
    },
    "Moderate": {
        "max_volatility": 0.35,
        "min_sharpe": 0.3,
        "max_drawdown": -0.25,
        "description": "Balanced risk-return profile",
        "color": "#ffc107"
    },
    "Aggressive": {
        "max_volatility": 0.60,
        "min_sharpe": 0.0,
        "max_drawdown": -0.40,
        "description": "High risk, high potential returns",
        "color": "#dc3545"
    }
}

# Investment Styles
INVESTMENT_STYLES = {
    "Value": {
        "focus": "Undervalued stocks with strong fundamentals",
        "metrics_weight": {"sharpe": 0.3, "volatility": 0.4, "momentum": 0.1, "technical": 0.2}
    },
    "Growth": {
        "focus": "High-growth potential companies",
        "metrics_weight": {"sharpe": 0.2, "volatility": 0.2, "momentum": 0.4, "technical": 0.2}
    },
    "Momentum": {
        "focus": "Strong price and earnings momentum",
        "metrics_weight": {"sharpe": 0.2, "volatility": 0.1, "momentum": 0.5, "technical": 0.2}
    },
    "Quality": {
        "focus": "High-quality companies with consistent performance",
        "metrics_weight": {"sharpe": 0.4, "volatility": 0.3, "momentum": 0.1, "technical": 0.2}
    }
}

# Scoring Thresholds
SCORING_THRESHOLDS = {
    "sharpe_excellent": 1.5,
    "sharpe_good": 1.0,
    "sharpe_fair": 0.5,
    "volatility_low": 0.15,
    "volatility_moderate": 0.25,
    "volatility_high": 0.40,
    "momentum_strong": 0.10,
    "momentum_weak": -0.05,
    "technical_bullish": 0.05,
    "technical_bearish": -0.02
}

# Chart Configuration
CHART_CONFIG = {
    "colors": {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e", 
        "success": "#2ca02c",
        "danger": "#d62728",
        "warning": "#ff7f0e",
        "info": "#17becf",
        "buy": "#00c851",
        "sell": "#ff4444", 
        "hold": "#ffbb33"
    },
    "templates": {
        "main": "plotly_white",
        "dark": "plotly_dark"
    }
}

# File Paths
PATHS = {
    "data_raw": "data/raw/",
    "data_processed": "data/processed/",
    "models": "data/models/",
    "exports": "data/exports/",
    "assets": "assets/"
}

# Ensure directories exist
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)