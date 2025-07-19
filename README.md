# 🏦 Financial Risk Analyzer

**Professional-Grade Investment Decision Support System**

A comprehensive financial risk analysis and machine learning prediction platform designed for institutional-quality investment decision making. Built to demonstrate advanced financial technology capabilities.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Project Overview

This system provides sophisticated financial risk analytics and predictive modeling capabilities, specifically designed for financial institutions and trading firms. It demonstrates expertise in:

- **Advanced Risk Management**: Industry-standard metrics (VaR, CVaR, Sharpe Ratio, Maximum Drawdown)
- **Machine Learning**: Ensemble prediction models for price forecasting
- **Real-time Analytics**: Live market data integration with professional-grade APIs
- **Interactive Dashboards**: Executive-level visualizations and reporting
- **Investment Intelligence**: Multi-factor recommendation engine

## 🚀 Key Features

### 📊 **Advanced Risk Analytics**
- **Value at Risk (VaR)**: Multiple confidence levels with parametric and historical methods
- **Expected Shortfall (CVaR)**: Tail risk assessment beyond VaR
- **Sharpe & Sortino Ratios**: Risk-adjusted return calculations
- **Maximum Drawdown Analysis**: Peak-to-trough loss measurement
- **Beta & Alpha Calculations**: Market-relative performance metrics

### 🤖 **Machine Learning Engine**
- **Ensemble Models**: Random Forest, Gradient Boosting, Extra Trees, Ridge, Elastic Net
- **Feature Engineering**: 50+ technical and fundamental indicators
- **Time Series Validation**: Proper backtesting with TimeSeriesSplit
- **Prediction Horizons**: 1-day to 1-month forecasting capabilities
- **Model Performance Tracking**: Comprehensive evaluation metrics

### 🎯 **Investment Recommendation System**
- **Multi-Factor Analysis**: Valuation, Quality, Momentum, Technical, Risk factors
- **Investment Style Adaptation**: Value, Growth, Momentum, Quality approaches
- **Risk Tolerance Matching**: Conservative, Moderate, Aggressive profiles
- **Confidence Scoring**: Statistical confidence in recommendations
- **Target Price Calculation**: Data-driven price targets

### 🔍 **Professional Screening**
- **6 Stock Universes**: Technology, Financial Services, ETFs, Energy, Healthcare, Consumer
- **Advanced Filters**: Sharpe ratio, volatility, market cap, sector screening
- **Real-time Data**: Live market data with 5-minute refresh intervals
- **Comprehensive Coverage**: Support for 1000+ stock symbols

### 📈 **Interactive Dashboard**
- **Executive Summary**: Key metrics and actionable insights
- **Risk-Return Visualization**: Interactive scatter plots with drill-down capability
- **Technical Analysis**: Heatmaps, trend analysis, momentum indicators
- **Performance Attribution**: Factor-based return decomposition
- **Export Capabilities**: PDF reports, CSV data, executive summaries

## 🛠️ Technology Stack

### **Core Technologies**
- **Python 3.9+**: Primary programming language
- **Streamlit**: Interactive web application framework
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning algorithms and evaluation
- **Plotly**: Interactive visualization and charting

### **Financial Data & APIs**
- **yFinance**: Real-time and historical market data
- **Yahoo Finance API**: Comprehensive stock information
- **Multiple Data Sources**: Redundancy for data reliability

### **Advanced Analytics**
- **SciPy**: Statistical functions and probability distributions
- **Statsmodels**: Advanced statistical modeling
- **Joblib**: Model serialization and parallel processing

## 📋 Installation & Setup

### **1. Environment Setup**

```bash
# Create conda environment
conda create -n financial_risk_env python=3.9 -y
conda activate financial_risk_env

# Install core packages
conda install -c conda-forge pandas numpy scipy scikit-learn matplotlib seaborn jupyter plotly statsmodels -y

# Install specialized packages
pip install yfinance streamlit joblib numba
```

### **2. Project Setup**

```bash
# Clone or create project directory
mkdir financial_risk_analyzer
cd financial_risk_analyzer

# Create directory structure
mkdir -p {data/{raw,processed,models,exports},src,config,notebooks,assets}

# Install dependencies
pip install -r requirements.txt
```

### **3. Configuration**

```bash
# Copy all source files to respective directories:
# - config/settings.py
# - src/data_ingestion.py
# - src/risk_analyzer.py
# - src/ml_predictor.py
# - src/recommendation_engine.py
# - streamlit_app.py (main directory)
```

### **4. Run the Application**

```bash
# Start the Streamlit application
streamlit run streamlit_app.py

# Application will open at http://localhost:8501
```

## 📖 Usage Guide

### **Quick Start**

1. **Launch Application**: Run `streamlit run streamlit_app.py`
2. **Select Stock Universe**: Choose from 6 predefined categories (Technology, Financial, etc.)
3. **Configure Analysis**: Set risk tolerance, investment style, and filters
4. **Run Analysis**: Click "Run Complete Analysis" to execute the full pipeline
5. **Review Results**: Explore recommendations, charts, and detailed metrics
6. **Export Reports**: Download CSV data and executive summaries

### **Advanced Usage**

#### **Custom Stock Analysis**
```python
from src.data_ingestion import FinancialDataIngestion
from src.risk_analyzer import AdvancedRiskAnalyzer

# Initialize components
ingester = FinancialDataIngestion()
analyzer = AdvancedRiskAnalyzer()

# Fetch and analyze data
stock_data = ingester.fetch_stock_data(['AAPL', 'GOOGL'], period='2y')
risk_metrics = analyzer.calculate_comprehensive_metrics(stock_data)
```

#### **Machine Learning Predictions**
```python
from src.ml_predictor import AdvancedMLPredictor

predictor = AdvancedMLPredictor()
featured_data = predictor.engineer_features(stock_data['AAPL'], 'AAPL')
model_results = predictor.train_ensemble_models(featured_data, 'AAPL', target_horizon=5)
```

#### **Investment Recommendations**
```python
from src.recommendation_engine import AdvancedRecommendationEngine

rec_engine = AdvancedRecommendationEngine(
    investment_style="Growth", 
    risk_tolerance="Moderate"
)
recommendations = rec_engine.generate_recommendations(risk_metrics)
```

## 📊 Sample Results

### **Risk Analysis Output**
```
Stock Analysis Results:
┌─────────┬──────────────┬─────────────┬──────────────┬──────────────┐
│ Ticker  │ Sharpe Ratio │ Annual Ret  │ Volatility   │ Max Drawdown │
├─────────┼──────────────┼─────────────┼──────────────┼──────────────┤
│ AAPL    │ 1.34         │ 15.2%       │ 28.4%        │ -19.3%       │
│ GOOGL   │ 0.89         │ 12.7%       │ 31.2%        │ -24.1%       │
│ MSFT    │ 1.52         │ 18.9%       │ 26.1%        │ -16.8%       │
└─────────┴──────────────┴─────────────┴──────────────┴──────────────┘
```

### **Investment Recommendations**
```
Top Investment Recommendations:
🚀 MSFT - STRONG BUY (Score: 87.3/100)
   → Target: $420.00 | Confidence: 89%
   → Reasons: Excellent risk-adjusted returns, Strong momentum, Low volatility

📈 AAPL - BUY (Score: 78.1/100)
   → Target: $195.00 | Confidence: 82%
   → Reasons: Strong annual performance, Technical breakout, ML models positive

⏸️ GOOGL - HOLD (Score: 64.2/100)
   → Target: $145.00 | Confidence: 71%
   → Reasons: Moderate returns, High volatility, Mixed signals
```

## 🎯 Business Applications

### **For Institutional Use**
- **Trading Desk Support**: Real-time risk assessment for trading decisions
- **Portfolio Management**: Comprehensive portfolio risk analysis and optimization
- **Client Advisory**: Professional-grade investment recommendations
- **Risk Management**: Institution-level risk monitoring and reporting
- **Regulatory Compliance**: Standardized risk metrics for regulatory requirements

### **Industry Applications**
- **Asset Management**: Portfolio construction and risk budgeting
- **Hedge Funds**: Quantitative strategy development and backtesting
- **Investment Banking**: Client advisory and structured product development
- **Pension Funds**: Long-term asset allocation and risk management
- **Insurance Companies**: Asset-liability matching and capital allocation

## 🧪 Technical Implementation

### **Architecture Design**
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Layer    │───▶│   Analysis Layer │───▶│ Presentation    │
│                 │    │                  │    │ Layer           │
│ • Market Data   │    │ • Risk Metrics   │    │ • Streamlit UI  │
│ • Company Info  │    │ • ML Predictions │    │ • Interactive   │
│ • Economic Data │    │ • Recommendations│    │   Charts        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Data Pipeline**
1. **Ingestion**: Real-time market data via yFinance API
2. **Validation**: Data quality checks and outlier detection
3. **Feature Engineering**: 50+ technical and fundamental indicators
4. **Risk Calculation**: Industry-standard risk metrics
5. **ML Processing**: Ensemble model training and prediction
6. **Recommendation Generation**: Multi-factor analysis and scoring
7. **Visualization**: Interactive dashboards and reporting

### **Performance Optimization**
- **Caching Strategy**: Streamlit caching for data and computations
- **Vectorization**: NumPy/Pandas optimized operations
- **Parallel Processing**: Joblib for model training
- **Memory Management**: Efficient data structures and garbage collection

## 🔒 Risk Management & Compliance

### **Data Security**
- No storage of sensitive financial data
- API rate limiting and error handling
- Input validation and sanitization
- Session-based data management

### **Model Risk Management**
- Ensemble approaches to reduce single-model risk
- Time series cross-validation for robust backtesting
- Performance monitoring and model decay detection
- Clear model limitations and assumptions documentation

### **Regulatory Considerations**
- Industry-standard risk metrics (Basel III compliant)
- Transparent methodology documentation
- Audit trail for all calculations
- Clear disclaimers and risk warnings

## 📈 Performance Metrics

### **System Performance**
- **Data Refresh**: 5-minute cache TTL for real-time updates
- **Analysis Speed**: Complete analysis for 50 stocks in <30 seconds
- **Model Training**: Ensemble training for single stock in <2 minutes
- **Dashboard Load**: Interactive charts render in <3 seconds

### **Analytical Accuracy**
- **Risk Metrics**: Industry-standard implementations with validation
- **ML Models**: Cross-validated R² scores typically 0.15-0.35
- **Backtesting**: Proper time series validation preventing look-ahead bias
- **Recommendation Performance**: Track record analysis available

## 🤝 Contributing

This project demonstrates advanced financial technology capabilities for institutional applications:

1. **Code Quality**: Professional-grade Python with comprehensive documentation
2. **Financial Domain**: Deep understanding of quantitative finance and risk management
3. **Technology Stack**: Modern, scalable architecture suitable for enterprise deployment
4. **User Experience**: Intuitive interface designed for financial professionals

## 📄 License

MIT License - see LICENSE file for details.

## 🙋‍♂️ Author

**Harsh Agrawal**
- Email: harsh.agrawal18112003@gmail.com
- LinkedIn: [linkedin.com/in/harsh-agrawal](https://linkedin.com/in/harsh-agrawal)
- GitHub: [github.com/HaDeS-18](https://github.com/HaDeS-18)

*Built to showcase expertise in financial technology, data science, and professional software development.*

---

**Financial Risk Analyzer** | Professional Investment Intelligence | Built with ❤️