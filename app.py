import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append('src')
sys.path.append('config')

from data_ingestion import FinancialDataIngestion
from risk_analyzer import AdvancedRiskAnalyzer
from ml_predictor import AdvancedMLPredictor
from recommendation_engine import AdvancedRecommendationEngine, RecommendationType
from settings import *

# Page configuration
st.set_page_config(
    page_title=APP_CONFIG["title"],
    page_icon=APP_CONFIG["icon"],
    layout=APP_CONFIG["layout"],
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    
    .metric-card {
        background: black;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
        color: white;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .recommendation-strong-buy {
        background: linear-gradient(135deg, #00c851 0%, #007e33 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,200,81,0.3);
    }
    
    .recommendation-buy {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(40,167,69,0.3);
    }
    
    .recommendation-hold {
        background: linear-gradient(135deg, #ffc107 0%, #fd7e14 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(255,193,7,0.3);
    }
    
    .recommendation-sell {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(220,53,69,0.3);
    }
    
    .recommendation-strong-sell {
        background: linear-gradient(135deg, #721c24 0%, #dc3545 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(114,28,36,0.3);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        color: black;
    }
    
    .analysis-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        color: black;
    }
    
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        color: black;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        margin-top: 3rem;
    }
    
    .recommendations-text {
        color: black;
    }
    
    .recommended-stocks {
        color: black;
    }
</style>
""", unsafe_allow_html=True)


# General CSS styling
st.markdown(
    """
    <style>
    body, .stText, .stMarkdown, .stDataFrame, .stTable, .stMetric, .stButton, .stSidebar {
        color: black; /* Set a default text color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

class FinancialDashboard:
    """Professional financial analysis dashboard for ION Group demonstration."""
    
    def __init__(self):
        self.data_ingester = FinancialDataIngestion()
        self.risk_analyzer = AdvancedRiskAnalyzer()
        self.ml_predictor = AdvancedMLPredictor()
        
        # Initialize session state
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'stock_data' not in st.session_state:
            st.session_state.stock_data = {}
        if 'risk_metrics' not in st.session_state:
            st.session_state.risk_metrics = pd.DataFrame()
        if 'recommendations' not in st.session_state:
            st.session_state.recommendations = []

    def render_header(self):
        """Render the main application header."""
        st.markdown(f"""
        <div class="main-header">
            <h1>{APP_CONFIG["icon"]} {APP_CONFIG["title"]}</h1>
            <p style="font-size: 1.2em; margin-top: 1rem;">{APP_CONFIG["subtitle"]}</p>
            <p style="font-size: 0.9em; opacity: 0.9;">Version {APP_CONFIG["version"]} | Real-time Financial Intelligence</p>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar with filters and controls."""
        with st.sidebar:
            st.markdown('<div class="sidebar-header"><h2>🎯 Analysis Configuration</h2></div>', unsafe_allow_html=True)
            
            # Stock Universe Selection
            st.subheader("📊 Stock Universe")
            universe_choice = st.selectbox(
                "Select Stock Category",
                options=list(STOCK_UNIVERSES.keys()),
                help="Choose a predefined category of stocks to analyze"
            )
            
            # Display universe description
            universe_info = STOCK_UNIVERSES[universe_choice]
            st.info(f"**{universe_choice}**: {universe_info['description']}")
            
            # Custom stock selection
            available_stocks = universe_info['stocks']
            selected_stocks = st.multiselect(
                "Customize Stock Selection",
                options=available_stocks,
                default=available_stocks[:8],  # Default to first 8
                help="Select specific stocks from the chosen universe"
            )
            
            # Analysis Parameters
            st.subheader("⚙️ Analysis Parameters")
            
            time_period = st.selectbox(
                "Historical Data Period",
                options=['1y', '2y', '3y', '5y'],
                index=1,
                help="Amount of historical data to analyze"
            )
            
            # Investment Profile
            st.subheader("👤 Investment Profile")
            
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                options=list(RISK_PROFILES.keys()),
                index=1,
                help="Your comfort level with investment risk"
            )
            
            # Display risk profile info
            risk_info = RISK_PROFILES[risk_tolerance]
            st.markdown(f"""
            **{risk_tolerance} Profile:**
            - {risk_info['description']}
            - Max Volatility: {risk_info['max_volatility']:.1%}
            - Min Sharpe Ratio: {risk_info['min_sharpe']:.1f}
            """)
            
            investment_style = st.selectbox(
                "Investment Style",
                options=list(INVESTMENT_STYLES.keys()),
                index=2,
                help="Your preferred investment approach"
            )
            
            # Display investment style info
            style_info = INVESTMENT_STYLES[investment_style]
            st.info(f"**{investment_style}**: {style_info['focus']}")
            
            # Advanced Filters
            with st.expander("🔧 Advanced Filters"):
                min_sharpe = st.slider(
                    "Minimum Sharpe Ratio",
                    min_value=-1.0,
                    max_value=3.0,
                    value=risk_info['min_sharpe'],
                    step=0.1,
                    help="Filter stocks by minimum risk-adjusted returns"
                )
                
                max_volatility = st.slider(
                    "Maximum Volatility",
                    min_value=0.05,
                    max_value=1.0,
                    value=risk_info['max_volatility'],
                    step=0.05,
                    format="%.2f",
                    help="Filter stocks by maximum volatility"
                )
                
                min_market_cap = st.selectbox(
                    "Minimum Market Cap",
                    options=["Any", "Small Cap ($300M+)", "Mid Cap ($2B+)", "Large Cap ($10B+)"],
                    index=0
                )
            
            # Analysis Controls
            st.subheader("🚀 Analysis Controls")
            
            if st.button("🔄 Run Complete Analysis", type="primary", use_container_width=True):
                if selected_stocks:
                    self.run_analysis(selected_stocks, time_period, risk_tolerance, investment_style, 
                                    min_sharpe, max_volatility)
                else:
                    st.error("Please select at least one stock to analyze")
            
            # Export Options
            if st.session_state.analysis_complete:
                st.subheader("💾 Export Results")
                
                if st.button("📥 Download Analysis Report", use_container_width=True):
                    self.generate_export_data()
        
        return selected_stocks, time_period, risk_tolerance, investment_style, min_sharpe, max_volatility

    def run_analysis(self, stocks, period, risk_tolerance, investment_style, min_sharpe, max_volatility):
        """Run the complete financial analysis pipeline."""
        
        with st.spinner('🔄 Running comprehensive financial analysis...'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Data Ingestion
                status_text.text("📊 Fetching market data...")
                progress_bar.progress(20)
                
                stock_data = self.data_ingester.fetch_stock_data(stocks, period)
                
                if not stock_data:
                    st.error("❌ Failed to fetch stock data. Please try again.")
                    return
                
                # Step 2: Risk Analysis
                status_text.text("⚖️ Calculating risk metrics...")
                progress_bar.progress(50)
                
                risk_metrics = self.risk_analyzer.calculate_comprehensive_metrics(stock_data)
                
                # Step 3: Apply Filters
                status_text.text("🔍 Applying filters...")
                progress_bar.progress(70)
                
                filtered_metrics = self.apply_filters(risk_metrics, min_sharpe, max_volatility)
                
                # Step 4: Generate Recommendations
                status_text.text("🎯 Generating recommendations...")
                progress_bar.progress(85)
                
                rec_engine = AdvancedRecommendationEngine(investment_style, risk_tolerance)
                recommendations = rec_engine.generate_recommendations(filtered_metrics)
                
                # Step 5: Complete
                status_text.text("✅ Analysis complete!")
                progress_bar.progress(100)
                
                # Store results in session state
                st.session_state.stock_data = stock_data
                st.session_state.risk_metrics = filtered_metrics
                st.session_state.recommendations = recommendations
                st.session_state.analysis_complete = True
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✅ Analysis completed successfully! Found {len(recommendations)} investment opportunities.")
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)

    def apply_filters(self, risk_metrics, min_sharpe, max_volatility):
        """Apply user-defined filters to the risk metrics."""
        if risk_metrics.empty:
            return risk_metrics
        
        filtered = risk_metrics[
            (risk_metrics['Sharpe_Ratio'] >= min_sharpe) &
            (risk_metrics['Volatility'] <= max_volatility)
        ]
        
        return filtered

    def render_dashboard_overview(self):
        """Render the main dashboard overview."""
        if not st.session_state.analysis_complete:
            st.markdown("""
            <div class="analysis-section">
                <h2>🚀 Welcome to ION Financial Risk Analyzer</h2>
                <p>Professional-grade investment analysis platform designed for institutional decision-making.</p>
                <h3>🎯 Key Features:</h3>
                <ul>
                    <li><strong>Advanced Risk Analytics:</strong> VaR, CVaR, Sharpe Ratio, Maximum Drawdown</li>
                    <li><strong>Machine Learning Predictions:</strong> Ensemble models for price forecasting</li>
                    <li><strong>Professional Recommendations:</strong> Buy/Sell/Hold signals with detailed reasoning</li>
                    <li><strong>Real-time Data:</strong> Live market data integration with institutional-grade APIs</li>
                    <li><strong>Comprehensive Screening:</strong> Multi-factor analysis across 6 stock universes</li>
                </ul>
                <p><strong>👈 Configure your analysis in the sidebar and click "Run Complete Analysis" to begin.</strong></p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Main dashboard with results
        recommendations = st.session_state.recommendations
        risk_metrics = st.session_state.risk_metrics
        
        if recommendations:
            # Key Metrics Overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_analyzed = len(risk_metrics)
                st.metric("🔍 Stocks Analyzed", total_analyzed)
            
            with col2:
                buy_signals = len([r for r in recommendations if r.recommendation.value in ['STRONG_BUY', 'BUY']])
                st.metric("📈 Buy Signals", buy_signals, delta=f"{buy_signals/total_analyzed:.1%}")
            
            with col3:
                avg_score = np.mean([r.score for r in recommendations])
                st.metric("⭐ Average Score", f"{avg_score:.1f}/100")
            
            with col4:
                high_confidence = len([r for r in recommendations if r.confidence > 0.8])
                st.metric("🎯 High Confidence", high_confidence, delta=f"{high_confidence/len(recommendations):.1%}")

    def render_recommendations_section(self):
        """Render the investment recommendations section."""
        if not st.session_state.analysis_complete or not st.session_state.recommendations:
            return
        
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.header("🎯 Investment Recommendations")
        
        recommendations = st.session_state.recommendations
        
        # Recommendation Summary
        rec_summary = {}
        for rec in recommendations:
            rec_type = rec.recommendation.value
            rec_summary[rec_type] = rec_summary.get(rec_type, 0) + 1
        
        # Display summary
        summary_cols = st.columns(len(rec_summary))
        for i, (rec_type, count) in enumerate(rec_summary.items()):
            with summary_cols[i]:
                color = self.get_recommendation_color(rec_type)
                st.markdown(f"""
                <div style="background: {color}; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                    <h4>{rec_type.replace('_', ' ')}</h4>
                    <h2>{count}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed Recommendations
        for i, rec in enumerate(recommendations):
            with st.expander(f"{self.get_recommendation_emoji(rec.recommendation.value)} {rec.ticker} - {rec.recommendation.value.replace('_', ' ')} (Score: {rec.score:.1f})", expanded=i<3):
                
                col1, col2, col3 = st.columns([2, 2, 3])
                
                with col1:
                    st.metric("Current Price", f"${rec.current_price:.2f}")
                    if rec.target_price:
                        upside = (rec.target_price - rec.current_price) / rec.current_price
                        st.metric("Target Price", f"${rec.target_price:.2f}", delta=f"{upside:.1%}")
                    st.metric("Confidence", f"{rec.confidence:.0%}")
                
                with col2:
                    # Get stock metrics
                    if rec.ticker in st.session_state.risk_metrics.index:
                        metrics = st.session_state.risk_metrics.loc[rec.ticker]
                        st.metric("Sharpe Ratio", f"{metrics.get('Sharpe_Ratio', 0):.2f}")
                        st.metric("Annual Return", f"{metrics.get('Annual_Return', 0):.1%}")
                        st.metric("Volatility", f"{metrics.get('Volatility', 0):.1%}")
                
                with col3:
                    # Recommendation styling
                    rec_class = f"recommendation-{rec.recommendation.value.lower().replace('_', '-')}"
                    st.markdown(f'<div class="{rec_class}"><p class="recommendations-text">{rec.recommendation.value.replace("_", " ")}</p></div>', unsafe_allow_html=True)
                    
                    st.write("**💡 Key Reasons:**")
                    for reason in rec.reasons[:3]:
                        st.write(f"• {reason}")
                    
                    if rec.risk_factors:
                        st.write("**⚠️ Risk Factors:**")
                        for risk in rec.risk_factors[:2]:
                            st.write(f"• {risk}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    def render_interactive_charts(self):
        """Render interactive visualization section."""
        if not st.session_state.analysis_complete:
            return
        
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.header("📊 Interactive Market Analysis")
        
        risk_metrics = st.session_state.risk_metrics
        
        if risk_metrics.empty:
            st.warning("No data available for visualization.")
            return
        
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Risk-Return Profile", "📈 Performance Analysis", "🔍 Technical Dashboard", "⚖️ Risk Breakdown"])
        
        with tab1:
            self.render_risk_return_chart(risk_metrics)
        
        with tab2:
            self.render_performance_charts(risk_metrics)
        
        with tab3:
            self.render_technical_dashboard(risk_metrics)
        
        with tab4:
            self.render_risk_breakdown(risk_metrics)
        
        st.markdown('</div>', unsafe_allow_html=True)

    def render_risk_return_chart(self, risk_metrics):
        """Render risk-return scatter plot."""
        fig = px.scatter(
            risk_metrics,
            x='Volatility',
            y='Annual_Return',
            size='Sharpe_Ratio',
            color='Sharpe_Ratio',
            hover_name=risk_metrics.index,
            title="Risk-Return Profile Analysis",
            labels={
                'Volatility': 'Annualized Volatility',
                'Annual_Return': 'Expected Annual Return',
                'index': 'Stock Symbol'
            },
            color_continuous_scale='RdYlGn',
            size_max=25
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode='closest'
        )
        
        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=risk_metrics['Volatility'].median(), line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Quadrant analysis
        st.subheader("📍 Quadrant Analysis")
        med_vol = risk_metrics['Volatility'].median()
        med_ret = risk_metrics['Annual_Return'].median()
        
        quadrants = {
            "🟢 High Return, Low Risk": risk_metrics[(risk_metrics['Annual_Return'] > med_ret) & (risk_metrics['Volatility'] < med_vol)],
            "🟡 High Return, High Risk": risk_metrics[(risk_metrics['Annual_Return'] > med_ret) & (risk_metrics['Volatility'] >= med_vol)],
            "🟠 Low Return, Low Risk": risk_metrics[(risk_metrics['Annual_Return'] <= med_ret) & (risk_metrics['Volatility'] < med_vol)],
            "🔴 Low Return, High Risk": risk_metrics[(risk_metrics['Annual_Return'] <= med_ret) & (risk_metrics['Volatility'] >= med_vol)]
        }
        
        for quad_name, quad_data in quadrants.items():
            if not quad_data.empty:
                st.write(f"**{quad_name}:** {', '.join(quad_data.index.tolist())}")

    def render_performance_charts(self, risk_metrics):
        """Render performance analysis charts."""
        col1, col2 = st.columns(2)
        
        with col1:
            # Sharpe Ratio comparison
            fig_sharpe = px.bar(
                risk_metrics,
                x=risk_metrics.index,
                y='Sharpe_Ratio',
                title="Sharpe Ratio Comparison",
                color='Sharpe_Ratio',
                color_continuous_scale='RdYlGn'
            )
            fig_sharpe.update_layout(height=400, xaxis_title="Stock", yaxis_title="Sharpe Ratio")
            st.plotly_chart(fig_sharpe, use_container_width=True)
        
        with col2:
            # Return distribution
            fig_returns = px.histogram(
                risk_metrics,
                x='Annual_Return',
                nbins=20,
                title="Annual Return Distribution",
                color_discrete_sequence=['#667eea']
            )
            fig_returns.update_layout(height=400)
            st.plotly_chart(fig_returns, use_container_width=True)

    def render_technical_dashboard(self, risk_metrics):
        """Render technical analysis dashboard."""
        # Technical indicators heatmap
        technical_cols = ['Price_vs_SMA20', 'Price_vs_SMA50', 'RSI', 'Momentum_1M', 'Momentum_3M']
        available_cols = [col for col in technical_cols if col in risk_metrics.columns]
        
        if available_cols:
            fig_heatmap = px.imshow(
                risk_metrics[available_cols].T,
                x=risk_metrics.index,
                y=available_cols,
                color_continuous_scale='RdYlGn',
                aspect='auto',
                title="Technical Analysis Heatmap"
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)

    def render_risk_breakdown(self, risk_metrics):
        """Render risk analysis breakdown."""
        col1, col2 = st.columns(2)
        
        with col1:
            # VaR comparison
            if 'VaR_5%_Historical' in risk_metrics.columns:
                fig_var = px.bar(
                    risk_metrics,
                    x=risk_metrics.index,
                    y='VaR_5%_Historical',
                    title="Value at Risk (5% Level)",
                    color='VaR_5%_Historical',
                    color_continuous_scale='RdYlBu_r'
                )
                fig_var.update_layout(height=400)
                st.plotly_chart(fig_var, use_container_width=True)
        
        with col2:
            # Maximum Drawdown
            if 'Max_Drawdown' in risk_metrics.columns:
                fig_dd = px.bar(
                    risk_metrics,
                    x=risk_metrics.index,
                    y='Max_Drawdown',
                    title="Maximum Drawdown Analysis",
                    color='Max_Drawdown',
                    color_continuous_scale='RdYlBu_r'
                )
                fig_dd.update_layout(height=400)
                st.plotly_chart(fig_dd, use_container_width=True)

    def render_data_table(self):
        """Render detailed data table."""
        if not st.session_state.analysis_complete:
            return
        
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.header("📋 Detailed Analysis Table")
        
        risk_metrics = st.session_state.risk_metrics
        
        if not risk_metrics.empty:
            # Format the display dataframe
            display_df = risk_metrics.copy()
            
            # Format numeric columns
            numeric_cols = display_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if 'Price' in col:
                    display_df[col] = display_df[col].map('${:.2f}'.format)
                elif any(keyword in col for keyword in ['Return', 'Volatility', 'Drawdown', 'VaR']):
                    display_df[col] = display_df[col].map('{:.2%}'.format)
                elif col in ['Sharpe_Ratio', 'Sortino_Ratio', 'Beta', 'Alpha']:
                    display_df[col] = display_df[col].map('{:.3f}'.format)
                elif 'RSI' in col:
                    display_df[col] = display_df[col].map('{:.1f}'.format)
            
            # Select key columns for display
            key_columns = [
                'Current_Price', 'Annual_Return', 'Volatility', 'Sharpe_Ratio', 
                'Max_Drawdown', 'VaR_5%_Historical', 'Beta', 'RSI'
            ]
            available_columns = [col for col in key_columns if col in display_df.columns]
            
            st.dataframe(
                display_df[available_columns],
                use_container_width=True,
                height=400
            )
            
            # Summary statistics
            st.subheader("📊 Portfolio Summary Statistics")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.metric("Average Sharpe Ratio", f"{risk_metrics['Sharpe_Ratio'].mean():.2f}")
                st.metric("Portfolio Volatility", f"{risk_metrics['Volatility'].mean() * 100:.1f}%")
            
            with summary_col2:
                st.metric("Best Performer", risk_metrics.loc[risk_metrics['Annual_Return'].idxmax()].name)
                st.metric("Highest Sharpe", risk_metrics.loc[risk_metrics['Sharpe_Ratio'].idxmax()].name)
            
            with summary_col3:
                st.metric("Lowest Risk", risk_metrics.loc[risk_metrics['Volatility'].idxmin()].name)
                st.metric("Most Oversold", risk_metrics.loc[risk_metrics.get('RSI', pd.Series([50]*len(risk_metrics))).idxmin()].name)
        
        st.markdown('</div>', unsafe_allow_html=True)

    def generate_export_data(self):
        """Generate and provide export data."""
        if not st.session_state.analysis_complete:
            return
        
        # Create comprehensive export
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Risk metrics CSV
        risk_csv = st.session_state.risk_metrics.to_csv()
        st.download_button(
            label="📊 Download Risk Analysis (CSV)",
            data=risk_csv,
            file_name=f"ion_risk_analysis_{timestamp}.csv",
            mime="text/csv"
        )
        
        # Recommendations CSV
        if st.session_state.recommendations:
            rec_data = []
            for rec in st.session_state.recommendations:
                rec_data.append({
                    'Ticker': rec.ticker,
                    'Recommendation': rec.recommendation.value,
                    'Score': rec.score,
                    'Confidence': rec.confidence,
                    'Current_Price': rec.current_price,
                    'Target_Price': rec.target_price,
                    'Reasons': ' | '.join(rec.reasons),
                    'Risk_Factors': ' | '.join(rec.risk_factors)
                })
            
            rec_df = pd.DataFrame(rec_data)
            rec_csv = rec_df.to_csv(index=False)
            
            st.download_button(
                label="🎯 Download Recommendations (CSV)",
                data=rec_csv,
                file_name=f"ion_recommendations_{timestamp}.csv",
                mime="text/csv"
            )

    def get_recommendation_color(self, rec_type):
        """Get color for recommendation type."""
        colors = {
            'STRONG_BUY': 'linear-gradient(135deg, #00c851 0%, #007e33 100%)',
            'BUY': 'linear-gradient(135deg, #28a745 0%, #20c997 100%)',
            'HOLD': 'linear-gradient(135deg, #ffc107 0%, #fd7e14 100%)',
            'SELL': 'linear-gradient(135deg, #dc3545 0%, #c82333 100%)',
            'STRONG_SELL': 'linear-gradient(135deg, #721c24 0%, #dc3545 100%)'
        }
        return colors.get(rec_type, '#6c757d')

    def get_recommendation_emoji(self, rec_type):
        """Get emoji for recommendation type."""
        emojis = {
            'STRONG_BUY': '🚀',
            'BUY': '📈',
            'HOLD': '⏸️',
            'SELL': '📉',
            'STRONG_SELL': '⚠️'
        }
        return emojis.get(rec_type, '📊')

    def render_footer(self):
        """Render application footer."""
        st.markdown("---")
        st.markdown("""
        <div class="footer">
            <p><strong>ION Financial Risk Analyzer</strong> | Built with Streamlit | Professional Investment Analysis</p>
            <p>Demonstrating institutional-grade financial technology capabilities for ION Group</p>
            <p style="font-size: 0.8em; color: #adb5bd;">
                This application showcases advanced financial analysis, machine learning, and risk management techniques
                suitable for professional trading and investment management platforms.
            </p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application entry point."""
    dashboard = FinancialDashboard()
    
    # Render main components
    dashboard.render_header()
    
    # Sidebar
    selected_stocks, time_period, risk_tolerance, investment_style, min_sharpe, max_volatility = dashboard.render_sidebar()
    
    # Main content area
    dashboard.render_dashboard_overview()
    
    if st.session_state.analysis_complete:
        dashboard.render_recommendations_section()
        dashboard.render_interactive_charts()
        dashboard.render_data_table()
    
    # Footer
    dashboard.render_footer()

if __name__ == "__main__":
    main()