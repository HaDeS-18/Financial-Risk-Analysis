#!/usr/bin/env python3
"""
Financial Risk Analyzer - Standalone Analysis Script
Professional-grade financial analysis for command-line execution
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project directories to path
sys.path.append('src')
sys.path.append('config')

try:
    from data_ingestion import FinancialDataIngestion
    from risk_analyzer import AdvancedRiskAnalyzer
    from ml_predictor import AdvancedMLPredictor
    from recommendation_engine import AdvancedRecommendationEngine
    from settings import STOCK_UNIVERSES
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure all source files are in the correct directories:")
    print("- src/data_ingestion.py")
    print("- src/risk_analyzer.py") 
    print("- src/ml_predictor.py")
    print("- src/recommendation_engine.py")
    print("- config/settings.py")
    sys.exit(1)

def print_banner():
    """Print application banner."""
    print("="*80)
    print("🏦 FINANCIAL RISK ANALYZER")
    print("Professional Investment Decision Support System")
    print("="*80)
    print()

def print_section_header(title):
    """Print section header."""
    print(f"\n{'='*20} {title} {'='*20}")

def analyze_portfolio(stocks, period='2y', investment_style='Balanced', risk_tolerance='Moderate'):
    """
    Run complete financial analysis for a portfolio of stocks.
    
    Args:
        stocks: List of stock symbols
        period: Analysis period (1y, 2y, 3y, 5y)
        investment_style: Investment approach
        risk_tolerance: Risk level
    
    Returns:
        Tuple of (risk_metrics, recommendations, ml_results)
    """
    print_section_header("INITIALIZING ANALYSIS COMPONENTS")
    
    # Initialize analysis components
    data_ingester = FinancialDataIngestion()
    risk_analyzer = AdvancedRiskAnalyzer()
    ml_predictor = AdvancedMLPredictor()
    rec_engine = AdvancedRecommendationEngine(investment_style, risk_tolerance)
    
    print("✅ All components initialized successfully")
    
    print_section_header("DATA INGESTION")
    print(f"📊 Fetching data for {len(stocks)} stocks over {period} period...")
    
    # Fetch stock data
    stock_data = data_ingester.fetch_stock_data(stocks, period)
    
    if not stock_data:
        print("❌ Failed to fetch any stock data")
        return None, None, None
    
    print(f"✅ Successfully fetched data for {len(stock_data)} stocks")
    
    # Display data summary
    print("\n📈 Data Summary:")
    for ticker, data in stock_data.items():
        print(f"  {ticker}: {len(data)} days | Latest: ${data['Close'].iloc[-1]:.2f}")
    
    print_section_header("RISK ANALYSIS")
    print("⚖️ Calculating comprehensive risk metrics...")
    
    # Calculate risk metrics
    risk_metrics = risk_analyzer.calculate_comprehensive_metrics(stock_data)
    
    if risk_metrics.empty:
        print("❌ No risk metrics calculated")
        return None, None, None
    
    print(f"✅ Risk analysis completed for {len(risk_metrics)} securities")
    
    # Display risk summary
    print("\n📊 Risk Metrics Summary:")
    print(f"{'Ticker':<8} {'Sharpe':<8} {'Return':<10} {'Vol':<8} {'Drawdown':<10}")
    print("-" * 50)
    
    for ticker in risk_metrics.index:
        metrics = risk_metrics.loc[ticker]
        print(f"{ticker:<8} {metrics.get('Sharpe_Ratio', 0):<8.2f} "
              f"{metrics.get('Annual_Return', 0):<10.1%} "
              f"{metrics.get('Volatility', 0):<8.1%} "
              f"{metrics.get('Max_Drawdown', 0):<10.1%}")
    
    print_section_header("MACHINE LEARNING PREDICTIONS")
    print("🤖 Training ML models and generating predictions...")
    
    ml_results = {}
    for ticker in list(stock_data.keys())[:3]:  # Limit to 3 stocks for demo
        print(f"Training models for {ticker}...")
        
        try:
            # Engineer features
            featured_data = ml_predictor.engineer_features(stock_data[ticker], ticker)
            featured_data = ml_predictor.create_target_variables(featured_data)
            
            # Train models
            results = ml_predictor.train_ensemble_models(featured_data, ticker, target_horizon=5)
            
            if results:
                ml_results[ticker] = results
                best_model = min(results.keys(), key=lambda k: results[k]['performance']['mse'])
                r2_score = results[best_model]['performance']['r2']
                print(f"  ✅ {ticker}: Best model = {best_model} (R² = {r2_score:.3f})")
            else:
                print(f"  ⚠️ {ticker}: Model training failed")
                
        except Exception as e:
            print(f"  ❌ {ticker}: Error during ML training - {str(e)}")
    
    print(f"✅ ML analysis completed for {len(ml_results)} stocks")
    
    print_section_header("INVESTMENT RECOMMENDATIONS")
    print("🎯 Generating investment recommendations...")
    
    # Generate recommendations
    recommendations = rec_engine.generate_recommendations(risk_metrics)
    
    if not recommendations:
        print("❌ No recommendations generated")
        return risk_metrics, None, ml_results
    
    print(f"✅ Generated {len(recommendations)} investment recommendations")
    
    # Display top recommendations
    print(f"\n🏆 Top Investment Recommendations:")
    print(f"{'Rank':<4} {'Ticker':<8} {'Recommendation':<12} {'Score':<8} {'Confidence':<10} {'Target':<10}")
    print("-" * 70)
    
    for i, rec in enumerate(recommendations[:10], 1):
        target_str = f"${rec.target_price:.2f}" if rec.target_price else "N/A"
        print(f"{i:<4} {rec.ticker:<8} {rec.recommendation.value.replace('_', ' '):<12} "
              f"{rec.score:<8.1f} {rec.confidence:<10.1%} {target_str:<10}")
    
    return risk_metrics, recommendations, ml_results

def display_detailed_analysis(risk_metrics, recommendations, ml_results):
    """Display detailed analysis results."""
    
    print_section_header("DETAILED RECOMMENDATION ANALYSIS")
    
    # Group recommendations by type
    rec_by_type = {}
    for rec in recommendations:
        rec_type = rec.recommendation.value
        if rec_type not in rec_by_type:
            rec_by_type[rec_type] = []
        rec_by_type[rec_type].append(rec)
    
    # Display recommendations by category
    for rec_type, recs in rec_by_type.items():
        print(f"\n{rec_type.replace('_', ' ')} ({len(recs)} stocks):")
        for rec in recs:
            print(f"  📊 {rec.ticker} (Score: {rec.score:.1f})")
            if rec.reasons:
                print(f"     💡 {rec.reasons[0]}")
            if rec.risk_factors:
                print(f"     ⚠️  {rec.risk_factors[0]}")
    
    print_section_header("PORTFOLIO STATISTICS")
    
    # Portfolio-level statistics
    total_stocks = len(risk_metrics)
    avg_sharpe = risk_metrics['Sharpe_Ratio'].mean()
    avg_return = risk_metrics['Annual_Return'].mean()
    avg_vol = risk_metrics['Volatility'].mean()
    
    best_sharpe = risk_metrics.loc[risk_metrics['Sharpe_Ratio'].idxmax()]
    worst_drawdown = risk_metrics.loc[risk_metrics['Max_Drawdown'].idxmin()]
    
    print(f"📊 Portfolio Overview:")
    print(f"   • Total Securities Analyzed: {total_stocks}")
    print(f"   • Average Sharpe Ratio: {avg_sharpe:.2f}")
    print(f"   • Average Annual Return: {avg_return:.1%}")
    print(f"   • Average Volatility: {avg_vol:.1%}")
    print(f"   • Best Risk-Adjusted Return: {best_sharpe.name} (Sharpe: {best_sharpe['Sharpe_Ratio']:.2f})")
    print(f"   • Largest Drawdown: {worst_drawdown.name} ({worst_drawdown['Max_Drawdown']:.1%})")
    
    if ml_results:
        print_section_header("MACHINE LEARNING INSIGHTS")
        print("🤖 ML Model Performance Summary:")
        
        for ticker, results in ml_results.items():
            print(f"\n{ticker} Model Performance:")
            for model_name, result in results.items():
                if model_name != 'Ensemble' and 'performance' in result:
                    perf = result['performance']
                    print(f"   • {model_name}: R² = {perf['r2']:.3f}, RMSE = {perf['rmse']:.4f}")

def save_results(risk_metrics, recommendations, ml_results):
    """Save analysis results to files."""
    print_section_header("SAVING RESULTS")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Save risk metrics
        risk_file = f"data/exports/risk_analysis_{timestamp}.csv"
        risk_metrics.to_csv(risk_file)
        print(f"✅ Risk metrics saved to: {risk_file}")
        
        # Save recommendations
        if recommendations:
            rec_data = []
            for rec in recommendations:
                rec_data.append({
                    'Ticker': rec.ticker,
                    'Recommendation': rec.recommendation.value,
                    'Score': rec.score,
                    'Confidence': rec.confidence,
                    'Current_Price': rec.current_price,
                    'Target_Price': rec.target_price,
                    'Reasons': ' | '.join(rec.reasons),
                    'Risk_Factors': ' | '.join(rec.risk_factors),
                    'Time_Horizon': rec.time_horizon
                })
            
            rec_df = pd.DataFrame(rec_data)
            rec_file = f"data/exports/recommendations_{timestamp}.csv"
            rec_df.to_csv(rec_file, index=False)
            print(f"✅ Recommendations saved to: {rec_file}")
        
        # Save ML results summary
        if ml_results:
            ml_summary = []
            for ticker, results in ml_results.items():
                for model_name, result in results.items():
                    if 'performance' in result:
                        perf = result['performance']
                        ml_summary.append({
                            'Ticker': ticker,
                            'Model': model_name,
                            'R2_Score': perf['r2'],
                            'RMSE': perf['rmse'],
                            'MAE': perf['mae']
                        })
            
            if ml_summary:
                ml_df = pd.DataFrame(ml_summary)
                ml_file = f"data/exports/ml_results_{timestamp}.csv"
                ml_df.to_csv(ml_file, index=False)
                print(f"✅ ML results saved to: {ml_file}")
        
        # Generate executive summary
        summary_file = f"data/exports/executive_summary_{timestamp}.txt"
        generate_executive_summary(risk_metrics, recommendations, ml_results, summary_file)
        print(f"✅ Executive summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"❌ Error saving results: {str(e)}")

def generate_executive_summary(risk_metrics, recommendations, ml_results, filename):
    """Generate and save executive summary report."""
    
    with open(filename, 'w') as f:
        f.write("FINANCIAL RISK ANALYZER - EXECUTIVE SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}\n\n")
        
        f.write("ANALYSIS OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Securities Analyzed: {len(risk_metrics)}\n")
        f.write(f"Recommendations Generated: {len(recommendations) if recommendations else 0}\n")
        f.write(f"ML Models Trained: {len(ml_results) if ml_results else 0}\n\n")
        
        if recommendations:
            buy_recs = [r for r in recommendations if r.recommendation.value in ['STRONG_BUY', 'BUY']]
            sell_recs = [r for r in recommendations if r.recommendation.value in ['STRONG_SELL', 'SELL']]
            
            f.write("KEY FINDINGS\n")
            f.write("-" * 15 + "\n")
            f.write(f"Buy Recommendations: {len(buy_recs)} securities\n")
            f.write(f"Sell Recommendations: {len(sell_recs)} securities\n")
            f.write(f"Average Portfolio Sharpe Ratio: {risk_metrics['Sharpe_Ratio'].mean():.2f}\n")
            f.write(f"Average Volatility: {risk_metrics['Volatility'].mean():.1%}\n\n")
            
            f.write("TOP 5 RECOMMENDATIONS\n")
            f.write("-" * 25 + "\n")
            for i, rec in enumerate(recommendations[:5], 1):
                f.write(f"{i}. {rec.ticker} - {rec.recommendation.value.replace('_', ' ')}\n")
                f.write(f"   Score: {rec.score:.1f}/100 | Confidence: {rec.confidence:.1%}\n")
                if rec.reasons:
                    f.write(f"   Reason: {rec.reasons[0]}\n")
                f.write("\n")
        
        f.write("RISK ANALYSIS SUMMARY\n")
        f.write("-" * 25 + "\n")
        f.write(f"Best Sharpe Ratio: {risk_metrics.loc[risk_metrics['Sharpe_Ratio'].idxmax()].name} ")
        f.write(f"({risk_metrics['Sharpe_Ratio'].max():.2f})\n")
        f.write(f"Lowest Volatility: {risk_metrics.loc[risk_metrics['Volatility'].idxmin()].name} ")
        f.write(f"({risk_metrics['Volatility'].min():.1%})\n")
        f.write(f"Largest Drawdown: {risk_metrics.loc[risk_metrics['Max_Drawdown'].idxmin()].name} ")
        f.write(f"({risk_metrics['Max_Drawdown'].min():.1%})\n\n")
        
        if ml_results:
            f.write("MACHINE LEARNING INSIGHTS\n")
            f.write("-" * 30 + "\n")
            f.write("Model Performance (R² Scores):\n")
            for ticker, results in ml_results.items():
                best_r2 = 0
                best_model = "None"
                for model_name, result in results.items():
                    if 'performance' in result and result['performance']['r2'] > best_r2:
                        best_r2 = result['performance']['r2']
                        best_model = model_name
                f.write(f"  {ticker}: {best_model} (R² = {best_r2:.3f})\n")
        
        f.write("\nDISCLAIMER\n")
        f.write("-" * 15 + "\n")
        f.write("This analysis is for informational purposes only and does not constitute\n")
        f.write("investment advice. Past performance does not guarantee future results.\n")
        f.write("Please consult with a qualified financial advisor before making investment decisions.\n")

def main():
    """Main application entry point."""
    print_banner()
    
    # Configuration
    default_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'JPM', 'BAC', 'NVDA', 'AMZN']
    
    print("🎯 CONFIGURATION OPTIONS")
    print("-" * 30)
    
    # Stock universe selection
    print("\nAvailable Stock Universes:")
    for i, (name, info) in enumerate(STOCK_UNIVERSES.items(), 1):
        print(f"  {i}. {name}: {info['description']}")
    
    try:
        universe_choice = input(f"\nSelect universe (1-{len(STOCK_UNIVERSES)}) or press Enter for default: ").strip()
        
        if universe_choice:
            universe_names = list(STOCK_UNIVERSES.keys())
            selected_universe = universe_names[int(universe_choice) - 1]
            stocks = STOCK_UNIVERSES[selected_universe]['stocks'][:8]  # Limit to 8 stocks
            print(f"✅ Selected: {selected_universe}")
        else:
            stocks = default_stocks
            print("✅ Using default technology stocks")
        
        print(f"📊 Analyzing stocks: {', '.join(stocks)}")
        
        # Analysis parameters
        period = input("\nAnalysis period (1y/2y/3y/5y) [default: 2y]: ").strip() or '2y'
        investment_style = input("Investment style (Value/Growth/Momentum/Quality/Balanced) [default: Balanced]: ").strip() or 'Balanced'
        risk_tolerance = input("Risk tolerance (Conservative/Moderate/Aggressive) [default: Moderate]: ").strip() or 'Moderate'
        
        print(f"\n⚙️ Configuration:")
        print(f"   • Period: {period}")
        print(f"   • Investment Style: {investment_style}")
        print(f"   • Risk Tolerance: {risk_tolerance}")
        
        # Confirm execution
        proceed = input("\n🚀 Run analysis? (y/N): ").strip().lower()
        if proceed != 'y':
            print("Analysis cancelled.")
            return
        
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE FINANCIAL ANALYSIS")
        print("="*80)
        
        # Run analysis
        risk_metrics, recommendations, ml_results = analyze_portfolio(
            stocks, period, investment_style, risk_tolerance
        )
        
        if risk_metrics is not None:
            # Display detailed results
            display_detailed_analysis(risk_metrics, recommendations, ml_results)
            
            # Save results
            save_results(risk_metrics, recommendations, ml_results)
            
            print_section_header("ANALYSIS COMPLETE")
            print("✅ Comprehensive financial analysis completed successfully!")
            print("\n📁 Results saved to data/exports/ directory")
            print("📊 For interactive analysis, run: streamlit run streamlit_app.py")
            
        else:
            print("❌ Analysis failed. Please check your configuration and try again.")
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Analysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("Please check your configuration and try again.")

def run_demo_analysis():
    """Run a quick demo analysis with predefined settings."""
    print_banner()
    print("🎬 RUNNING DEMO ANALYSIS")
    print("Using predefined configuration for quick demonstration\n")
    
    # Demo configuration
    demo_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    print(f"📊 Demo Stocks: {', '.join(demo_stocks)}")
    print("⚙️ Configuration: 1-year period, Balanced style, Moderate risk")
    print("\n" + "="*60)
    
    # Run analysis
    risk_metrics, recommendations, ml_results = analyze_portfolio(
        demo_stocks, '1y', 'Balanced', 'Moderate'
    )
    
    if risk_metrics is not None:
        display_detailed_analysis(risk_metrics, recommendations, ml_results)
        save_results(risk_metrics, recommendations, ml_results)
        
        print_section_header("DEMO COMPLETE")
        print("✅ Demo analysis completed!")
        print("📁 Results saved to data/exports/ directory")
    else:
        print("❌ Demo analysis failed.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Financial Risk Analyzer")
    parser.add_argument('--demo', action='store_true', help='Run demo analysis with predefined settings')
    parser.add_argument('--stocks', nargs='+', help='List of stock symbols to analyze')
    parser.add_argument('--period', default='2y', choices=['1y', '2y', '3y', '5y'], help='Analysis period')
    parser.add_argument('--style', default='Balanced', choices=['Value', 'Growth', 'Momentum', 'Quality', 'Balanced'], help='Investment style')
    parser.add_argument('--risk', default='Moderate', choices=['Conservative', 'Moderate', 'Aggressive'], help='Risk tolerance')
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo_analysis()
    elif args.stocks:
        print_banner()
        print("🚀 RUNNING CUSTOM ANALYSIS")
        print(f"📊 Stocks: {', '.join(args.stocks)}")
        print(f"⚙️ Period: {args.period} | Style: {args.style} | Risk: {args.risk}")
        print("\n" + "="*60)
        
        risk_metrics, recommendations, ml_results = analyze_portfolio(
            args.stocks, args.period, args.style, args.risk
        )
        
        if risk_metrics is not None:
            display_detailed_analysis(risk_metrics, recommendations, ml_results)
            save_results(risk_metrics, recommendations, ml_results)
            print_section_header("ANALYSIS COMPLETE")
            print("✅ Custom analysis completed!")
        else:
            print("❌ Analysis failed.")
    else:
        main()