import os
import argparse
from datetime import datetime
from data_loader import load_uidai_data
from news_fetcher import fetch_all_news
from analysis import MigrationAnalyzer

def generate_report(aadhar_df, news_df, merged_data, model_info, ranking, trend_data, output_path):
    """
    Generates a detailed markdown report of the migration analysis.
    """
    report_lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Header
    report_lines.append("# Aadhar Migration Prediction Report\n")
    report_lines.append(f"**Generated:** {timestamp}\n")
    report_lines.append("\n---\n")
    
    # Executive Summary
    report_lines.append("## Executive Summary\n\n")
    if not ranking.empty:
        top_state = ranking.iloc[0]
        report_lines.append(f"Based on **monthly trend analysis** of Aadhar demographic updates, ")
        report_lines.append(f"**{top_state['state']}** shows the highest predicted migration activity ")
        report_lines.append(f"with a probability of **{top_state['migration_probability']:.2f}%**.\n\n")
        report_lines.append(f"- **Trend Direction:** {top_state['trend_direction']}\n")
        report_lines.append(f"- **News Sentiment:** {top_state['news_sentiment']}\n")
        report_lines.append(f"- **News Adjustment Factor:** {top_state['news_adjustment']:.2f}x\n")
    report_lines.append("\n---\n")
    
    # Data Overview
    report_lines.append("## 1. Data Overview\n\n")
    
    report_lines.append("### Aadhar Demographic Data\n")
    report_lines.append(f"- **Total Records:** {len(aadhar_df):,}\n")
    report_lines.append(f"- **States Analyzed:** {len(aadhar_df['state'].unique())}\n")
    report_lines.append(f"- **Date Range:** {aadhar_df['date'].min()} to {aadhar_df['date'].max()}\n")
    
    report_lines.append("\n### News Data (Add-on Factor)\n")
    report_lines.append(f"- **Total Articles Fetched:** {len(news_df):,}\n")
    report_lines.append(f"- **Search Keywords:** jobs, migration, hiring, industrial growth, layoffs\n")
    
    # Monthly Trend Analysis
    report_lines.append("\n---\n")
    report_lines.append("## 2. Monthly Trend Analysis (Primary Prediction Method)\n\n")
    report_lines.append("The prediction is primarily based on month-over-month (MoM) trends in Aadhar updates.\n\n")
    report_lines.append("| State | Avg Monthly Updates | Avg Growth Rate | Trend | Latest Updates |\n")
    report_lines.append("|-------|---------------------|-----------------|-------|----------------|\n")
    
    if trend_data is not None and not trend_data.empty:
        for _, row in trend_data.iterrows():
            growth_str = f"{row['avg_growth_rate']:.1f}%" if not pd.isna(row['avg_growth_rate']) else "N/A"
            report_lines.append(f"| {row['state']} | {row['avg_monthly_updates']:,.0f} | {growth_str} | {row['trend_direction']} | {row['latest_updates']:,.0f} |\n")
    
    # News Adjustment
    report_lines.append("\n---\n")
    report_lines.append("## 3. News Sentiment Adjustment (Secondary Add-on)\n\n")
    report_lines.append("News activity adjusts the base trend prediction:\n")
    report_lines.append("- **Positive News** (hiring, industrial growth) → Increases prediction\n")
    report_lines.append("- **Negative News** (layoffs) → Decreases prediction\n")
    report_lines.append("- **Adjustment Factor Range:** 0.5x to 1.5x\n\n")
    
    report_lines.append("| State | News Sentiment | Adjustment Factor |\n")
    report_lines.append("|-------|----------------|-------------------|\n")
    
    if not ranking.empty:
        for _, row in ranking.iterrows():
            report_lines.append(f"| {row['state']} | {row['news_sentiment']} | {row['news_adjustment']:.2f}x |\n")
    
    # Final Results
    report_lines.append("\n---\n")
    report_lines.append("## 4. Migration Prediction Results\n\n")
    report_lines.append("### Final Ranking (Trend + News Adjustment)\n\n")
    report_lines.append("| Rank | State | Base Prediction | News Adj | Final Prediction | Probability |\n")
    report_lines.append("|------|-------|-----------------|----------|------------------|-------------|\n")
    
    if not ranking.empty:
        for i, row in ranking.iterrows():
            report_lines.append(f"| {i+1} | {row['state']} | {row['base_prediction']:,.0f} | {row['news_adjustment']:.2f}x | {row['final_prediction']:,.0f} | {row['migration_probability']:.2f}% |\n")
    
    # Reasoning
    report_lines.append("\n---\n")
    report_lines.append("## 5. Methodology & Reasoning\n\n")
    report_lines.append("### How Predictions Are Made\n\n")
    report_lines.append("1. **Monthly Trend Analysis (Primary)**\n")
    report_lines.append("   - Calculate month-over-month changes in Aadhar updates\n")
    report_lines.append("   - Identify trend direction (increasing/decreasing)\n")
    report_lines.append("   - Extrapolate next month's expected updates using linear trend\n\n")
    report_lines.append("2. **News Adjustment (Secondary)**\n")
    report_lines.append("   - Analyze current news sentiment for each state\n")
    report_lines.append("   - Apply adjustment factor (0.5x to 1.5x) to base prediction\n")
    report_lines.append("   - Positive economic news boosts prediction, negative news reduces it\n\n")
    report_lines.append("3. **Probability Calculation**\n")
    report_lines.append("   - `Probability = (State's Final Prediction / Total Predictions) × 100%`\n")
    
    # Conclusion
    report_lines.append("\n---\n")
    report_lines.append("## 6. Conclusion & Recommendations\n\n")
    if not ranking.empty:
        top3 = ranking.head(3)
        report_lines.append("### Top Migration Hotspots\n\n")
        for i, row in top3.iterrows():
            report_lines.append(f"{i+1}. **{row['state']}** ({row['migration_probability']:.2f}%) - {row['trend_direction']} trend, {row['news_sentiment']} news\n")
        
        report_lines.append("\n### Recommendations\n\n")
        report_lines.append("- Monitor Aadhar update centers in high-probability states\n")
        report_lines.append("- Allocate resources proportionally to predicted probabilities\n")
        report_lines.append("- Track monthly trends for early warning of migration shifts\n")
        report_lines.append("- Update predictions weekly as new news data becomes available\n")
    
    report_lines.append("\n---\n")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Aadhar Migration Prediction System")
    parser.add_argument("--data_dir", default=os.path.join(os.getcwd(), "Data"), help="Directory containing Aadhar CSV files")
    parser.add_argument("--test_run", action="store_true", help="Run with a small subset or synthetic news")
    parser.add_argument("--output", default="migration_report.md", help="Output report filename")
    args = parser.parse_args()
    
    print("=== Starting Aadhar Migration Prediction System ===")
    
    # 1. Load Aadhar Data
    print("\n[Step 1] Loading Aadhar Data...")
    aadhar_df = load_uidai_data(args.data_dir)
    print(f"Loaded {len(aadhar_df)} Aadhar update records.")
    
    if aadhar_df.empty:
        print("CRITICAL: No Aadhar data found. Exiting.")
        return
        
    states = aadhar_df['state'].unique().tolist()
    print(f"Detected States: {states}")
    
    # 2. Fetch News Data
    print("\n[Step 2] Fetching News Data...")
    if args.test_run:
        print("Test run mode: Fetching news for only the first state to save time.")
        states = states[:1]
        
    keywords = ['jobs', 'migration', 'hiring', 'industrial growth', 'layoffs']
    news_df = fetch_all_news(states, queries=keywords)
    print(f"Fetched {len(news_df)} news articles.")
    
    # 3. Prepare Data
    print("\n[Step 3] Preparing Data...")
    analyzer = MigrationAnalyzer()
    merged_data = analyzer.prepare_data(aadhar_df, news_df)
    
    if merged_data.empty:
        print("Analysis failed: No data after merging.")
        return
        
    print(f"Prepared {len(merged_data)} state-month data points.")
    
    # 4. Monthly Comparison
    print("\n[Step 4] Monthly Comparison Analysis...")
    monthly_comparison, pct_change = analyzer.get_monthly_comparison(merged_data)
    
    print("\nMonthly Updates by State (Absolute Values):")
    print("-" * 80)
    print(monthly_comparison.to_string())
    print("-" * 80)
    
    print("\nMonth-over-Month % Change:")
    print("-" * 80)
    # Format percentage display
    pct_display = pct_change.map(lambda x: f"{x:+.1f}%" if x != 0 else "---")
    print(pct_display.to_string())
    print("-" * 80)
    
    # 5. Train News Correlation Model (for reference)
    print("\n[Step 5] Analyzing News Correlations...")
    model_info = analyzer.train_model(merged_data)
    
    # 6. Predict using Monthly Trends + News Adjustment
    print("\n[Step 6] Predicting Migration (Monthly Trends + News Adjustment)...")
    ranking, trend_data = analyzer.predict_next_cycle(merged_data, news_df)
    
    if not ranking.empty:
        print("\n=== Predicted Migration Hotspots (Ranked) ===")
        print(f"{'Rank':<5} | {'State':<18} | {'Trend':<12} | {'News':<10} | {'Base Pred':<12} | {'Adj':<6} | {'Prob':<8}")
        print("-" * 85)
        
        for i, row in ranking.iterrows():
            print(f"{i+1:<5} | {row['state']:<18} | {row['trend_direction']:<12} | {row['news_sentiment']:<10} | {row['base_prediction']:>10,.0f} | {row['news_adjustment']:.2f}x | {row['migration_probability']:.2f}%")
            
        print("-" * 85)
        
        print("\nTop 3 States likely to see Migration Activity:")
        for i, row in ranking.head(3).iterrows():
            print(f"{i+1}. {row['state']} ({row['migration_probability']:.2f}%) - {row['trend_direction']} trend, {row['news_sentiment']} news")
    else:
        print("Could not generate ranking.")
    
    # 7. Generate Report
    print("\n[Step 7] Generating Detailed Report...")
    import pandas as pd
    report_path = generate_report(aadhar_df, news_df, merged_data, model_info, ranking, trend_data, args.output)
    print(f"Report saved to: {report_path}")
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    import pandas as pd
    main()
