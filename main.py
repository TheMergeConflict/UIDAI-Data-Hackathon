import os
import argparse
from datetime import datetime
from data_loader import load_uidai_data
from news_fetcher import fetch_all_news
from analysis import MigrationAnalyzer

def generate_report(aadhar_df, news_df, merged_data, model_info, ranking, output_path):
    """
    Generates a detailed markdown report of the migration analysis.
    """
    report_lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Header
    report_lines.append("# Aadhar Migration Prediction Report")
    report_lines.append(f"\n**Generated:** {timestamp}\n")
    report_lines.append("---\n")
    
    # Executive Summary
    report_lines.append("## Executive Summary\n")
    if not ranking.empty:
        top_state = ranking.iloc[0]
        report_lines.append(f"Based on current news activity and historical Aadhar demographic update patterns, ")
        report_lines.append(f"**{top_state['state']}** has the highest predicted migration probability at **{top_state['migration_probability']:.2f}%**.\n")
    report_lines.append("\n---\n")
    
    # Data Overview
    report_lines.append("## 1. Data Overview\n")
    report_lines.append("### Aadhar Demographic Data\n")
    report_lines.append(f"- **Total Records:** {len(aadhar_df):,}\n")
    report_lines.append(f"- **States Analyzed:** {len(aadhar_df['state'].unique())}\n")
    report_lines.append(f"- **Date Range:** {aadhar_df['date'].min()} to {aadhar_df['date'].max()}\n")
    
    report_lines.append("\n### News Data\n")
    report_lines.append(f"- **Total Articles Fetched:** {len(news_df):,}\n")
    report_lines.append(f"- **Search Keywords:** jobs, migration, hiring, industrial growth, layoffs\n")
    
    # State-wise Summary
    report_lines.append("\n---\n")
    report_lines.append("## 2. State-wise Data Summary\n")
    report_lines.append("| State | Months | Avg Monthly Updates | Total News Articles |\n")
    report_lines.append("|-------|--------|---------------------|---------------------|\n")
    
    news_cols = [c for c in merged_data.columns if c.endswith('_news_count')]
    for state in merged_data['state'].unique():
        state_data = merged_data[merged_data['state'] == state]
        months = len(state_data)
        avg_updates = state_data['total_updates'].mean()
        total_news = state_data[news_cols].sum().sum()
        report_lines.append(f"| {state} | {months} | {avg_updates:,.0f} | {total_news:,.0f} |\n")
    
    # Model Analysis
    report_lines.append("\n---\n")
    report_lines.append("## 3. Prediction Model Analysis\n")
    report_lines.append("### Model Details\n")
    report_lines.append(f"- **Algorithm:** Custom Linear Regression (NumPy-based)\n")
    report_lines.append(f"- **Training Samples:** {model_info['train_samples']}\n")
    report_lines.append(f"- **Test Samples:** {model_info['test_samples']}\n")
    report_lines.append(f"- **Root Mean Square Error:** {model_info['rmse']:,.0f}\n")
    
    report_lines.append("\n### Key Factors Influencing Migration\n")
    report_lines.append("| Factor | Impact | Interpretation |\n")
    report_lines.append("|--------|--------|----------------|\n")
    
    for factor, info in model_info['factors'].items():
        report_lines.append(f"| {factor} | {info['impact']} | {info['interpretation']} |\n")
    
    report_lines.append(f"\n**Strongest Correlation:** {model_info['strongest_correlation']}\n")
    
    # Reasoning
    report_lines.append("\n### Reasoning Behind the Model\n")
    report_lines.append("The model uses news activity as a leading indicator of migration patterns. ")
    report_lines.append("The logic is:\n")
    report_lines.append("- **Hiring/Industrial Growth News** → Economic opportunities → Attracts migrants\n")
    report_lines.append("- **Jobs News** → May indicate competition/saturation → Mixed effect\n")
    report_lines.append("- **Layoffs News** → Economic distress → Discourages migration or triggers outflux\n")
    report_lines.append("- **Migration News** → Already happening events → Lagging indicator\n")
    
    # Final Results
    report_lines.append("\n---\n")
    report_lines.append("## 4. Migration Prediction Results\n")
    report_lines.append("### State Rankings by Migration Probability\n")
    report_lines.append("| Rank | State | Probability | Analysis |\n")
    report_lines.append("|------|-------|-------------|----------|\n")
    
    for i, row in ranking.iterrows():
        state = row['state']
        prob = row['migration_probability']
        if prob > 50:
            analysis = "High migration activity expected"
        elif prob > 20:
            analysis = "Moderate migration activity expected"
        elif prob > 5:
            analysis = "Low migration activity expected"
        else:
            analysis = "Minimal migration activity expected"
        report_lines.append(f"| {i+1} | {state} | {prob:.2f}% | {analysis} |\n")
    
    # Conclusion
    report_lines.append("\n---\n")
    report_lines.append("## 5. Conclusion & Recommendations\n")
    if not ranking.empty:
        top3 = ranking.head(3)
        report_lines.append("### Top Migration Hotspots\n")
        for i, row in top3.iterrows():
            report_lines.append(f"{i+1}. **{row['state']}** ({row['migration_probability']:.2f}%)\n")
        
        report_lines.append("\n### Recommendations\n")
        report_lines.append("- Monitor Aadhar update centers in high-probability states for increased activity\n")
        report_lines.append("- Allocate resources proportionally to predicted migration probabilities\n")
        report_lines.append("- Track news trends weekly to update predictions\n")
    
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
        
    # Get unique states for news fetching
    states = aadhar_df['state'].unique().tolist()
    print(f"Detected States: {states}")
    
    # 2. Fetch News Data
    print("\n[Step 2] Fetching News Data...")
    if args.test_run:
        print("Test run mode: Fetching news for only the first state to save time.")
        states = states[:1]
        
    # Keywords to search
    keywords = ['jobs', 'migration', 'hiring', 'industrial growth', 'layoffs']
    news_df = fetch_all_news(states, queries=keywords)
    print(f"Fetched {len(news_df)} news articles.")
    
    # 3. Analyze and Predict
    print("\n[Step 3] Running Analysis Engine...")
    analyzer = MigrationAnalyzer()
    merged_data = analyzer.prepare_data(aadhar_df, news_df)
    
    if merged_data.empty:
        print("Analysis failed: No data after merging.")
        return
        
    print(f"Prepared {len(merged_data)} state-month data points for modeling.")
    
    # Show state-wise summary
    print("\nState-wise Data Summary:")
    print(f"{'State':<20} | {'Months':<7} | {'Avg Updates':<12} | {'Total News':<12}")
    print("-" * 60)
    news_cols = [c for c in merged_data.columns if c.endswith('_news_count')]
    for state in merged_data['state'].unique():
        state_data = merged_data[merged_data['state'] == state]
        months = len(state_data)
        avg_updates = state_data['total_updates'].mean()
        total_news = state_data[news_cols].sum().sum()
        print(f"{state:<20} | {months:<7} | {avg_updates:>10,.0f} | {total_news:>10,.0f}")
    print("-" * 60)
    
    print("\n[Step 4] Training Prediction Model...")
    model_info = analyzer.train_model(merged_data)
    
    print("\n[Step 5] Forecasting & Ranking States...")
    # For prediction, we use the news_df we fetched (which represents "current/recent" news)
    ranking = analyzer.predict_next_cycle(news_df)
    
    if not ranking.empty:
        print("\n=== Predicted Migration Hotspots (Ranked) ===")
        # formatting header
        print(f"{'Rank':<5} | {'State':<20} | {'Probability':<15}")
        print("-" * 50)
        
        for i, row in ranking.iterrows():
            state_name = row['state']
            prob = row['migration_probability']
            print(f"{i+1:<5} | {state_name:<20} | {prob:.2f}%")
            
        print("-" * 50)
        
        print("\nTop 3 States likely to see Migration Influx/Outflux:")
        for i, row in ranking.head(3).iterrows():
            print(f"{i+1}. {row['state']} ({row['migration_probability']:.2f}%)")
    else:
        print("Could not generate ranking.")
    
    # Generate Report
    print("\n[Step 6] Generating Detailed Report...")
    report_path = generate_report(aadhar_df, news_df, merged_data, model_info, ranking, args.output)
    print(f"Report saved to: {report_path}")
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
