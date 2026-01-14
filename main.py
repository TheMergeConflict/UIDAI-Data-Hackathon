import os
import argparse
from datetime import datetime
import pandas as pd
import shutil
import glob
from data_loader import load_uidai_data
from news_fetcher import fetch_all_news
from analysis import MigrationAnalyzer

def generate_report(aadhar_df, news_df, merged_data, model_info, ranking, trend_data, user_file, comparison_results, internal_analysis, output_path):
    base, ext = os.path.splitext(output_path)
    if not ext:
        output_path = f"{output_path}.md"
        ext = ".md"
    
    ext = ext.lower()
    
    if ext == '.pdf':
        return _generate_pdf_report(aadhar_df, news_df, merged_data, model_info, ranking, trend_data, user_file, comparison_results, internal_analysis, output_path)
    elif ext in ['.md', '.markdown']:
        return _generate_md_report(aadhar_df, news_df, merged_data, model_info, ranking, trend_data, user_file, comparison_results, internal_analysis, output_path)
    elif ext == '.txt':
        return _generate_txt_report(aadhar_df, news_df, merged_data, model_info, ranking, trend_data, user_file, comparison_results, internal_analysis, output_path)
    else:
        print(f"Unsupported format '{ext}'. Defaulting to Markdown.")
        return _generate_md_report(aadhar_df, news_df, merged_data, model_info, ranking, trend_data, user_file, comparison_results, internal_analysis, f"{base}.md")

def _generate_pdf_report(aadhar_df, news_df, merged_data, model_info, ranking, trend_data, user_file, comparison_results, internal_analysis, output_path):
    try:
        from fpdf import FPDF
    except ImportError:
        os.system('pip install fpdf')
        from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Aadhar Migration Prediction Report (Integrated)', 0, 1, 'C')
            self.ln(5)
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, f"Generated: {timestamp}", 0, 1)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Executive Summary", 0, 1)
    if not ranking.empty:
        top = ranking.iloc[0]
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 7, f"Based on trend analysis and real-time market bias, {top['state']} shows the highest migration probability ({top['migration_probability']:.2f}%). Bias Reason: {top['bias_reason']}.")
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Integrated Prediction Ranking", 0, 1)
    r_widths = [10, 35, 25, 15, 15, 25, 15, 50]
    r_headers = ['Rank', 'State', 'Base', 'News', 'Bias', 'Final', 'Prob', 'Primary Reason']
    pdf.set_fill_color(220, 230, 255)
    pdf.set_font("Arial", 'B', 8)
    for i, h in enumerate(r_headers):
        pdf.cell(r_widths[i], 10, h, 1, 0, 'C', 1)
    pdf.ln()
    
    pdf.set_font("Arial", size=8)
    for i, row in ranking.iterrows():
        if pdf.get_y() > 260:
            pdf.add_page()
        pdf.cell(r_widths[0], 10, str(i+1), 1, 0, 'C')
        pdf.cell(r_widths[1], 10, str(row['state']), 1)
        pdf.cell(r_widths[2], 10, f"{row['base_prediction']:,.0f}", 1, 0, 'R')
        pdf.cell(r_widths[3], 10, f"{row['news_adjustment']:.2f}x", 1, 0, 'C')
        pdf.cell(r_widths[4], 10, f"{row['momentum_bias']:.2f}x", 1, 0, 'C')
        pdf.cell(r_widths[5], 10, f"{row['final_prediction']:,.0f}", 1, 0, 'R')
        pdf.cell(r_widths[6], 10, f"{row['migration_probability']:.2f}%", 1, 0, 'R')
        pdf.cell(r_widths[7], 10, str(row['bias_reason']), 1)
        pdf.ln()

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Methodology", 0, 1)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, "1. Trend: Baseline forecast from MoM Aadhar update patterns.\n2. News: Adjusted by hiring/layoff sentiment.\n3. Bias: Alignment with new user logs or national YoY performance.")

    pdf.output(output_path)
    return output_path

def _generate_txt_report(aadhar_df, news_df, merged_data, model_info, ranking, trend_data, user_file, comparison_results, internal_analysis, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("AADHAR MIGRATION PREDICTION REPORT (INTEGRATED)\n")
        f.write("==============================================\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("FINAL MIGRATION RANKING\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<5} | {'State':<20} | {'Prob':<8} | {'Bias Reason'}\n")
        f.write("-" * 80 + "\n")
        if not ranking.empty:
            for i, row in ranking.iterrows():
                f.write(f"{i+1:<5} | {row['state']:<20} | {row['migration_probability']:>7.2f}% | {row['bias_reason']}\n")
        f.write("-" * 80 + "\n\n")
        f.write("METHODOLOGY\n")
        f.write("1. Trends: Base forecast from monthly updates.\n")
        f.write("2. News: Adjusted by hiring/layoff sentiment.\n")
        f.write("3. Market Bias: Integrated alignment with new data or national YoY performance.\n")
    return output_path

def _generate_md_report(aadhar_df, news_df, merged_data, model_info, ranking, trend_data, user_file, comparison_results, internal_analysis, output_path):
    report_lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines.append("# Aadhar Migration Prediction Report\n")
    report_lines.append(f"**Generated:** {timestamp}\n")
    report_lines.append("\n---\n")
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
    report_lines.append("## 1. Data Overview\n\n")
    report_lines.append("### Aadhar Demographic Data\n")
    report_lines.append(f"- **Total Records:** {len(aadhar_df):,}\n")
    report_lines.append(f"- **States Analyzed:** {len(aadhar_df['state'].unique())}\n")
    report_lines.append(f"- **Date Range:** {aadhar_df['date'].min()} to {aadhar_df['date'].max()}\n")
    report_lines.append("\n### News Data (Add-on Factor)\n")
    report_lines.append(f"- **Total Articles Fetched:** {len(news_df):,}\n")
    report_lines.append(f"- **Search Keywords:** jobs, migration, hiring, industrial growth, layoffs\n")
    report_lines.append("\n---\n")
    report_lines.append("## 2. Monthly Trend Analysis (Primary Prediction Method)\n\n")
    report_lines.append("The prediction is primarily based on month-over-month (MoM) trends in Aadhar updates.\n\n")
    report_lines.append("| State | Avg Monthly Updates | Avg Growth Rate | Trend | Latest Updates |\n")
    report_lines.append("|-------|---------------------|-----------------|-------|----------------|\n")
    if trend_data is not None and not trend_data.empty:
        for _, row in trend_data.iterrows():
            growth_str = f"{row['avg_growth_rate']:.1f}%" if not pd.isna(row['avg_growth_rate']) else "N/A"
            report_lines.append(f"| {row['state']} | {row['avg_monthly_updates']:,.0f} | {growth_str} | {row['trend_direction']} | {row['latest_updates']:,.0f} |\n")
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
    report_lines.append("## 4. Integrated Migration Analysis & Predictions\n\n")
    report_lines.append("This table combines monthly trends, news sentiment, and market momentum (comparison bias) into a single unified forecast.\n\n")
    report_lines.append("| Rank | State | Base Prediction | News Adj | Market Bias | Final Prediction | Prob | Primary Bias Reason |\n")
    report_lines.append("|------|-------|-----------------|----------|-------------|------------------|------|---------------------|\n")
    if not ranking.empty:
        for i, row in ranking.iterrows():
            report_lines.append(f"| {i+1} | {row['state']} | {row['base_prediction']:,.0f} | {row['news_adjustment']:.2f}x | {row['momentum_bias']:.2f}x | {row['final_prediction']:,.0f} | {row['migration_probability']:.2f}% | {row['bias_reason']} |\n")
    report_lines.append("\n---\n")
    report_lines.append("## 5. Methodology & Reasoning\n\n")
    report_lines.append("### Unified Prediction Logic\n\n")
    report_lines.append("1. **Trend Analysis**: Base prediction derived from month-over-month growth patterns.\n")
    report_lines.append("2. **News Factor**: Real-time adjustment (0.5x - 1.5x) based on hiring/job market news.\n")
    report_lines.append("3. **Market Alignment (Integrated)**: Dynamic alignment based on user-provided logs or internal YoY performance compared to the national baseline.\n")
    report_lines.append("\n---\n")
    report_lines.append("## 6. Conclusion & Recommendations\n\n")
    if not ranking.empty:
        top3 = ranking.head(3)
        report_lines.append("### Top Predicted Migration Hotspots\n\n")
        for i, row in top3.iterrows():
            report_lines.append(f"{i+1}. **{row['state']}** ({row['migration_probability']:.2f}%) - {row['bias_reason']}\n")
        report_lines.append("\n### Recommendations\n\n")
        report_lines.append("- Prioritize resource deployment based on integrated probability scores.\n")
        report_lines.append("- Monitor the 'Primary Bias Reason' for specific local market triggers.\n")
    report_lines.append("\n---\n")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    return output_path

def move_comparison_files(source_path, target_dir):
    print(f"\n[Post-Execution] Processing file movement to {target_dir}...")
    files_to_move = []
    if os.path.isdir(source_path):
        files_to_move = glob.glob(os.path.join(source_path, "*.csv"))
    elif os.path.isfile(source_path) and source_path.endswith('.csv'):
        files_to_move = [source_path]
        
    for f in files_to_move:
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip().lower() for c in df.columns]
            state = df['state'].iloc[0] if 'state' in df.columns and len(df) > 0 else "UnknownState"
            year = "UnknownYear"
            if 'date' in df.columns and len(df) > 0:
                dt = pd.to_datetime(df['date'].iloc[0], dayfirst=True, errors='coerce')
                if not pd.isna(dt):
                    year = dt.year
            
            target_file = f"{state}_{year}.csv".replace(" ", "_")
            dest = os.path.join(target_dir, target_file)
            counter = 1
            while os.path.exists(dest):
                dest = os.path.join(target_dir, f"{state}_{year}_{counter}.csv".replace(" ", "_"))
                counter += 1
            shutil.move(f, dest)
            print(f"  - Moved: {os.path.basename(f)} -> {os.path.basename(dest)}")
        except Exception as e:
            print(f"  - Error moving {f}: {e}")

    if os.path.isdir(source_path):
        for item in os.listdir(source_path):
            item_path = os.path.join(source_path, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"  - Warning: Could not clean up {item_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Aadhar Migration Prediction System")
    parser.add_argument("--data_dir", default=os.path.join(os.getcwd(), "Data"), help="Directory containing Aadhar CSV files")
    parser.add_argument("--test_run", action="store_true", help="Run with a small subset or synthetic news")
    parser.add_argument("--compare", help="Path to user CSV file for comparison")
    parser.add_argument("--output", default="migration_report_v2.md", help="Output report filename")
    args = parser.parse_args()
    
    print("=== Starting Aadhar Migration Prediction System ===")
    print("\n[Step 1] Loading Aadhar Data...")
    aadhar_df = load_uidai_data(args.data_dir)
    print(f"Loaded {len(aadhar_df)} Aadhar update records.")
    
    if aadhar_df.empty:
        print("CRITICAL: No Aadhar data found. Exiting.")
        return
        
    states = aadhar_df['state'].unique().tolist()
    print(f"Detected States: {states}")
    
    print("\n[Step 2] Fetching News Data...")
    if args.test_run:
        print("Test run mode: Fetching news for only the first state to save time.")
        states = states[:1]
        
    keywords = ['jobs', 'migration', 'hiring', 'industrial growth', 'layoffs']
    news_df = fetch_all_news(states, queries=keywords)
    print(f"Fetched {len(news_df)} news articles.")
    
    print("\n[Step 3] Preparing Data...")
    analyzer = MigrationAnalyzer()
    merged_data = analyzer.prepare_data(aadhar_df, news_df)
    
    if merged_data.empty:
        print("Analysis failed: No data after merging.")
        return
        
    print(f"Prepared {len(merged_data)} state-month data points.")
    
    print("\n[Step 4] Monthly Comparison Analysis...")
    monthly_comparison, pct_change = analyzer.get_monthly_comparison(merged_data)
    
    print("\nMonthly Updates by State (Absolute Values):")
    print("-" * 80)
    print(monthly_comparison.to_string())
    print("-" * 80)
    
    print("\nMonth-over-Month % Change:")
    print("-" * 80)
    pct_display = pct_change.map(lambda x: f"{x:+.1f}%" if x != 0 else "---")
    print(pct_display.to_string())
    print("-" * 80)

    print("\n[Step 5] Analyzing News Correlations...")
    model_info = analyzer.train_model(merged_data)
    
    comparison_results = pd.DataFrame()
    internal_analysis = {}
    
    if args.compare:
        print(f"\n[Step 6] Comparing with User Data: {args.compare}...")
        try:
            if os.path.isdir(args.compare):
                user_df = load_uidai_data(args.compare)
            else:
                user_df = pd.read_csv(args.compare)
                user_df.columns = [c.strip().lower() for c in user_df.columns]
                if 'date' in user_df.columns:
                    user_df['date'] = pd.to_datetime(user_df['date'], dayfirst=True, errors='coerce')
                demo_cols = [c for c in user_df.columns if c.startswith('demo_')]
                if demo_cols and 'total_updates' not in user_df.columns:
                    user_df['total_updates'] = user_df[demo_cols].sum(axis=1)
            
            if not user_df.empty:
                comparison_results = analyzer.perform_comparison(merged_data, user_df)
            
            if not comparison_results.empty:
                print("\n=== User Data Comparison (Delta vs Reference) ===")
                print(f"{'State':<20} | {'Month':<8} | {'Ref':<8} | {'User':<8} | {'Diff':<8} | {'Dev %':<8}")
                print("-" * 75)
                for _, row in comparison_results.iterrows():
                    month_str = str(row['month_year'])
                    print(f"{row['state']:<20} | {month_str:<8} | {row['ref_updates']:<8.0f} | {row['user_updates']:<8.0f} | {row['diff']:<8.0f} | {row['pct_deviation']:.1f}%")
                print("-" * 75)
            else:
                print("Comparison yielded no matching records.")
        except Exception as e:
            print(f"Error loading comparison file: {e}")
    else:
        print("\n[Step 6] No external data provided. Performing Internal Market Analysis...")
        internal_analysis = analyzer.perform_internal_comparison(merged_data)
        
        if 'state_vs_national' in internal_analysis:
            df_vn = internal_analysis['state_vs_national']
            print("\n=== State Growth vs National Average ===")
            print(f"{'State':<20} | {'Growth %':<10} | {'Nat Avg %':<10} | {'Status'}")
            print("-" * 65)
            for _, row in df_vn.iterrows():
                status = "OUTPERFORMING" if row['is_outperforming'] else "UNDERPERFORMING"
                print(f"{row['state']:<20} | {row['state_growth']:>10.1f}% | {row['national_growth']:>10.1f}% | {status}")
            print("-" * 65)

    print("\n[Step 7] Predicting Migration (Trends + News + Momentum Bias)...")
    ranking, trend_data = analyzer.predict_next_cycle(merged_data, news_df, comparison_results, internal_analysis)
    
    if not ranking.empty:
        print("\n=== Predicted Migration Hotspots (Ranked) ===")
        print(f"{'Rank':<5} | {'State':<18} | {'Trend':<12} | {'News':<8} | {'Bias':<6} | {'Final Pred':<10} | {'Prob':<8}")
        print("-" * 90)
        for i, row in ranking.iterrows():
            print(f"{i+1:<5} | {row['state']:<18} | {row['trend_direction']:<12} | {row['news_sentiment']:<8} | {row['momentum_bias']:<6.2f} | {row['final_prediction']:>10,.0f} | {row['migration_probability']:.2f}%")
        print("-" * 90)
        
        print("\nTop Insights:")
        for i, row in ranking.head(3).iterrows():
            print(f"- {row['state']}: {row['bias_reason']} ({row['migration_probability']:.2f}%)")
    else:
        print("Could not generate ranking.")

    print("\n[Step 8] Generating Detailed Report...")
    report_path = generate_report(aadhar_df, news_df, merged_data, model_info, ranking, trend_data, args.compare, comparison_results, internal_analysis, args.output)
    print(f"Report saved to: {report_path}")
    
    if args.compare and not comparison_results.empty:
        move_comparison_files(args.compare, args.data_dir)
        
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
