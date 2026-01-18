import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile

try:
    from analysis import MigrationAnalyzer
    from news_fetcher import fetch_all_news
    from data_loader import load_uidai_data # don't need this, but just in case 
except ImportError as e:
    print(f"CRITICAL ERROR: Missing modules. Ensure analysis.py and news_fetcher.py are present.\n{e}")

app = Flask(__name__)
CORS(app)  # react ui 

# configure temp upload folder to store CSVs briefly during processing
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def parse_logs(msg):
    """Format logs for the frontend console"""
    return f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}"

# --- ROUTES ---

@app.route('/', methods=['GET'])
def index():
    """Fixes the 404 error when you visit localhost:5000 in the browser"""
    return jsonify({
        "status": "online",
        "message": "Aadhar Migration Backend is Running!",
        "instructions": "Go to the React UI to upload files and run analysis."
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Used by the React App to check if server is online"""
    return jsonify({"status": "online", "message": "Ready to process"})

@app.route('/analyze', methods=['POST'])
def analyze_migration():
    logs = []
    try:
        logs.append(parse_logs("Received analysis request from UI."))
        
        # handle uploads
        if 'files' not in request.files:
            return jsonify({"error": "No files uploaded"}), 400
        
        files = request.files.getlist('files')
        dfs = []
        
        logs.append(parse_logs(f"Processing {len(files)} uploaded files..."))
        
        # parse
        for file in files:
            if file.filename == '':
                continue
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # basic CSV loading logic matching data_loader
                df = pd.read_csv(filepath)
                df.columns = [c.strip().lower() for c in df.columns]
                
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
                
                # Calculate total updates if missing
                demo_cols = [c for c in df.columns if c.startswith('demo_')]
                if demo_cols and 'total_updates' not in df.columns:
                    df['total_updates'] = df[demo_cols].sum(axis=1)
                elif 'total_updates' not in df.columns:
                    df['total_updates'] = 0
                
                dfs.append(df)
            except Exception as e:
                logs.append(parse_logs(f"Error reading {filename}: {str(e)}"))

        if not dfs:
            return jsonify({"error": "No valid data found in uploaded files"}), 400

        full_df = pd.concat(dfs, ignore_index=True)
        states = full_df['state'].unique().tolist() if 'state' in full_df.columns else []
        logs.append(parse_logs(f"Data Loaded: {len(full_df)} rows across {len(states)} states."))

        # fetch news
        logs.append(parse_logs("Fetching live news signals (Google RSS)..."))
        
        # limit to first few states to avoid browser time outs
        # this is off, to ensure that all states are processed
        # but it causes it to take far longer to run
        # target_states = states[:5] if len(states) > 5 else states
        
        target_states = states
        
        keywords = ['jobs', 'migration', 'layoffs']
        
        try:
            logs.append(parse_logs(f"Querying news for {len(target_states)} states. This may take a moment..."))
            news_df = fetch_all_news(target_states, queries=keywords)
            logs.append(parse_logs(f"Fetched {len(news_df)} relevant news articles."))
        except Exception as e:
            logs.append(parse_logs(f"News fetch warning: {e}"))
            news_df = pd.DataFrame() # continue even if news fails

        # analyse
        logs.append(parse_logs("Executing MigrationAnalyzer logic..."))
        analyzer = MigrationAnalyzer()
        
        # prepare data structure expected by your class
        merged_data = analyzer.prepare_data(full_df, news_df)
        
        # run predictions
        # note: comparison_results and internal_analysis passed as empty for this endpoint
        # unless we expand the API to accept reference files separately.
        ranking, trend_data = analyzer.predict_next_cycle(merged_data, news_df)
        
        # format results for frontend
        results = []
        if not ranking.empty:
            for _, row in ranking.iterrows():
                # convert numpy to native types
                results.append({
                    "state": str(row['state']),
                    "basePred": int(row['base_prediction']),
                    "newsAdj": round(float(row['news_adjustment']), 2),
                    "finalPred": int(row['final_prediction']),
                    "probability": round(float(row['migration_probability']), 2),
                    "biasReason": str(row['bias_reason']),
                    "newsSentiment": str(row['news_sentiment'])
                })

        logs.append(parse_logs("Analysis complete. Sending results to Dashboard."))
        
        return jsonify({
            "status": "success",
            "logs": logs,
            "results": results,
            "meta": {
                "total_records": len(full_df),
                "news_count": len(news_df)
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "logs": logs}), 500

# a small thing on localhost:5000 to ensure its running
if __name__ == '__main__':
    print("-------------------------------------------------------")
    print(" AADHAR MIGRATION BACKEND SERVER")
    print("-------------------------------------------------------")
    print(" 1. Server running on: http://127.0.0.1:5000")
    print(" 2. Open the React Frontend to upload files.")
    print("-------------------------------------------------------")
    app.run(debug=True, port=5000)