import pandas as pd
import glob
import os

def load_uidai_data(directory_path):
    """
    Loads all CSV files in the directory and aggregates them.
    Assumes columns: date, state, district, pincode, demo_age_...
    """
    all_files = glob.glob(os.path.join(directory_path, "*.csv"))
    df_list = []
    
    print(f"Found {len(all_files)} CSV files in {directory_path}")
    
    for f in all_files:
        try:
            # Inspect first to avoid errors? No, just try read
            df = pd.read_csv(f)
            # Basic validation
            if 'state' in df.columns:
                df_list.append(df)
            else:
                print(f"Skipping {f}: 'state' column not found.")
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    if not df_list:
        print("No valid data loaded.")
        return pd.DataFrame()
    
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Clean column names
    full_df.columns = [c.strip().lower() for c in full_df.columns]
    
    # Parse Date
    if 'date' in full_df.columns:
        full_df['date'] = pd.to_datetime(full_df['date'], dayfirst=True, errors='coerce')
    
    # Calculate Total Updates
    # Sum all columns starting with 'demo_'
    demo_cols = [c for c in full_df.columns if c.startswith('demo_')]
    if demo_cols:
        full_df['total_updates'] = full_df[demo_cols].sum(axis=1)
    else:
        # If no demo columns, maybe there's already a count? 
        # For now, default to 0 if missing, but we saw them in the file.
        full_df['total_updates'] = 0
        
    return full_df

if __name__ == "__main__":
    # Test run
    path = os.getcwd()
    print(f"Loading data from {path}...")
    df = load_uidai_data(path)
    print(f"Loaded {len(df)} rows.")
    if not df.empty:
        print(df.head())
        print("Columns:", df.columns.tolist())
        print("States found:", df['state'].unique())
