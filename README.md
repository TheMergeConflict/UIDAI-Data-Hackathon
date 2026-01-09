# Aadhar Migration Prediction System

## Overview

The Aadhar Migration Prediction System is a machine learning-based tool that analyzes demographic update patterns from UIDAI Aadhar data and correlates them with current news trends to predict future migration cycles across Indian states.

---

## How It Works

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Data Loader   │     │  News Fetcher   │     │    Analyzer     │
│  (data_loader)  │     │ (news_fetcher)  │     │   (analysis)    │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         │    Aadhar CSV Data    │   Google News RSS     │
         │                       │                       │
         └───────────┬───────────┴───────────────────────┘
                     │
                     ▼
              ┌──────────────┐
              │   main.py    │
              │ (Orchestrator)│
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │    Report    │
              │ (.md output) │
              └──────────────┘
```

---

## Workflow

### Step 1: Load Aadhar Data (`data_loader.py`)

- Reads all CSV files from the `Data/` folder
- Each CSV contains demographic update records for a state
- Extracts: date, state name, total updates count
- Combines all state data into a single DataFrame

### Step 2: Fetch News Data (`news_fetcher.py`)

- For each detected state, fetches news from Google News RSS
- Search keywords: `jobs`, `migration`, `hiring`, `industrial growth`, `layoffs`
- Extracts: title, published date, region, query category
- Rate-limited to 1 request/second to respect server limits

### Step 3: Prepare Data (`analysis.py` - `prepare_data`)

- Aggregates Aadhar data by state and month
- Aggregates news counts by state, month, and category
- Merges both datasets on state + month
- Creates features: `{category}_news_count` for each news category

### Step 4: Train Model (`analysis.py` - `train_model`)

- Uses a custom NumPy-based Linear Regression implementation
- **Target variable:** `total_updates` (Aadhar demographic changes)
- **Features:** News counts per category
- Trains on 80% of data, tests on 20%
- Outputs: coefficients, RMSE, factor interpretations

### Step 5: Predict & Rank (`analysis.py` - `predict_next_cycle`)

- Aggregates current news counts per state
- Uses trained model to predict migration magnitude
- Converts predictions to probability percentages (relative share)
- Ranks states by migration probability

### Step 6: Generate Report (`main.py` - `generate_report`)

- Creates a detailed markdown report with:
  - Executive summary
  - Data overview
  - State-wise analysis
  - Model insights and reasoning
  - Predictions with interpretations
  - Recommendations

---

## The Prediction Logic

### Core Hypothesis

> **News activity in a region serves as a leading indicator of migration patterns.**

Economic news (hiring, industrial growth) attracts migrants seeking opportunities, while negative news (layoffs) may trigger outward migration.

### Feature Interpretation

| News Category       | Expected Effect on Migration |
|---------------------|------------------------------|
| Hiring News         | **Positive** - Job opportunities attract migrants |
| Industrial Growth   | **Positive** - Economic development draws workers |
| Jobs News           | **Mixed** - May indicate competition or opportunity |
| Layoffs News        | **Negative** - Economic distress discourages migration |
| Migration News      | **Lagging** - Reports on already-happening events |

### Model Coefficients

The linear regression model learns weights for each news category:
- **Positive coefficient** = More news → More predicted migration
- **Negative coefficient** = More news → Less predicted migration
- **Magnitude** = Strength of the relationship

### Probability Calculation

```
State Probability = (State's Predicted Value / Total Predicted Value) × 100%
```

This gives a relative ranking showing which states are most likely to experience migration activity compared to others.

---

## File Structure

```
UIDAI Hackathon/
├── Data/                    # Aadhar CSV files (one per state)
│   ├── Andhra Pradesh.csv
│   ├── Delhi.csv
│   └── ...
├── main.py                  # Main orchestrator & report generator
├── data_loader.py           # Loads and parses Aadhar CSVs
├── news_fetcher.py          # Fetches news from Google RSS
├── analysis.py              # ML model and prediction logic
├── migration_report.md      # Generated output report
└── README.md                # This documentation
```

---

## Usage

### Basic Run
```bash
python main.py
```

### Test Mode (faster, fewer API calls)
```bash
python main.py --test_run
```

### Custom Data Directory
```bash
python main.py --data_dir "C:/path/to/csv/files"
```

### Custom Output Report
```bash
python main.py --output "custom_report.md"
```

---

## Limitations & Considerations

1. **Live Data Variability**: News is fetched in real-time; different runs may yield different results as news updates.

2. **Correlation ≠ Causation**: The model finds statistical relationships, not causal links between news and migration.

3. **Data Quality**: Predictions are only as good as the input data. Missing or incomplete Aadhar data affects accuracy.

4. **Linear Model**: The simple linear regression may not capture complex non-linear relationships. More sophisticated models (Random Forest, XGBoost) could improve accuracy.

5. **News Relevance**: Google News results may include tangentially related articles that add noise to predictions.

---

## Future Enhancements

- [ ] Add data caching to ensure reproducible results
- [ ] Implement more advanced ML models (ensemble methods)
- [ ] Add sentiment analysis on news articles
- [ ] Include historical weather and economic indicators
- [ ] Create interactive visualization dashboard

---

## Dependencies

- Python 3.x
- pandas
- numpy
- feedparser (for RSS parsing)

Install with:
```bash
pip install pandas numpy feedparser
```
