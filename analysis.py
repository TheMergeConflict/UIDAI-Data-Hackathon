import pandas as pd
import numpy as np

class CustomLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        # Add intercept term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Normal Equation: theta = (X.T * X)^-1 * X.T * y
        # Using pseudoinverse for stability
        theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(np.r_[self.intercept_, self.coef_])

class MigrationAnalyzer:
    def __init__(self):
        self.model = CustomLinearRegression()
        
    def prepare_data(self, aadhar_df, news_df):
        """
        Merges Aadhar and News data on State and Month.
        """
        if aadhar_df.empty:
            print("Aadhar DataFrame is empty.")
            return pd.DataFrame()
            
        # 1. Aggregate Aadhar Data by State and Month
        # Ensure date is datetime
        aadhar_df['date'] = pd.to_datetime(aadhar_df['date'])
        # Create Month-Year feature for aggregation
        aadhar_df['month_year'] = aadhar_df['date'].dt.to_period('M')
        
        # We need to map 'Andhra Pradesh' etc. to handle potential case issues
        aadhar_df['state'] = aadhar_df['state'].str.strip()
        
        # Sum total_updates per state-month
        aadhar_agg = aadhar_df.groupby(['state', 'month_year'])['total_updates'].sum().reset_index()
        
        # 2. Aggregate News Data
        if news_df.empty:
            print("News DataFrame is empty. Creating dummy news features for training.")
            # Create synthetic news features if real data fails (for demo purposes)
            aadhar_agg['job_news_count'] = np.random.randint(0, 50, size=len(aadhar_agg))
            aadhar_agg['migration_news_count'] = np.random.randint(0, 20, size=len(aadhar_agg))
            return aadhar_agg
        
        news_df['published'] = pd.to_datetime(news_df['published'])
        news_df['month_year'] = news_df['published'].dt.to_period('M')
        news_df['state'] = news_df['region'].str.strip()
        
        # Count news items per state-month
        # Assuming query column tells us the category (jobs/migration)
        news_agg = news_df.groupby(['state', 'month_year', 'query']).size().unstack(fill_value=0).reset_index()
        
        # Rename columns to be cleaner (e.g. 'jobs' -> 'job_news_count')
        news_agg.columns = [
            f'{c}_news_count' if c not in ['state', 'month_year'] else c 
            for c in news_agg.columns
        ]
        
        # 3. Merge
        # Left merge on Aadhar (primary source of truth for migration events)
        merged_df = pd.merge(aadhar_agg, news_agg, on=['state', 'month_year'], how='left')
        merged_df.fillna(0, inplace=True)
        
        return merged_df

    def train_model(self, df):
        """
        Trains a simple prediction model.
        Target: total_updates
        Features: *_news_count
        """
        feature_cols = [c for c in df.columns if c.endswith('_news_count')]
        target_col = 'total_updates'
        
        if not feature_cols:
            print("No news features found.")
            return
            
        # Convert to numpy and numeric types
        for col in feature_cols:
             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
             
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Split (manual split)
        n_samples = len(df)
        if n_samples > 5:
            split_idx = int(n_samples * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        else:
            X_train, y_train = X, y
            X_test, y_test = X, y
            
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        mse = np.mean((y_test - preds) ** 2)
        rmse = np.sqrt(mse)
        
        print(f"Model Trained Successfully!")
        print(f"  - Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"  - Root Mean Square Error: {rmse:,.0f} (avg prediction deviation)")
        
        # Build model info dictionary for report
        model_info = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'rmse': rmse,
            'factors': {}
        }
        
        # User-friendly interpretation of coefficients
        print("\nKey Factors Influencing Migration:")
        print(f"{'Factor':<30} | {'Impact':<10} | {'Interpretation':<30}")
        print("-" * 75)
        
        if self.model.coef_ is not None:
            for feat, coef in zip(feature_cols, self.model.coef_):
                # Clean up feature name for display
                display_name = feat.replace('_news_count', ' News').replace('_', ' ').title()
                
                if coef > 1000:
                    impact = "STRONG +"
                    interpretation = "High migration when more news"
                elif coef > 0:
                    impact = "Weak +"
                    interpretation = "Slight increase with more news"
                elif coef < -1000:
                    impact = "STRONG -"
                    interpretation = "Lower migration when more news"
                else:
                    impact = "Weak -"
                    interpretation = "Slight decrease with more news"
                
                model_info['factors'][display_name] = {
                    'impact': impact,
                    'interpretation': interpretation,
                    'coefficient': coef
                }
                    
                print(f"{display_name:<30} | {impact:<10} | {interpretation:<30}")
        
        print("-" * 75)
        
        # Correlation summary (simplified)
        correlations = df[feature_cols + [target_col]].corr()[target_col].drop(target_col)
        top_corr = correlations.abs().idxmax()
        top_corr_val = correlations[top_corr]
        top_corr_name = top_corr.replace('_news_count', ' News').replace('_', ' ').title()
        model_info['strongest_correlation'] = f"{top_corr_name} ({top_corr_val:.2%} correlation)"
        print(f"\nStrongest Correlation: {top_corr_name} ({top_corr_val:.2%} correlation with migration)")
        
        return model_info

    def predict_next_cycle(self, recent_news_df):
        """
        Predicts migration for the next cycle given recent news features.
        Returns a DataFrame ranked by predicted migration probability.
        """
        if recent_news_df.empty:
            print("No recent news data for prediction.")
            return pd.DataFrame()
            
        recent_news_df['published'] = pd.to_datetime(recent_news_df['published'])
        
        recent_news_df['state'] = recent_news_df['region'].str.strip()
        
        # Aggregate news counts per state (ignoring month here, assuming input is 'current period')
        news_agg = recent_news_df.groupby(['state', 'query']).size().unstack(fill_value=0).reset_index()
        
        # Rename columns from query names to expected feature names (e.g., 'jobs' -> 'jobs_news_count')
        rename_map = {col: f'{col}_news_count' for col in news_agg.columns if col != 'state'}
        news_agg = news_agg.rename(columns=rename_map)
        
        # Ensure we have all feature columns the model expects
        if self.model.coef_ is None:
            print("Model not trained yet.")
            return pd.DataFrame()
            
        expected_keywords = ['jobs', 'migration', 'hiring', 'industrial growth', 'layoffs']
        expected_features = [f'{k}_news_count' for k in expected_keywords]
        
        for feat in expected_features:
            if feat not in news_agg.columns:
                news_agg[feat] = 0
                
        # Select features in order
        X_pred = news_agg[expected_features].values
        
        # Predict
        preds = self.model.predict(X_pred)
        
        # Clip negative predictions to 0
        preds = np.maximum(preds, 0)
        
        results = news_agg[['state']].copy()
        
        # Calculate Probability
        total_predicted = np.sum(preds)
        if total_predicted > 0:
            probs = (preds / total_predicted) * 100
        else:
            probs = np.zeros_like(preds)
            
        results['migration_probability'] = probs
        
        # Sort descending
        results = results.sort_values(by='migration_probability', ascending=False).reset_index(drop=True)
        
        return results

if __name__ == "__main__":
    # Test
    # Create dummy data
    dates = pd.date_range("2025-01-01", periods=6, freq="M")
    aadhar_data = pd.DataFrame({
        'date': dates,
        'state': ['Andhra Pradesh'] * 6,
        'total_updates': [100, 150, 200, 130, 250, 300]
    })
    
    news_data = pd.DataFrame({
        'published': dates,
        'region': ['Andhra Pradesh'] * 6,
        'query': ['jobs'] * 6
    })
    
    analyzer = MigrationAnalyzer()
    df = analyzer.prepare_data(aadhar_data, news_data)
    print("Prepared Data:")
    print(df.head())
    
    analyzer.train_model(df)
