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
        self.monthly_data = None
        self.trend_data = None
        self.news_factors = None
        
    def prepare_data(self, aadhar_df, news_df):
        """
        Merges Aadhar and News data on State and Month.
        """
        if aadhar_df.empty:
            print("Aadhar DataFrame is empty.")
            return pd.DataFrame()
            
        # 1. Aggregate Aadhar Data by State and Month
        aadhar_df['date'] = pd.to_datetime(aadhar_df['date'])
        aadhar_df['month_year'] = aadhar_df['date'].dt.to_period('M')
        aadhar_df['state'] = aadhar_df['state'].str.strip()
        
        # Sum total_updates per state-month
        aadhar_agg = aadhar_df.groupby(['state', 'month_year'])['total_updates'].sum().reset_index()
        
        # Sort by state and month for proper trend analysis
        aadhar_agg = aadhar_agg.sort_values(['state', 'month_year']).reset_index(drop=True)
        
        # 2. Aggregate News Data
        if news_df.empty:
            print("News DataFrame is empty. Creating dummy news features.")
            aadhar_agg['job_news_count'] = np.random.randint(0, 50, size=len(aadhar_agg))
            aadhar_agg['migration_news_count'] = np.random.randint(0, 20, size=len(aadhar_agg))
            return aadhar_agg
        
        news_df['published'] = pd.to_datetime(news_df['published'])
        news_df['month_year'] = news_df['published'].dt.to_period('M')
        news_df['state'] = news_df['region'].str.strip()
        
        # Count news items per state-month
        news_agg = news_df.groupby(['state', 'month_year', 'query']).size().unstack(fill_value=0).reset_index()
        
        news_agg.columns = [
            f'{c}_news_count' if c not in ['state', 'month_year'] else c 
            for c in news_agg.columns
        ]
        
        # 3. Merge
        merged_df = pd.merge(aadhar_agg, news_agg, on=['state', 'month_year'], how='left')
        merged_df.fillna(0, inplace=True)
        
        # Sort by state and month
        merged_df = merged_df.sort_values(['state', 'month_year']).reset_index(drop=True)
        
        return merged_df

    def calculate_monthly_trends(self, df):
        """
        Calculate month-over-month changes and trends for each state.
        Returns trend data with growth rates and predictions.
        """
        trend_results = []
        
        for state in df['state'].unique():
            state_data = df[df['state'] == state].sort_values('month_year').copy()
            
            if len(state_data) < 2:
                continue
            
            # Calculate month-over-month change
            state_data['prev_updates'] = state_data['total_updates'].shift(1)
            state_data['mom_change'] = state_data['total_updates'] - state_data['prev_updates']
            state_data['mom_pct_change'] = (state_data['mom_change'] / state_data['prev_updates']) * 100
            
            # Get stats
            avg_updates = state_data['total_updates'].mean()
            avg_growth_rate = state_data['mom_pct_change'].dropna().mean()
            latest_updates = state_data['total_updates'].iloc[-1]
            latest_growth = state_data['mom_pct_change'].iloc[-1] if len(state_data) > 1 else 0
            
            # Calculate trend direction using linear regression on updates
            x = np.arange(len(state_data)).reshape(-1, 1)
            y = state_data['total_updates'].values
            
            # Simple linear fit
            x_mean = x.mean()
            y_mean = y.mean()
            slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
            
            # Predict next month using trend
            predicted_next = latest_updates + slope
            predicted_next = max(predicted_next, 0)  # No negative predictions
            
            trend_results.append({
                'state': state,
                'months_analyzed': len(state_data),
                'avg_monthly_updates': avg_updates,
                'avg_growth_rate': avg_growth_rate,
                'latest_updates': latest_updates,
                'latest_growth_rate': latest_growth if not np.isnan(latest_growth) else 0,
                'trend_slope': slope,
                'trend_direction': 'Increasing' if avg_growth_rate > 0 else 'Decreasing',
                'base_predicted_next': predicted_next,
                'monthly_data': state_data.to_dict('records')
            })
        
        self.trend_data = pd.DataFrame(trend_results)
        return self.trend_data

    def calculate_news_adjustment(self, news_df):
        """
        Calculate news-based adjustment factors for each state.
        Returns adjustment multipliers based on current news sentiment.
        """
        if news_df.empty:
            return {}
        
        news_df['state'] = news_df['region'].str.strip()
        
        # Aggregate by state and query
        news_agg = news_df.groupby(['state', 'query']).size().unstack(fill_value=0)
        
        adjustments = {}
        
        for state in news_agg.index:
            state_news = news_agg.loc[state]
            
            # Positive factors (attract migration)
            positive_score = state_news.get('hiring', 0) * 2 + state_news.get('industrial growth', 0) * 1.5
            
            # Negative factors (discourage migration)
            negative_score = state_news.get('layoffs', 0) * 2
            
            # Neutral/mixed factors
            neutral_score = state_news.get('jobs', 0) * 0.5 + state_news.get('migration', 0) * 0.5
            
            # Calculate adjustment factor (centered at 1.0)
            total_news = positive_score + negative_score + neutral_score
            if total_news > 0:
                net_sentiment = (positive_score - negative_score + neutral_score) / total_news
                # Scale to adjustment factor (0.5 to 1.5 range)
                adjustment = 1.0 + (net_sentiment * 0.5)
                adjustment = max(0.5, min(1.5, adjustment))  # Clamp
            else:
                adjustment = 1.0
            
            adjustments[state] = {
                'adjustment_factor': adjustment,
                'positive_news': positive_score,
                'negative_news': negative_score,
                'total_news': total_news,
                'sentiment': 'Positive' if net_sentiment > 0.1 else ('Negative' if net_sentiment < -0.1 else 'Neutral')
            }
        
        self.news_factors = adjustments
        return adjustments

    def predict_next_cycle(self, merged_data, news_df):
        """
        Predicts migration for the next cycle using:
        1. Primary: Monthly trend analysis
        2. Secondary: News adjustment factor
        """
        # Step 1: Calculate monthly trends
        print("\n--- Monthly Trend Analysis ---")
        trend_data = self.calculate_monthly_trends(merged_data)
        
        if trend_data.empty:
            print("Could not calculate trends.")
            return pd.DataFrame(), {}
        
        # Step 2: Calculate news adjustments
        print("\n--- News Adjustment Factors ---")
        news_adjustments = self.calculate_news_adjustment(news_df)
        
        # Step 3: Combine predictions
        results = []
        
        for _, row in trend_data.iterrows():
            state = row['state']
            base_prediction = row['base_predicted_next']
            
            # Get news adjustment
            adj_info = news_adjustments.get(state, {'adjustment_factor': 1.0, 'sentiment': 'Neutral'})
            adjustment = adj_info['adjustment_factor']
            
            # Apply adjustment
            adjusted_prediction = base_prediction * adjustment
            
            results.append({
                'state': state,
                'latest_updates': row['latest_updates'],
                'trend_direction': row['trend_direction'],
                'avg_growth_rate': row['avg_growth_rate'],
                'base_prediction': base_prediction,
                'news_sentiment': adj_info['sentiment'],
                'news_adjustment': adjustment,
                'final_prediction': adjusted_prediction
            })
        
        results_df = pd.DataFrame(results)
        
        # Calculate probability based on final predictions
        total_pred = results_df['final_prediction'].sum()
        if total_pred > 0:
            results_df['migration_probability'] = (results_df['final_prediction'] / total_pred) * 100
        else:
            results_df['migration_probability'] = 0
        
        # Sort by probability
        results_df = results_df.sort_values('migration_probability', ascending=False).reset_index(drop=True)
        
        return results_df, self.trend_data

    def get_monthly_comparison(self, merged_data):
        """
        Returns data formatted for monthly comparison display with percentages.
        """
        if merged_data.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Pivot to show states as columns, months as rows
        comparison = merged_data.pivot_table(
            index='month_year', 
            columns='state', 
            values='total_updates',
            aggfunc='sum'
        ).fillna(0)
        
        # Sort by month
        comparison = comparison.sort_index()
        
        # Calculate percentage change for each state
        pct_change = comparison.pct_change() * 100
        pct_change = pct_change.fillna(0)
        
        return comparison, pct_change

    def train_model(self, df):
        """
        Trains the news correlation model (kept for backward compatibility).
        Now primarily used for understanding news-migration correlation.
        """
        feature_cols = [c for c in df.columns if c.endswith('_news_count')]
        target_col = 'total_updates'
        
        if not feature_cols:
            print("No news features found.")
            return {'train_samples': 0, 'test_samples': 0, 'rmse': 0, 'factors': {}, 'strongest_correlation': 'N/A'}
            
        for col in feature_cols:
             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
              
        X = df[feature_cols].values
        y = df[target_col].values
        
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
        
        model_info = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'rmse': rmse,
            'factors': {}
        }
        
        print(f"News Correlation Model Trained:")
        print(f"  - Samples: {len(X_train)} train, {len(X_test)} test")
        print(f"  - RMSE: {rmse:,.0f}")
        
        if self.model.coef_ is not None:
            print("\nNews Factor Correlations:")
            for feat, coef in zip(feature_cols, self.model.coef_):
                display_name = feat.replace('_news_count', ' News').replace('_', ' ').title()
                
                if coef > 1000:
                    impact = "STRONG +"
                elif coef > 0:
                    impact = "Weak +"
                elif coef < -1000:
                    impact = "STRONG -"
                else:
                    impact = "Weak -"
                
                model_info['factors'][display_name] = {
                    'impact': impact,
                    'coefficient': coef
                }
                print(f"  {display_name}: {impact}")
        
        # Correlation summary
        correlations = df[feature_cols + [target_col]].corr()[target_col].drop(target_col)
        top_corr = correlations.abs().idxmax()
        top_corr_val = correlations[top_corr]
        top_corr_name = top_corr.replace('_news_count', ' News').replace('_', ' ').title()
        model_info['strongest_correlation'] = f"{top_corr_name} ({top_corr_val:.2%})"
        
        return model_info


if __name__ == "__main__":
    # Test
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
    
    trends = analyzer.calculate_monthly_trends(df)
    print("\nTrend Data:")
    print(trends)
