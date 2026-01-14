import pandas as pd
import numpy as np

class CustomLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
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
        if aadhar_df.empty:
            print("Aadhar DataFrame is empty.")
            return pd.DataFrame()
            
        aadhar_df['date'] = pd.to_datetime(aadhar_df['date'])
        aadhar_df['month_year'] = aadhar_df['date'].dt.to_period('M')
        aadhar_df['state'] = aadhar_df['state'].str.strip()
        
        aadhar_agg = aadhar_df.groupby(['state', 'month_year'])['total_updates'].sum().reset_index()
        aadhar_agg = aadhar_agg.sort_values(['state', 'month_year']).reset_index(drop=True)
        
        if news_df.empty:
            print("News DataFrame is empty. Creating dummy news features.")
            aadhar_agg['job_news_count'] = np.random.randint(0, 50, size=len(aadhar_agg))
            aadhar_agg['migration_news_count'] = np.random.randint(0, 20, size=len(aadhar_agg))
            return aadhar_agg
        
        news_df['published'] = pd.to_datetime(news_df['published'])
        news_df['month_year'] = news_df['published'].dt.to_period('M')
        news_df['state'] = news_df['region'].str.strip()
        
        news_agg = news_df.groupby(['state', 'month_year', 'query']).size().unstack(fill_value=0).reset_index()
        news_agg.columns = [
            f'{c}_news_count' if c not in ['state', 'month_year'] else c 
            for c in news_agg.columns
        ]
        
        merged_df = pd.merge(aadhar_agg, news_agg, on=['state', 'month_year'], how='left')
        merged_df.fillna(0, inplace=True)
        merged_df = merged_df.sort_values(['state', 'month_year']).reset_index(drop=True)
        
        return merged_df

    def calculate_monthly_trends(self, df):
        trend_results = []
        for state in df['state'].unique():
            state_data = df[df['state'] == state].sort_values('month_year').copy()
            if len(state_data) < 2:
                continue
            
            state_data['prev_updates'] = state_data['total_updates'].shift(1)
            state_data['mom_change'] = state_data['total_updates'] - state_data['prev_updates']
            state_data['mom_pct_change'] = (state_data['mom_change'] / state_data['prev_updates']) * 100
            
            avg_updates = state_data['total_updates'].mean()
            avg_growth_rate = state_data['mom_pct_change'].dropna().mean()
            latest_updates = state_data['total_updates'].iloc[-1]
            latest_growth = state_data['mom_pct_change'].iloc[-1] if len(state_data) > 1 else 0
            
            x = np.arange(len(state_data)).reshape(-1, 1)
            y = state_data['total_updates'].values
            x_mean = x.mean()
            y_mean = y.mean()
            slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
            
            predicted_next = latest_updates + slope
            predicted_next = max(predicted_next, 0)
            
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
        if news_df.empty:
            return {}
        
        news_df['state'] = news_df['region'].str.strip()
        news_agg = news_df.groupby(['state', 'query']).size().unstack(fill_value=0)
        
        adjustments = {}
        for state in news_agg.index:
            state_news = news_agg.loc[state]
            positive_score = state_news.get('hiring', 0) * 2 + state_news.get('industrial growth', 0) * 1.5
            negative_score = state_news.get('layoffs', 0) * 2
            neutral_score = state_news.get('jobs', 0) * 0.5 + state_news.get('migration', 0) * 0.5
            
            total_news = positive_score + negative_score + neutral_score
            if total_news > 0:
                net_sentiment = (positive_score - negative_score + neutral_score) / total_news
                adjustment = 1.0 + (net_sentiment * 0.5)
                adjustment = max(0.5, min(1.5, adjustment))
            else:
                adjustment = 1.0
                net_sentiment = 0
            
            adjustments[state] = {
                'adjustment_factor': adjustment,
                'positive_news': positive_score,
                'negative_news': negative_score,
                'total_news': total_news,
                'sentiment': 'Positive' if net_sentiment > 0.1 else ('Negative' if net_sentiment < -0.1 else 'Neutral')
            }
        
        self.news_factors = adjustments
        return adjustments

    def predict_next_cycle(self, merged_data, news_df, comparison_results=pd.DataFrame(), internal_analysis={}):
        print("\n--- Monthly Trend Analysis ---")
        trend_data = self.calculate_monthly_trends(merged_data)
        if trend_data.empty:
            print("Could not calculate trends.")
            return pd.DataFrame(), pd.DataFrame()
        
        print("\n--- News Adjustment Factors ---")
        news_adjustments = self.calculate_news_adjustment(news_df)
        
        results = []
        for _, row in trend_data.iterrows():
            state = row['state']
            base_prediction = row['base_predicted_next']
            adj_info = news_adjustments.get(state, {'adjustment_factor': 1.0, 'sentiment': 'Neutral'})
            adjustment = adj_info['adjustment_factor']
            
            momentum_bias = 1.0
            bias_reason = "Baseline"
            
            if not comparison_results.empty:
                state_comp = comparison_results[comparison_results['state'] == state]
                if not state_comp.empty:
                    latest_dev = state_comp.sort_values('month_year').iloc[-1]['pct_deviation']
                    momentum_bias = 1.0 + (latest_dev / 100.0)
                    momentum_bias = max(0.8, min(1.2, momentum_bias))
                    bias_reason = f"User Data Deviation ({latest_dev:+.1f}%)"
            elif internal_analysis:
                if 'state_vs_national' in internal_analysis:
                    vn = internal_analysis['state_vs_national']
                    state_vn = vn[vn['state'] == state]
                    if not state_vn.empty and state_vn.iloc[0]['is_outperforming']:
                        momentum_bias *= 1.1
                        bias_reason = "Outperforming National Avg"
                
                if 'yoy_analysis' in internal_analysis:
                    yoy = internal_analysis['yoy_analysis']
                    state_yoy = yoy[yoy['state'] == state].sort_values('month')
                    if not state_yoy.empty:
                        latest_yoy = state_yoy.iloc[-1]['yoy_growth']
                        if latest_yoy > 20:
                            momentum_bias *= 1.05
                            if bias_reason == "Baseline":
                                bias_reason = "Strong YoY Growth"
                            else:
                                bias_reason += " + Strong YoY"

            final_prediction = base_prediction * adjustment * momentum_bias
            results.append({
                'state': state,
                'latest_updates': row['latest_updates'],
                'trend_direction': row['trend_direction'],
                'avg_growth_rate': row['avg_growth_rate'],
                'base_prediction': base_prediction,
                'news_sentiment': adj_info['sentiment'],
                'news_adjustment': adjustment,
                'momentum_bias': momentum_bias,
                'bias_reason': bias_reason,
                'final_prediction': final_prediction
            })
        
        results_df = pd.DataFrame(results)
        total_pred = results_df['final_prediction'].sum()
        if total_pred > 0:
            results_df['migration_probability'] = (results_df['final_prediction'] / total_pred) * 100
        else:
            results_df['migration_probability'] = 0
        
        results_df = results_df.sort_values('migration_probability', ascending=False).reset_index(drop=True)
        return results_df, trend_data

    def get_monthly_comparison(self, merged_data):
        if merged_data.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        comparison = merged_data.pivot_table(
            index='month_year', 
            columns='state', 
            values='total_updates',
            aggfunc='sum'
        ).fillna(0)
        comparison = comparison.sort_index()
        pct_change = comparison.pct_change() * 100
        pct_change = pct_change.fillna(0)
        return comparison, pct_change

    def perform_comparison(self, reference_df, user_df):
        if reference_df.empty or user_df.empty:
            return pd.DataFrame()
            
        if 'month_year' not in user_df.columns:
            user_df['date'] = pd.to_datetime(user_df['date'])
            user_df['month_year'] = user_df['date'].dt.to_period('M')
            
        user_agg = user_df.groupby(['state', 'month_year'])['total_updates'].sum().reset_index()
        user_agg.rename(columns={'total_updates': 'user_updates'}, inplace=True)
        
        ref_agg = reference_df[['state', 'month_year', 'total_updates']].copy()
        ref_agg.rename(columns={'total_updates': 'ref_updates'}, inplace=True)
        
        comp_df = pd.merge(ref_agg, user_agg, on=['state', 'month_year'], how='inner')
        comp_df['diff'] = comp_df['user_updates'] - comp_df['ref_updates']
        comp_df['pct_deviation'] = comp_df.apply(
            lambda row: (row['diff'] / row['ref_updates']) * 100 if row['ref_updates'] != 0 else 0,
            axis=1
        )
        return comp_df.sort_values(['state', 'month_year'])

    def perform_internal_comparison(self, df):
        if df.empty:
            return {}
            
        results = {}
        latest_month = df['month_year'].max()
        latest_data = df[df['month_year'] == latest_month].sort_values('total_updates', ascending=False)
        results['state_rankings'] = latest_data[['state', 'total_updates']]
        
        national_agg = df.groupby('month_year')['total_updates'].mean().reset_index()
        results['national_avg_trend'] = national_agg
        
        all_months = sorted(df['month_year'].unique())
        if len(all_months) >= 12:
            yoy_results = []
            for state in df['state'].unique():
                state_df = df[df['state'] == state].set_index('month_year')
                for month in all_months:
                    prev_year_month = month - 12
                    if prev_year_month in state_df.index and month in state_df.index:
                        current = state_df.loc[month, 'total_updates']
                        previous = state_df.loc[prev_year_month, 'total_updates']
                        yoy_change = ((current - previous) / previous) * 100 if previous != 0 else 0
                        yoy_results.append({
                            'state': state,
                            'month': month,
                            'current': current,
                            'previous': previous,
                            'yoy_growth': yoy_change
                        })
            results['yoy_analysis'] = pd.DataFrame(yoy_results)
        
        if len(all_months) >= 2:
            prev_month = all_months[-2]
            curr_month = all_months[-1]
            comparison_data = []
            national_total_curr = df[df['month_year'] == curr_month]['total_updates'].sum()
            national_total_prev = df[df['month_year'] == prev_month]['total_updates'].sum()
            national_growth = ((national_total_curr - national_total_prev) / national_total_prev) * 100 if national_total_prev != 0 else 0
            
            for state in df['state'].unique():
                state_curr = df[(df['state'] == state) & (df['month_year'] == curr_month)]['total_updates'].sum()
                state_prev = df[(df['state'] == state) & (df['month_year'] == prev_month)]['total_updates'].sum()
                state_growth = ((state_curr - state_prev) / state_prev) * 100 if state_prev != 0 else 0
                comparison_data.append({
                    'state': state,
                    'state_growth': state_growth,
                    'national_growth': national_growth,
                    'is_outperforming': state_growth > national_growth
                })
            results['state_vs_national'] = pd.DataFrame(comparison_data)
        return results

    def train_model(self, df):
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
        rmse = np.sqrt(np.mean((y_test - preds) ** 2))
        model_info = {'train_samples': len(X_train), 'test_samples': len(X_test), 'rmse': rmse, 'factors': {}}
        
        print(f"News Correlation Model Trained:")
        print(f"  - Samples: {len(X_train)} train, {len(X_test)} test")
        print(f"  - RMSE: {rmse:,.0f}")
        
        if self.model.coef_ is not None:
            print("\nNews Factor Correlations:")
            for feat, coef in zip(feature_cols, self.model.coef_):
                display_name = feat.replace('_news_count', ' News').replace('_', ' ').title()
                impact = "STRONG +" if coef > 1000 else ("Weak +" if coef > 0 else ("STRONG -" if coef < -1000 else "Weak -"))
                model_info['factors'][display_name] = {'impact': impact, 'coefficient': coef}
                print(f"  {display_name}: {impact}")
        
        correlations = df[feature_cols + [target_col]].corr()[target_col].drop(target_col)
        top_corr = correlations.abs().idxmax()
        top_corr_name = top_corr.replace('_news_count', ' News').replace('_', ' ').title()
        model_info['strongest_correlation'] = f"{top_corr_name} ({correlations[top_corr]:.2%})"
        return model_info

if __name__ == "__main__":
    dates = pd.date_range("2025-01-01", periods=6, freq="M")
    aadhar_data = pd.DataFrame({'date': dates, 'state': ['Andhra Pradesh'] * 6, 'total_updates': [100, 150, 200, 130, 250, 300]})
    news_data = pd.DataFrame({'published': dates, 'region': ['Andhra Pradesh'] * 6, 'query': ['jobs'] * 6})
    analyzer = MigrationAnalyzer()
    df = analyzer.prepare_data(aadhar_data, news_data)
    print("Prepared Data:")
    print(df.head())
    trends = analyzer.calculate_monthly_trends(df)
    print("\nTrend Data:")
    print(trends)
