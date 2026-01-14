import feedparser
import pandas as pd
import urllib.parse
from datetime import datetime
import time

def fetch_google_news(query, region=""):
    full_query = f"{query} {region}".strip()
    encoded_query = urllib.parse.quote(full_query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    print(f"Fetching news for: {full_query}")
    try:
        feed = feedparser.parse(url)
    except Exception as e:
        print(f"Error fetching feed: {e}")
        return pd.DataFrame()
    
    news_items = []
    if not feed.entries:
        print(f"No entries found for {full_query}")
        return pd.DataFrame()
        
    for entry in feed.entries:
        try:
            pub_date = pd.to_datetime(entry.published)
        except:
            pub_date = datetime.now()
            
        news_items.append({
            'title': entry.title,
            'link': entry.link,
            'published': pub_date,
            'summary': entry.summary if 'summary' in entry else '',
            'region': region,
            'query': query
        })
        
    df = pd.DataFrame(news_items)
    return df

def fetch_all_news(regions, queries=['jobs', 'migration', 'industry', 'hiring']):
    all_news = []
    for region in regions:
        for q in queries:
            df = fetch_google_news(q, region)
            if not df.empty:
                all_news.append(df)
            time.sleep(1)
            
    if all_news:
        return pd.concat(all_news, ignore_index=True)
    return pd.DataFrame()

if __name__ == "__main__":
    regions = ["Andhra Pradesh", "Telangana"]
    df = fetch_all_news(regions, queries=['jobs'])
    print(f"Fetched {len(df)} articles.")
    if not df.empty:
        print(df[['title', 'published', 'region']].head())
