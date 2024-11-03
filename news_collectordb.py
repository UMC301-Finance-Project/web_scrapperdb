import os
import re
import glob
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import finnhub
import datetime as dt
import shutil
import json

class StockNewsExtractor:
    def __init__(self, api_key=None, output_dir='stock_news'):
        """
        Initialize the Stock News Extractor

        Args:
            api_key (str, optional): Finnhub API key. If not provided, will look for environment variable.
            output_dir (str, optional): Directory to save news files
        """
        self.api_key = api_key or os.environ.get('FINNHUB_API_KEY')
        if not self.api_key:
            raise ValueError("No Finnhub API key found. Set FINNHUB_API_KEY environment variable or pass key directly.")
        
        # Delete existing directory if it exists
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        self.output_dir = output_dir
        # Ensure output directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'summaries'), exist_ok=True)

    def _is_valid_news_article(self, article):
        """
        Validate if the article is a legitimate news piece

        Args:
            article (dict): News article dictionary

        Returns:
            bool: True if article is valid, False otherwise
        """
        ad_keywords = [
            'advertisement', 'sponsored', 'promoted', 'paid content',
            'this is an advertisement', 'ads by', 'sponsored by',
            'zacks.com', 'financial research', 'stock market analysis'
        ]
        
        if not all(key in article for key in ['headline', 'summary', 'url']):
            return False
        
        headline_lower = article['headline'].lower()
        summary_lower = article.get('summary', '').lower()
        
        if any(keyword in headline_lower or keyword in summary_lower for keyword in ad_keywords):
            return False
        
        if len(summary_lower.strip()) < 20:
            return False
        
        spam_patterns = [
            r'\b(buy|sell)\s+now\b',
            r'limited\s+time\s+offer',
            r'exclusive\s+deal',
            r'\$\d+\s*profit'
        ]
        
        if any(re.search(pattern, headline_lower) or re.search(pattern, summary_lower) for pattern in spam_patterns):
            return False
        
        return True

    def extract_and_save_news_summaries(self, ticker_symbol, days_back=30):
        """
        Extract news, save to CSV, and save summaries to a text file

        Args:
            ticker_symbol (str): Stock ticker symbol
            days_back (int, optional): Number of days to look back

        Returns:
            list: List of valid news summaries with dates
        """
        finnhub_client = finnhub.Client(api_key=self.api_key)
        today = datetime.now(dt.UTC)
        past_date = today - timedelta(days=days_back)
        
        news = finnhub_client.company_news(
            ticker_symbol, 
            _from=past_date.strftime('%Y-%m-%d'), 
            to=today.strftime('%Y-%m-%d')
        )
        
        if not news:
            print(f"No news found for ticker {ticker_symbol}")
            return []
        
        valid_news = [article for article in news if self._is_valid_news_article(article)]
        if not valid_news:
            print(f"No valid news articles found for ticker {ticker_symbol}")
            return []
        
        news_df = pd.DataFrame(valid_news)
        news_df['ticker'] = ticker_symbol
        news_df['readable_datetime'] = pd.to_datetime(news_df['datetime'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filepath = os.path.join(self.output_dir, f"{ticker_symbol}_news_{timestamp}.csv")
        news_df.to_csv(csv_filepath, index=False, encoding='utf-8')
        print(f"News saved to {csv_filepath}")
        
        self._save_summaries(news_df, ticker_symbol, timestamp)
        return news_df

    def _save_summaries(self, news_df, ticker_symbol, timestamp):
        """Save news summaries to a text file."""
        detailed_summaries = [
            f"[{row['readable_datetime']}] {row['headline']}: {row['summary']}"
            for _, row in news_df.iterrows()
        ]
        
        summary_filepath = os.path.join(self.output_dir, 'summaries', f"{ticker_symbol}_summaries_{timestamp}.txt")
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            f.write(f"News Summaries for {ticker_symbol} on {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            for idx, summary in enumerate(detailed_summaries, 1):
                f.write(f"{idx}. {summary}\n\n")
        
        print(f"Summaries saved to {summary_filepath}")
        print(f"Total valid news articles: {len(detailed_summaries)}")

class NewsDatabase:
    def __init__(self, db_name='stock_news.db'):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        # Create the table if it doesn't exist
        self.create_table()

    def create_table(self):
        """Create a table for news articles if it doesn't exist."""
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_symbol TEXT,
            headline TEXT,
            published_date TEXT,
            url TEXT,
            sentiment REAL
        )
        ''')
    
    def insert_data(self, ticker, headline, published_date, url):
        """Insert a news article into the database."""
        self.cursor.execute('''
        INSERT INTO news_articles (stock_symbol, headline, published_date, url)
        VALUES (?, ?, ?, ?)
        ''', (ticker, headline, published_date, url))
    
    def fetch_data(self):
        """Fetch all news articles from the database."""
        return self.cursor.execute('SELECT * FROM news_articles').fetchall()
    
    def fetch_data_by_ticker(self, ticker):
        """Fetch news articles for a specific stock ticker."""
        return self.cursor.execute('SELECT * FROM news_articles WHERE stock_symbol = ?', (ticker,)).fetchall()
    
    def close(self):
        """Commit changes and close the database connection."""
        self.connection.commit()
        self.connection.close()

def get_nifty_50_tickers():
    """Returns a list of NIFTY 50 stock tickers."""
    return [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'TSLA', 'NVDA', 'JPM'
    ]

def main():

# Load secrets from secrets.json
    with open('secrets.json') as f:
        secrets = json.load(f)

    api_key = secrets.get('FINNHUB_API_KEY')
    try:
        extractor = StockNewsExtractor(api_key)
        nifty_tickers = get_nifty_50_tickers()
        db = NewsDatabase()
        
        for ticker in nifty_tickers:
            ticker_news = extractor.extract_and_save_news_summaries(ticker, days_back=7)
            for idx,its in ticker_news.iterrows():
                db.insert_data(its['ticker'], its['headline'], its['readable_datetime'], its['url'])  # Replace with actual headline fetching
            
        db.close()
    
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
