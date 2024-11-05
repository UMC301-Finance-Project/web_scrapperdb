import os
import sqlite3
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Custom modules
import rss_scrapper

class StockNewsExtractor:
    def __init__(self, output_dir='stock_news'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)     
        self.news_df = rss_scrapper.fetch_news_for_tickers(rss_scrapper.ticker_aliases)
        print("News Dataframe created")

    def extract_and_save_news_summaries(self, ticker):
        """Extract and save news summaries for a given stock ticker."""
        # Fetch articles
        articles = self.news_df[self.news_df['ticker'] == ticker]
        return articles

class NewsDatabase:
    def __init__(self, db_name='stock_news.db'):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
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
            sentiment REAL,
            embedding BLOB
        )
        ''')
        self.connection.commit()

    def insert_data(self, ticker, headline, published_date, url):
        """Insert a news article into the database."""
        # Generate sentiment and embedding
        sentiment = -2  # Placeholder; replace with sentiment analysis if needed
        embedding = self.embedding_model.encode(headline)
        
        # Insert into SQLite database
        self.cursor.execute('''
        INSERT INTO news_articles (stock_symbol, headline, published_date, url, sentiment, embedding)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (ticker, headline, published_date, url, sentiment, embedding.tobytes()))
        self.connection.commit()
        print(f"Inserted data for {ticker} into SQLite")

    def close(self):
        """Commit changes and close the database connection."""
        self.connection.commit()
        self.connection.close()

def get_nifty_50_tickers():
    """Returns a list of NIFTY 50 stock tickers."""
    return rss_scrapper.ticker_aliases.keys()

def main():
    # Initialize the StockNewsExtractor
    extractor = StockNewsExtractor()
    ndb = NewsDatabase()
    
    # Extract and save news summaries for each ticker
    for ticker in get_nifty_50_tickers():
        news_df = extractor.extract_and_save_news_summaries(ticker)
        for index, row in news_df.iterrows():
            ndb.insert_data(row['ticker'], row['title'], row['pub_date'], row['link'])
    
    # Close the database connection
    ndb.close()

if __name__ == "__main__":
    main()
