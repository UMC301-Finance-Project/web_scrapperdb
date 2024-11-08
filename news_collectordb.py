import os
import sqlite3
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Custom modules
import rss_scrapper
import absa
from absa import SentimentAnalyser
class StockNewsExtractor:
    def __init__(self, output_dir='stock_news'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)     
        self.news_df = rss_scrapper.fetch_news_for_tickers(rss_scrapper.ticker_aliases)
        print("News Dataframe created")

    def extract_and_save_news_summaries(self, ticker):
        """Extract and save news summaries for a given stock ticker."""
        articles = self.news_df[self.news_df['ticker'] == ticker]
        return articles

class NewsDatabase:
    def __init__(self, db_name='stock_news.db', aspects=None):
        """Initialize the NewsDatabase with a specified SQLite database name and aspects list."""
        
        # Delete the existing database file if it exists
        if os.path.exists(db_name):
            os.remove(db_name)
        
        # Connect to SQLite database
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define the list of aspects
        self.aspects = aspects if aspects else ['default_aspect']  # Provide a default list or handle if aspects is None
        
        # Initialize the SentimentAnalyser with the specified aspects
        self.sentiment_analyser = SentimentAnalyser(aspects=self.aspects)
        
        # Create the table with columns for each aspect
        self.create_table()

    def create_table(self):
        """Create a table for news articles with columns for each aspect's score."""
        # Basic schema with fixed columns
        columns = '''
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_symbol TEXT,
            headline TEXT,
            published_date TEXT,
            url TEXT,
            embedding BLOB,
        '''
        
        # Add a column for each aspect using the aspect's name directly
        for aspect in range(len(absa.aspects)-1):
            columns += f'{absa.aspects[aspect]} REAL,\n'
        
        columns+=f'{absa.aspects[-1]} REAL\n'
        # Create the table if it doesn't already exist
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS news_articles ({columns})
        ''')
        self.connection.commit()
        print(f"Table created with columns for each aspect with this query: {columns}")

    def insert_data(self, ticker, headline, published_date, url):
        """Insert data into the database, including scores for each aspect."""
        # General sentiment placeholder (update as needed for full sentiment analysis)
        
        # Generate embedding for the headline
        embedding = self.embedding_model.encode(headline)

        # Generate aspect-based sentiment scores for each aspect
        aspect_scores = self.sentiment_analyser.analyze_sentiment(headline)

        # Prepare SQL for dynamic insertion based on aspects
        aspect_columns = ', '.join(self.aspects)  # Create columns list for SQL
        placeholders = ', '.join(['?'] * (4 + len(self.aspects)))  # Placeholder for SQL
        values = [ticker, headline, published_date, url]+ list(aspect_scores.values())
        
        print(values, type(values))
        print(aspect_columns, type(aspect_columns))
        # Insert into the database
        self.cursor.execute(f'''
            INSERT INTO news_articles (stock_symbol, headline, published_date, url, {aspect_columns})
            VALUES ({placeholders})
        ''', values)
        self.connection.commit()
        print(f"Data inserted for {ticker} with aspect scores.")

    def close(self):
        """Commit changes and close the database connection."""
        self.connection.commit()
        self.connection.close()
        print("Database connection closed.")


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
