import os
import sqlite3
from datetime import datetime
from sentence_transformers import SentenceTransformer
import firebase_admin
from firebase_admin import credentials, firestore

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
        self.firestore_db = self.initialize_firestore()  # Initialize Firestore
        # Create the table if it doesn't exist
        self.create_table()

    def initialize_firestore(self):
        """Initialize Firestore with Firebase credentials."""
        cred_path = "./secrets.json"        
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        return firestore.client()

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

    def backup_and_recreate_news_collection(self):
            """Create a backup of the existing news_articles collection and reset it."""
            db = self.firestore_db
            news_articles_ref = db.collection('news_articles')
            backup_ref = db.collection('news_articles_bak')

            # Check if the backup collection exists and delete it if so
            backup_docs = list(backup_ref.limit(1).stream())
            if backup_docs:
                self.delete_collection(backup_ref)

            # Copy documents to backup
            docs = news_articles_ref.stream()
            for doc in docs:
                backup_ref.document(doc.id).set(doc.to_dict())

            # Delete the old news_articles collection
            self.delete_collection(news_articles_ref)


    def delete_collection(self, collection_ref, batch_size=100):
        """Deletes all documents in a collection."""
        docs = collection_ref.limit(batch_size).stream()
        deleted = 0
        for doc in docs:
            print(f"Deleting doc {doc.id} from {collection_ref.id}")
            doc.reference.delete()
            deleted += 1
        if deleted >= batch_size:
            return self.delete_collection(collection_ref, batch_size)

    def insert_data(self, ticker, headline, published_date, url):
        """Insert a news article into the database and Firestore."""
        # Generate sentiment and embedding
        sentiment = -2  # Placeholder; you may replace it with sentiment analysis
        embedding = self.embedding_model.encode(headline)

        # Store the embedding in Firestore as a list
        embedding_list = embedding.tolist()  # Convert to list for Firestore

        # Insert into SQLite database (optional)
        self.cursor.execute('''
        INSERT INTO news_articles (stock_symbol, headline, published_date, url, sentiment)
        VALUES (?, ?, ?, ?, ?)
        ''', (ticker, headline, published_date, url, sentiment))
        self.connection.commit()

        # Push to Firestore
        self.push_to_firestore(ticker, headline, published_date, url, sentiment, embedding_list)

    def push_to_firestore(self, ticker, headline, published_date, url, sentiment, embedding):
        """Push news article data to Firestore, including the embedding."""
        data = {
            'stock_symbol': ticker,
            'headline': headline,
            'published_date': published_date,
            'url': url,
            'sentiment': sentiment,
            'embedding': embedding  # Store embedding as a list
        }
        self.firestore_db.collection('news_articles').add(data)
        print(f"Document added to Firestore for {ticker}")

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
    
    # Backup and reset the Firestore collection before inserting new data
    ndb.backup_and_recreate_news_collection()
    
    # Extract and save news summaries for each ticker
    for ticker in get_nifty_50_tickers():
        news_df = extractor.extract_and_save_news_summaries(ticker)
        for index, row in news_df.iterrows():
            ndb.insert_data(row['ticker'], row['title'], row['pub_date'], row['link'])
    
    # Close the database connection
    ndb.close()

if __name__ == "__main__":
    main()
