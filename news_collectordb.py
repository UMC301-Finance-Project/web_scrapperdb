import os
import re
import shutil
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import json
import sqlite3
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
import feedparser
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
        cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')  # Path to your Firebase credentials file
        if not cred_path or not os.path.exists(cred_path):
            raise ValueError("Firebase credentials file path is not set or does not exist.")
        
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
    
    def insert_data(self, ticker, headline, published_date, url):
        """Insert a news article into the database and Firestore."""
        # Generate sentiment and embedding
        sentiment = -2
        embedding = self.embedding_model.encode(headline)
        
        # Insert into SQLite database
        self.cursor.execute('''
        INSERT INTO news_articles (stock_symbol, headline, published_date, url, sentiment, embedding)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (ticker, headline, published_date, url, sentiment, embedding.tobytes()))
        self.connection.commit()

        # Push to Firestore
        self.push_to_firestore(ticker, headline, published_date, url, sentiment)

    def push_to_firestore(self, ticker, headline, published_date, url, sentiment):
        """Push news article data to Firestore."""
        data = {
            'stock_symbol': ticker,
            'headline': headline,
            'published_date': published_date,
            'url': url,
            'sentiment': sentiment
        }
        self.firestore_db.collection('news_articles').add(data)
        print(f"Document added to Firestore for {ticker}")

    # Other existing methods ...

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
