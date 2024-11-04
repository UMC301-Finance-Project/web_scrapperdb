# Standard libraries
import os
import re
import shutil
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from urllib.parse import quote

# External libraries
import pandas as pd
import numpy as np
import json
import sqlite3
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
import feedparser

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
        # Original table with added embedding column
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
        sentiment = -2
        embedding = self.embedding_model.encode(headline)
        
        self.cursor.execute('''
        INSERT INTO news_articles (stock_symbol, headline, published_date, url, sentiment, embedding)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (ticker, headline, published_date, url, sentiment, embedding.tobytes()))
        self.connection.commit()
    
    def fetch_data(self):
        """Fetch all news articles from the database."""
        return self.cursor.execute('SELECT * FROM news_articles').fetchall()
    
    def fetch_data_by_ticker(self, ticker):
        """Fetch news articles for a specific stock ticker."""
        return self.cursor.execute(
            'SELECT * FROM news_articles WHERE stock_symbol = ?', 
            (ticker,)
        ).fetchall()
    
    def close(self):
        """Commit changes and close the database connection."""
        self.connection.commit()
        self.connection.close()

    # New RAG-related methods
    def find_similar_news(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find similar news articles using RAG."""
        query_embedding = self.embedding_model.encode(query)
        
        # Fetch all articles
        self.cursor.execute('''
            SELECT id, stock_symbol, headline, published_date, url, sentiment, embedding 
            FROM news_articles
        ''')
        articles = self.cursor.fetchall()
        
        # Calculate similarities
        similarities = []
        for article in articles:
            article_embedding = np.frombuffer(article[6], dtype=np.float32)
            similarity = np.dot(query_embedding, article_embedding) / \
                        (np.linalg.norm(query_embedding) * np.linalg.norm(article_embedding))
            similarities.append((similarity, article))
        
        # Sort and get top results
        similarities.sort(reverse=True)
        top_results = similarities[:top_k]
        
        # Format results
        results = []
        for similarity, article in top_results:
            results.append({
                'id': article[0],
                'stock_symbol': article[1],
                'headline': article[2],
                'published_date': article[3],
                'url': article[4],
                'sentiment': article[5],
                'similarity_score': float(similarity)
            })
        
        return results

    def get_sentiment_by_ticker(self, ticker: str) -> Dict:
        """Get sentiment statistics for a specific ticker."""
        self.cursor.execute('''
            SELECT 
                AVG(sentiment) as avg_sentiment,
                COUNT(*) as total_articles
            FROM news_articles
            WHERE stock_symbol = ?
        ''', (ticker,))
        
        stats = self.cursor.fetchone()
        return {
            'ticker': ticker,
            'average_sentiment': stats[0],
            'total_articles': stats[1]
        }

    def batch_insert_data(self, news_items: List[Tuple[str, str, str, str]]):
        """Batch insert multiple news articles."""
        for ticker, headline, published_date, url in news_items:
            self.insert_data(ticker, headline, published_date, url)

    def get_latest_news(self, ticker: str = None, limit: int = 10):
        """Get the most recent news articles, optionally filtered by ticker."""
        if ticker:
            self.cursor.execute('''
                SELECT * FROM news_articles 
                WHERE stock_symbol = ?
                ORDER BY published_date DESC 
                LIMIT ?
            ''', (ticker, limit))
        else:
            self.cursor.execute('''
                SELECT * FROM news_articles 
                ORDER BY published_date DESC 
                LIMIT ?
            ''', (limit,))
        return self.cursor.fetchall()

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
