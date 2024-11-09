import os
import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
import firebase_admin
from firebase_admin import credentials, firestore
from tqdm import tqdm
# Custom modules
import rss_scrapper
import absa
from absa import SentimentAnalyser

from tqdm import tqdm
tqdm.pandas(desc="Analyzing Sentiment")

class StockNewsExtractor:
    def __init__(self, output_dir='stock_news'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.news_df = rss_scrapper.fetch_news_for_tickers(rss_scrapper.ticker_aliases)
        print("News DataFrame created")

    def extract_and_save_news_summaries(self, ticker):
        """Extract and return news summaries for a given stock ticker."""
        return self.news_df[self.news_df['ticker'] == ticker]

class FirestoreUploader:
    def __init__(self):
        # Initialize Firestore connection
        if not firebase_admin._apps:
            cred = credentials.Certificate('secrets.json')  # Update with your Firebase service account key path
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()

    def upload_data(self, db_path='stock_news.db'):
        """Upload data from SQLite database to Firebase Firestore."""
        # Connect to SQLite database
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        # Fetch all records from the database
        cursor.execute("SELECT * FROM news_articles")
        rows = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        # Upload each row as a document to Firestore
        for row in rows:
            doc_data = dict(zip(column_names, row))
            doc_id = str(doc_data.get("id"))  # Use a unique field for the document ID
            # Add to Firestore
            self.db.collection("stock_news").document(doc_id).set(doc_data)
            print(f"Uploaded document ID: {doc_id} to Firestore.")

        # Close SQLite connection
        connection.close()
        print("Uploaded all data to Firestore and closed the SQLite connection.")
    
    def clear_collection(self, collection_name="stock_news"):
        """Delete all documents from the specified Firestore collection."""
        collection_ref = self.db.collection(collection_name)
        batch_size = 10  # Set batch size for deleting documents in chunks

        # Function to delete a batch of documents
        def delete_batch(batch_ref):
            for doc in batch_ref:
                doc.reference.delete()
            print(f"Deleted {len(batch_ref)} documents.")

        # Delete all documents in batches
        while True:
            docs = list(collection_ref.limit(batch_size).stream())
            if not docs:
                break
            delete_batch(docs)
        print(f"Cleared all documents from '{collection_name}' collection.")

def get_nifty_50_tickers():
    """Returns a list of NIFTY 50 stock tickers."""
    return rss_scrapper.ticker_aliases.keys()

def main():
    # Initialize the StockNewsExtractor and extract news data
    uploader = FirestoreUploader()
    uploader.clear_collection()
    if not os.path.exists('stock_news.db'):
        extractor = StockNewsExtractor()
        all_news_df = pd.DataFrame()

        # Extract and save news summaries for each ticker
        for ticker in tqdm(get_nifty_50_tickers(), desc="News Saver"):
            news_df = extractor.extract_and_save_news_summaries(ticker)
            
            # Concatenate results into a single DataFrame
            all_news_df = pd.concat([all_news_df, news_df], ignore_index=True)

        # Add sentiment analysis columns to the DataFrame
        sentiment_analyser = SentimentAnalyser(chaspects=absa.aspects)
        # Iterate over aspects with a progress bar
        for aspect in tqdm(absa.aspects, desc="Processing Aspects"):
            # Apply sentiment analysis for each aspect and show progress
            all_news_df[aspect] = all_news_df['title'].progress_apply(sentiment_analyser.analyze_sentiment)

        # Save the DataFrame to an SQLite database file
        with sqlite3.connect('stock_news.db') as conn:
            all_news_df.to_sql('news_articles', conn, if_exists='replace', index=False)
        print("DataFrame saved to stock_news.db")

    # Upload the data to Firestore
    
    uploader.upload_data()

if __name__ == "__main__":
    main()
