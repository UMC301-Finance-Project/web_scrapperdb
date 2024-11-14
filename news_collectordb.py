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
import expert_final
import copy

# Initialize tqdm progress bar for sentiment analysis
tqdm.pandas(desc="Analyzing Sentiment")

class StockNewsExtractor:
    def __init__(self, output_dir='stock_news'):
        """Initializes the extractor, creates output directory, and fetches news data."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.news_df = rss_scrapper.fetch_news_for_tickers(rss_scrapper.ticker_aliases)
        print("News DataFrame created")

    def extract_and_save_news_summaries(self, ticker):
        """Extract and return news summaries for a specific stock ticker."""
        return self.news_df[self.news_df['stock_symbol'] == ticker]

class FirestoreUploader:
    def __init__(self):
        """Initializes Firestore connection using Firebase credentials."""
        # Ensure Firebase is only initialized once
        if not firebase_admin._apps:
            cred = credentials.Certificate('news_db_secrets.json')  # Update with Firebase service account key
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()

    def upload_data(self, db_path='stock_news.db'):
        """Upload data from SQLite database to Firestore."""
        # Connect to the SQLite database
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        
        # Fetch all records from the database
        cursor.execute("SELECT * FROM news_articles")
        rows = cursor.fetchall()
        
        # Get column names for Firestore document fields
        column_names = [description[0] for description in cursor.description]
        
        count = 0
        # Upload each row as a Firestore document
        for row in rows:
            doc_data = dict(zip(column_names, row))  # Map column names to row values
            doc_id = str(count)  # Unique document ID for Firestore
            count += 1
            self.db.collection("stock_news").document(doc_id).set(doc_data)
            print(f"Uploaded document ID: {doc_id} to Firestore.")

        # Close SQLite connection
        connection.close()
        print("Uploaded all data to Firestore and closed the SQLite connection.")
    
    def clear_collection(self, collection_name="stock_news"):
        """Delete all documents in the specified Firestore collection."""
        collection_ref = self.db.collection(collection_name)
        batch_size = 10  # Set batch size for deleting documents in chunks

        def delete_batch(batch_ref):
            """Deletes a batch of documents."""
            for doc in batch_ref:
                doc.reference.delete()
            print(f"Deleted {len(batch_ref)} documents.")

        # Continuously delete documents in batches until the collection is empty
        while True:
            docs = list(collection_ref.limit(batch_size).stream())
            if not docs:
                break
            delete_batch(docs)
        print(f"Cleared all documents from '{collection_name}' collection.")

def get_nifty_50_tickers():
    """Returns a list of NIFTY 50 stock tickers."""
    return rss_scrapper.ticker_aliases.keys()

def calculate_means_per_ticker(df: pd.DataFrame):
    """Calculate mean sentiment score for each aspect per stock ticker, excluding zero values."""
    ret_dict = dict()
    aspects = absa.aspects  # List of sentiment aspects to analyze

    def mean_ignore_zeros(series):
        """Calculates mean of a series, ignoring zeros."""
        return series[series != 0].mean() if not series[series != 0].empty else 0

    # Calculate mean sentiment scores for each aspect, excluding zero values
    aspect_averages = df.groupby('stock_symbol')[aspects].transform(mean_ignore_zeros)
    aspect_averages.columns = [f"{aspect}_average" for aspect in aspects]  # Rename columns
    
    # Concatenate original DataFrame with the computed averages
    df = pd.concat([df, aspect_averages], axis=1)

    # Create a dictionary with average scores per ticker
    for ticker in list(get_nifty_50_tickers()):
        ret_dict[ticker] = {
            aspect: float(df[df['stock_symbol'] == ticker][aspect + '_average'].unique()) for aspect in aspects
        }
    
    return ret_dict

def save_to_sqlite_and_firestore(df, db_path='result_stock_news.db', table_name='stock_news_results', collection_name="stock_news_results"):
    """Save DataFrame to SQLite and upload it to Firebase Firestore."""
    with sqlite3.connect(db_path) as conn:
        # Save DataFrame to SQLite database
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"DataFrame saved to '{db_path}' as table '{table_name}'.")

    # Initialize FirestoreUploader
    uploader = FirestoreUploader()
    uploader.clear_collection(collection_name)  # Clear collection before uploading
    
    # Upload each row as a Firestore document
    for idx, row in df.iterrows():
        doc_id = str(idx)  # Use row index as document ID
        doc_data = row.to_dict()  # Convert row to dictionary
        uploader.db.collection(collection_name).document(doc_id).set(doc_data)
        print(f"Uploaded document ID: {doc_id} to Firestore collection '{collection_name}'.")
    
    print("All data uploaded to Firebase Firestore.")

def main():
    """Main function to process, analyze, and store stock news data."""
    # Initialize FirestoreUploader and clear old data
    uploader = FirestoreUploader()
    uploader.clear_collection()

    if not os.path.exists('stock_news.db'):
        print("bonjour")  # Placeholder for debugging
        extractor = StockNewsExtractor()
        all_news_df = pd.DataFrame()

        # Extract and save news summaries for each ticker
        for ticker in tqdm(get_nifty_50_tickers(), desc="News Saver"):
            news_df = extractor.extract_and_save_news_summaries(ticker)
            all_news_df = pd.concat([all_news_df, news_df], ignore_index=True)

        # Add sentiment analysis columns to DataFrame
        sentiment_analyser = SentimentAnalyser(chaspects=absa.aspects)
        # Process each article with sentiment analysis
        for index, row in tqdm(all_news_df.iterrows(), total=len(all_news_df), desc="Processing Articles"):
            aspect_scores = sentiment_analyser.analyze_sentiment(row['title'])  # Analyze sentiment
            for aspect, score in aspect_scores.items():
                all_news_df.at[index, aspect] = score  # Add scores to DataFrame

        # Save processed data to SQLite
        with sqlite3.connect('stock_news.db') as conn:
            all_news_df.to_sql('news_articles', conn, if_exists='replace', index=False)
        print("DataFrame saved to stock_news.db")

    uploader.upload_data()
    
    # Load data from SQLite for further processing
    connection = sqlite3.connect('stock_news.db')
    dfr = pd.read_sql("SELECT * FROM news_articles;", connection)
    parameter = calculate_means_per_ticker(dfr)
    connection.close()

    # Prepare expert model analysis
    experts = list(absa.aspects) + ['Trendlyne']
    result_df = pd.DataFrame(columns=experts)  # Initialize empty DataFrame

    # Process each ticker using the expert model
    for ticker in list(parameter.keys()):
        model = expert_final.MultiplicativeWeightsExpert(ticker, experts, parameter)  # Initialize expert model
        aspect_weights = model.weights  # Retrieve weights
        rating = model.model_rating  # Retrieve model rating
        verdict = model.verdict  # Retrieve model verdict

        # Separate Trendlyne value
        trendline = aspect_weights[-1]
        aspect_weights = aspect_weights[:-1]

        # Create a row for each ticker
        row = {expert: parameter[ticker][expert] if expert != 'Trendlyne' else trendline for i, expert in enumerate(experts)}
        row['stock_symbol'] = ticker
        row['rating'] = rating
        row['verdict'] = verdict

        # Append to result DataFrame
        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
    
    # Save final results to SQLite and Firestore
    save_to_sqlite_and_firestore(result_df)

    return result_df

if __name__ == "__main__":
    main()
