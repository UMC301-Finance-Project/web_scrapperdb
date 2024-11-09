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
        return self.news_df[self.news_df['stock_symbol'] == ticker]

class FirestoreUploader:
    def __init__(self):
        # Initialize Firestore connection
        if not firebase_admin._apps:
            cred = credentials.Certificate('news_db_secrets.json')  # Update with your Firebase service account key path
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
        
        count = 0
        # Upload each row as a document to Firestore
        for row in rows:
            doc_data = dict(zip(column_names, row))
            doc_id = str(count)  # Use a unique field for the document ID
            count +=1
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

def calculate_means_per_ticker(df: pd.DataFrame):
    """Calculate the mean sentiment score for each aspect for each stock ticker, excluding zero values."""
    # Get list of aspects from absa
    
    ret_dict = dict()

    aspects = absa.aspects
    
    # Define a custom mean function that ignores 0 values
    def mean_ignore_zeros(series):
            return series[series != 0].mean() if not series[series != 0].empty else 0

    # Group by stock symbol and apply the custom mean function to each aspect
    aspect_averages = df.groupby('stock_symbol')[aspects].transform(mean_ignore_zeros)
        
    # Rename columns to indicate they are averages
    aspect_averages.columns = [f"{aspect}_average" for aspect in aspects]
        
    # Concatenate the original dataframe with the averages
    df = pd.concat([df, aspect_averages], axis=1)

    for ticker in list(get_nifty_50_tickers()):
        ret_dict[ticker] = {
            aspect: float(df[df['stock_symbol']==ticker][aspect+'_average'].unique()) for aspect in aspects
        }
    
    return ret_dict



def save_to_sqlite_and_firestore(df, db_path='result_stock_news.db', table_name='stock_news_results', collection_name="stock_news_results"):
    """Converts the given DataFrame to an SQLite database and uploads it to Firebase Firestore."""
    # Connect to the SQLite database (or create it if it doesn't exist)
    with sqlite3.connect(db_path) as conn:
        # Save the DataFrame to the SQLite database as a new table (or replace if exists)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"DataFrame saved to '{db_path}' as table '{table_name}'.")

    # Initialize FirestoreUploader
    uploader = FirestoreUploader()
    uploader.clear_collection(collection_name)  # Clear the collection before uploading new data
    
    # Upload each row to Firestore
    for idx, row in df.iterrows():
        doc_id = str(idx)  # Use the row index as the document ID
        doc_data = row.to_dict()  # Convert row to dictionary
        uploader.db.collection(collection_name).document(doc_id).set(doc_data)
        print(f"Uploaded document ID: {doc_id} to Firestore collection '{collection_name}'.")
    
    print("All data uploaded to Firebase Firestore.")


def main():
    # Initialize the StockNewsExtractor and extract news data
    uploader = FirestoreUploader()
    uploader.clear_collection()
    if not os.path.exists('stock_news.db'):
        print("bonjour")
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
        for index, row in tqdm(all_news_df.iterrows(), total=len(all_news_df), desc="Processing Articles"):
            # Apply sentiment analysis for each aspect and store results in the respective columns
            aspect_scores = sentiment_analyser.analyze_sentiment(row['title'])
            # Store the aspect scores in the dataframe (one column per aspect)
            for aspect, score in aspect_scores.items():
                all_news_df.at[index, aspect] = score

        # Save the DataFrame to an SQLite database file
        with sqlite3.connect('stock_news.db') as conn:
            all_news_df.to_sql('news_articles', conn, if_exists='replace', index=False)
        print("DataFrame saved to stock_news.db")


    uploader.upload_data()
    # Connect to the SQLite database
    connection = sqlite3.connect('stock_news.db')
    # Execute the SQL queries and load the result into a pandas DataFrame
    dfr = pd.read_sql("SELECT * FROM news_articles;", connection)
    parameter = calculate_means_per_ticker(dfr)

    # Close the connection
    connection.close()

    experts = list(absa.aspects) + ['Trendlyne']
    # Initialize empty DataFrame
    result_df = pd.DataFrame(columns=experts)

    # Create rows for the DataFrame
    for ticker in list(parameter.keys()):
        
        
        # Initialize the model
        model = expert_final.MultiplicativeWeightsExpert(ticker, experts, parameter)
        aspect_weights = model.weights
        rating = model.model_rating
        verdict = model.verdict

        # Assuming 'aspect_weights' is a list that includes weights and trendline as the last value
        trendline = aspect_weights[-1]
        aspect_weights = aspect_weights[:-1]

        # Create a row for the current ticker
        row = {expert: parameter[ticker][expert] if expert != 'Trendlyne' else trendline
               for i, expert in enumerate(experts)}
        row['stock_symbol'] = ticker
        row['rating'] = rating
        row['verdict'] = verdict

        # Append the row to the result dataframe
        result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
    
           
    save_to_sqlite_and_firestore(result_df)

    return result_df
    



if __name__ == "__main__":
    main()
