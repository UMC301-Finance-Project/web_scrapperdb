import pandas as pd
import time
from typing import List, Dict
import feedparser
from urllib.parse import quote
from tqdm import tqdm  # For displaying progress bars during processing

# Define NIFTY-50 stock tickers with their respective aliases for search
ticker_aliases = {
    'ADANIPORTS.NS': ['Adani Ports'],
    'APOLLOHOSP.NS': ['Apollo Hospitals'],
    'ASIANPAINT.NS': ['Asian Paints'],
    'AXISBANK.NS': ['Axis Bank'],
    'BAJAJ-AUTO.NS': ['Bajaj Auto'],
    'BAJFINANCE.NS': ['Bajaj Finance'],
    'BAJAJFINSV.NS': ['Bajaj Finserv'],
    'BHARTIARTL.NS': ['Bharti Airtel'],
    'BPCL.NS': ['Bharat Petroleum', 'BPCL'],
    'BRITANNIA.NS': ['Britannia'],
    'CIPLA.NS': ['Cipla'],
    'COALINDIA.NS': ['Coal India'],
    'DIVISLAB.NS': ["Divi's Labs"],
    'DRREDDY.NS': ["Dr. Reddy's", "Dr. Reddy's Labs"],
    'EICHERMOT.NS': ['Eicher Motors'],
    'GRASIM.NS': ['Grasim Industries', 'Grasim'],
    'HCLTECH.NS': ['HCL Technologies', 'HCL Tech'],
    'HDFC.NS': ['HDFC'],
    'HDFCBANK.NS': ['HDFC Bank'],
    'HDFCLIFE.NS': ['HDFC Life'],
    'HEROMOTOCO.NS': ['Hero MotoCorp'],
    'HINDALCO.NS': ['Hindalco'],
    'HINDUNILVR.NS': ['Hindustan Unilever'],
    'ICICIBANK.NS': ['ICICI Bank'],
    'INDUSINDBK.NS': ['IndusInd Bank'],
    'INFY.NS': ['Infosys'],
    'ITC.NS': ['ITC'],
    'JSWSTEEL.NS': ['JSW Steel'],
    'KOTAKBANK.NS': ['Kotak Mahindra Bank', 'Kotak Bank'],
    'LT.NS': ['Larsen & Toubro', 'L&T'],
    'MARUTI.NS': ['Maruti Suzuki'],
    'M&M.NS': ['Mahindra & Mahindra'],
    'NESTLEIND.NS': ['Nestle India'],
    'NTPC.NS': ['NTPC'],
    'ONGC.NS': ['Oil & Natural Gas Corporation', 'ONGC'],
    'POWERGRID.NS': ['Power Grid'],
    'RELIANCE.NS': ['Reliance Industries', 'Reliance'],
    'SBILIFE.NS': ['SBI Life'],
    'SBIN.NS': ['State Bank of India', 'SBI'],
    'SHREECEM.NS': ['Shree Cement'],
    'SUNPHARMA.NS': ['Sun Pharmaceutical'],
    'TATACONSUM.NS': ['Tata Consumer Products', 'Tata Consumer'],
    'TATAMOTORS.NS': ['Tata Motors'],
    'TATASTEEL.NS': ['Tata Steel'],
    'TCS.NS': ['Tata Consultancy Services', 'TCS'],
    'TECHM.NS': ['Tech Mahindra'],
    'TITAN.NS': ['Titan Company'],
    'ULTRACEMCO.NS': ['UltraTech Cement'],
    'UPL.NS': ['UPL'],
    'WIPRO.NS': ['Wipro']
}

def fetch_rss_articles(alias: str) -> List[Dict]:
    """
    Fetch articles from Google News RSS feed for a specific alias, with retry logic for handling errors.
    
    Parameters:
    alias (str): A company alias to search for in the RSS feed.
    
    Returns:
    List[Dict]: A list of dictionaries, each containing information about an article.
    """
    # URL-encode the alias for safe inclusion in the search URL
    encoded_alias = quote(f'"{alias}"')
    url = f'https://news.google.com/rss/search?q={encoded_alias}&hl=en-IN&gl=IN&ceid=IN:en'
    
    # Set retry parameters for network errors
    retries = 3  # Number of retries
    delay = 2  # Delay between retries in seconds

    # Retry loop for fetching RSS feed
    for attempt in range(retries):
        try:
            # Parse the RSS feed with feedparser
            feed = feedparser.parse(url)
            articles = []   
            # Loop through feed entries and extract relevant data
            for entry in feed.entries:
                title = entry.title
                link = entry.link
                pub_date = entry.published
                # Append article details as a dictionary
                articles.append({'title': title, 'url': link, 'published_date': pub_date, 'alias': alias})
            return articles  # Return list of articles on success

        except Exception as e:
            # Print error message and retry after a delay
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)

    # Return empty list if all retry attempts fail
    print(f"Failed to fetch articles for alias '{alias}' after {retries} attempts.")
    return []

def fetch_news_for_tickers(ticker_aliases: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Fetch news articles for each ticker symbol and its aliases, using progress tracking.
    
    Parameters:
    ticker_aliases (Dict[str, List[str]]): A dictionary where each key is a stock ticker,
                                           and each value is a list of aliases for that ticker.
    
    Returns:
    pd.DataFrame: A DataFrame containing news articles for each alias, with stock symbols added.
    """
    all_articles = []
    # Loop over each ticker and its aliases, showing progress with tqdm
    for ticker, aliases in tqdm(ticker_aliases.items(), desc="Tickers", unit="ticker"):
        for alias in tqdm(aliases, desc=f"Processing aliases for {ticker}", leave=False, unit="alias"):
            # Fetch articles for each alias
            articles = fetch_rss_articles(alias)
            # Add stock ticker to each article and collect it
            for article in articles:
                article['stock_symbol'] = ticker
                all_articles.append(article)
    
    # Create a DataFrame from the list of articles and remove rows with missing data
    news_df = pd.DataFrame(all_articles)
    news_df = news_df.dropna()  # Drop any rows with NaN values
    return news_df
