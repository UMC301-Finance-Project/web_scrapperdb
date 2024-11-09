import pandas as pd
import time
from typing import List, Dict
import feedparser
from urllib.parse import quote
from tqdm import tqdm  # For progress tracking

# Define NIFTY-50 stock tickers with aliases
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
    'DIVISLAB.NS': ['Divi\'s Labs'],
    'DRREDDY.NS': ['Dr. Reddy\'s', 'Dr. Reddy\'s Labs'],
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
    Fetches articles from Google News RSS feed for a given alias, with retry logic.
    """
    encoded_alias = quote(f'"{alias}"')  # URL-encode the alias
    url = f'https://news.google.com/rss/search?q={encoded_alias}&hl=en-IN&gl=IN&ceid=IN:en'
    retries = 3  # Number of retries
    delay = 2  # Delay between retries (in seconds)

    for attempt in range(retries):
        try:
            # Parse RSS feed using feedparser
            feed = feedparser.parse(url)
            articles = []   
            for entry in feed.entries:
                title = entry.title
                link = entry.link
                pub_date = entry.published
                articles.append({'title': title, 'url': link, 'published_date': pub_date, 'alias': alias})
            return articles

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)

    print(f"Failed to fetch articles for alias '{alias}' after {retries} attempts.")
    return []

def fetch_news_for_tickers(ticker_aliases: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Fetch news articles for each ticker and its aliases, with progress tracking.
    """
    all_articles = []
    # Use tqdm to show progress for each ticker
    for ticker, aliases in tqdm(ticker_aliases.items(), desc="Tickers", unit="ticker"):
        for alias in tqdm(aliases, desc=f"Processing aliases for {ticker}", leave=False, unit="alias"):
            articles = fetch_rss_articles(alias)
            for article in articles:
                article['stock_symbol'] = ticker
                all_articles.append(article)
    news_df = pd.DataFrame(all_articles)
    news_df = news_df.dropna()
    return news_df
