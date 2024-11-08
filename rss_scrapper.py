import pandas as pd
import time
from typing import List, Dict
import feedparser
from urllib.parse import quote
from tqdm import tqdm  # For progress tracking

# Define NIFTY-50 stock tickers with aliases
ticker_aliases = {
    'ADANIPORTS.NS': ['Adani Ports', 'Adani Ports and Special Economic Zone'],
    'ASIANPAINT.NS': ['Asian Paints'],
    'AXISBANK.NS': ['Axis Bank'],
    'BAJFINANCE.NS': ['Bajaj Finance'],
    'BAJFINSERV.NS': ['Bajaj Finserv'],
    'BHARTIARTL.NS': ['Bharti Airtel', 'Airtel'],
    'BPCL.NS': ['Bharat Petroleum', 'BPCL'],
    'BRITANNIA.NS': ['Britannia', 'Britannia Industries'],
    'CIPLA.NS': ['Cipla'],
    'COALINDIA.NS': ['Coal India'],
    'DIVISLAB.NS': ["Divi's Laboratories"],
    'DRREDDY.NS': ["Dr. Reddy's Laboratories"],
    'EICHERMOT.NS': ['Eicher Motors'],
    'GAIL.NS': ['GAIL', 'GAIL (India)'],
    'GRASIM.NS': ['Grasim Industries'],
    'HCLTECH.NS': ['HCL Technologies'],
    'HDFCBANK.NS': ['HDFC Bank', 'HDFC'],
    'HDFCLIFE.NS': ['HDFC Life Insurance'],
    'HEROMOTOCO.NS': ['Hero MotoCorp'],
    'HINDALCO.NS': ['Hindalco Industries'],
    'HINDUNILVR.NS': ['Hindustan Unilever', 'HUL'],
    'ICICIBANK.NS': ['ICICI Bank', 'ICICI'],
    'ITC.NS': ['ITC Limited'],
    'INFY.NS': ['Infosys', 'Infosys Limited'],
    'KOTAKBANK.NS': ['Kotak Mahindra Bank', 'Kotak Bank'],
    'LARSEN.NS': ['Larsen & Toubro'],
    'M&M.NS': ['Mahindra & Mahindra', 'M&M'],
    'MARUTI.NS': ['Maruti Suzuki'],
    'NESTLEIND.NS': ['Nestlé India'],
    'NTPC.NS': ['NTPC'],
    'ONGC.NS': ['ONGC', 'Oil and Natural Gas Corporation'],
    'POWERGRID.NS': ['Power Grid Corporation'],
    'RELIANCE.NS': ['Reliance', 'Reliance Industries'],
    'SBIN.NS': ['SBI', 'State Bank of India'],
    'SHREECEM.NS': ['Shree Cement'],
    'SBILIFE.NS': ['SBI Life Insurance'],
    'SUNPHARMA.NS': ['Sun Pharmaceutical Industries'],
    'TATACONSUM.NS': ['Tata Consumer Products'],
    'TATAMOTORS.NS': ['Tata Motors'],
    'TATASTEEL.NS': ['Tata Steel'],
    'TECHM.NS': ['Tech Mahindra'],
    'ULTRACEMCO.NS': ['UltraTech Cement'],
    'WIPRO.NS': ['Wipro'],
    'ZEEL.NS': ['Zee Entertainment Enterprises']
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
                articles.append({'title': title, 'link': link, 'pub_date': pub_date, 'alias': alias})
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
                article['ticker'] = ticker
                all_articles.append(article)
    news_df = pd.DataFrame(all_articles)
    news_df = news_df.dropna(subset=['title']).drop_duplicates(subset='title')
    return news_df
