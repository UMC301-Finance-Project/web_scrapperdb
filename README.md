# Stock News Sentiment Analysis Pipeline

A Python-based solution for collecting, analyzing, and storing stock news headlines with sentiment scoring for NIFTY-50 companies. This project fetches news data from RSS feeds, processes it for multiple aspects, and stores it in a SQLite database for further analysis.

## Features

- **RSS News Collection**: Fetches stock-related news for NIFTY-50 companies from Google News RSS feeds, processing multiple aliases for each ticker.
- **Aspect-Based Sentiment Analysis**: Uses the `deberta-v3-base-absa-v1.1` model to score headlines based on key financial aspects, including Earnings, Revenue, Margins, and Sentiment.
- **Data Storage**: Stores results in an SQLite database, `stock_news.db`, with a schema accommodating ticker symbols, article titles, publication dates, URLs, and aspect-based sentiment scores.
- **Embedding Generation**: Generates sentence embeddings for headlines using the `all-MiniLM-L6-v2` model, allowing for future search and similarity queries.

## Project Structure

- **`StockNewsExtractor`**: Manages news data collection using a custom RSS scraper for stock aliases.
- **`NewsDatabase`**: Handles SQLite database creation and data insertion for news headlines, storing sentiment scores for multiple aspects.
- **`SentimentAnalyser`**: Leverages transformer models for both aspect-based and general sentiment analysis.

## Requirements

- Python 3.8+
- Transformers
- sentence-transformers
- pandas
- feedparser
- sqlite3
- torch

Install dependencies:

```bash
pip install -r requirements.txt

