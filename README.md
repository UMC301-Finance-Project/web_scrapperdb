# Stock News Sentiment Analysis System

A comprehensive system for collecting, analyzing, and storing news sentiment data for NIFTY-50 stocks using aspect-based sentiment analysis.

## Overview

This project fetches real-time news data for NIFTY-50 stocks from Google News RSS feeds and performs aspect-based sentiment analysis on various financial metrics. The system analyzes sentiment across multiple aspects like Earnings, Revenue, Margins, and more, storing the results in a SQLite database for further analysis.

## Features

- Real-time news collection from Google News RSS feeds
- Aspect-based sentiment analysis for financial news
- Support for all NIFTY-50 stocks
- Sentiment analysis across multiple financial aspects:
  - Earnings
  - Revenue
  - Margins
  - Dividend
  - EBITDA
  - Debt
  - Overall Sentiment
- SQLite database storage with embeddings for efficient retrieval
- Retry mechanism for robust news fetching

## Dependencies

```plaintext
pandas
feedparser
tqdm
sqlite3
torch
transformers
sentence-transformers
```

## Project Structure

### Core Components

1. `rss_scrapper.py`: News collection module
2. `news_collectordb.py`: Database management and coordination
3. `absa.py`: Aspect-based sentiment analysis implementation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-news-sentiment.git
cd stock-news-sentiment
```

2. Install required packages:
```bash
pip install pandas feedparser tqdm torch transformers sentence-transformers
```

3. Download required models:
- DeBERTa v3 for aspect-based sentiment analysis
- FinBERT for financial sentiment analysis

## Usage

Run the main script to start collecting and analyzing news:

```bash
python news_collectordb.py
```

## Technical Details

### Database Schema

The SQLite database stores the following information for each news article:
- Stock symbol
- Headline
- Published date
- URL
- Text embedding
- Sentiment scores for each financial aspect

### Sentiment Analysis

The system uses two types of sentiment analysis:
1. Aspect-based sentiment analysis using DeBERTa v3
2. General financial sentiment analysis using FinBERT

Sentiment scores range from -1 (negative) to 1 (positive), with 0 indicating neutral sentiment.

## Contributors

Pratyush Kant
Sirjan Hansda
