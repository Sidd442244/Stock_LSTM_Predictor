import yfinance as yf
from datetime import datetime, timedelta

def get_stock_data(ticker, years):

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365*years)

    df = yf.download(ticker, start=start_date, end=end_date)

    return df