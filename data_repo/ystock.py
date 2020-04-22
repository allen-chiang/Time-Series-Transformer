from datetime import datetime
import yfinance as yf

class ystock(object):
    """
    Fetching stock data from IEXFinance
    
    API Document: 
    - https://github.com/ranaroussi/yfinance
    - https://pypi.org/project/fix-yahoo-finance/0.1.0/
    ---
    Require:
    - yfinance

    """
    def __init__(self):
        super().__init__()
        
    def getStock(self, symbol):
        ticker = yf.Ticker(symbol)
        return ticker

    def getCompanyInfo(self, symbol):
        company_info = yf.Ticker(symbol).info
        return company_info

    """
    Historical Data
    ---
    Input:
    symbol: string
    period: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    (default is '1mo')
    start_date, end_date: String, "%Y-%m-%d", eg "2020-02-20"
    ---
    Return:
    date, open, high, low, close, volume, dividends, stock splits
    """

    def getHistorical(self, symbol, period = '1mo'):
        chart = yf.Ticker(symbol).history(period)
        return chart

    def getHistoricalByRange(self, symbol, start_date, end_date):
        chart = yf.Ticker(symbol).history(start_date = start_date, end_date = end_date)
        return chart

    def getActions(self, symbol):
        df = yf.Ticker(symbol).actions
        return df

    def getDividends(self, symbol):
        df = yf.Ticker(symbol).dividends
        return df

    def getSplits(self, symbol):
        df = yf.Ticker(symbol).splits
        return df

    def getSustainability(self, symbol):
        df = yf.Ticker(symbol).sustainability
        return df

    def getRecommendations(self, symbol):
        df = yf.Ticker(symbol).recommendations
        return df

    def getNextEvent(self, symbol):
        df = yf.Ticker(symbol).calendar
        return df

