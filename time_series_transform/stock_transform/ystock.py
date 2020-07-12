from datetime import datetime
import yfinance as yf

class ystock(object):
    # todo:
    # single symbol query
    # data format changes
    # local data download
    # multiple symbol query
    # async call

    """
    Fetching stock data from yahoo finance
    
    API Document: 
    - https://github.com/ranaroussi/yfinance
    - https://pypi.org/project/fix-yahoo-finance/0.1.0/
    ---
    Require:
    - yfinance

    """
    def __init__(self,symbol):
        self._symbol = symbol
        self._ticker = self._getStock(symbol)
    

    # getter and setter
    @property
    def ticker(self):
        return self._ticker

    @property
    def symbol(self):
        return self._symbol
    @symbol.setter
    def symbol(self, symbol):
        self._symbol = symbol
        self._ticker = yf.Ticker(symbol)
    
    def _getStock(self, symbol):
        ticker = yf.Ticker(symbol)
        return ticker


    def getCompanyInfo(self):
        return self.ticker.info

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

    def getHistoricalByPeriod(self, period = '1mo'):
        return self.ticker.history(period)

    def getHistoricalByRange(self, start_date, end_date):
        return self.ticker.history(start = start_date, end = end_date)

    def getActions(self):
        return self.ticker.actions

    def getDividends(self):
        return self.ticker.dividends

    def getSplits(self):
        return self.ticker.splits

    def getSustainability(self):
        return self.ticker.sustainability

    def getRecommendations(self):
        return self.ticker.recommendations

    def getNextEvent(self):
        return self.ticker.calendar

