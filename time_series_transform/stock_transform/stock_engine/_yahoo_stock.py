import yfinance as yf
from time_series_transform.stock_transform.stock_engine.engine_interface import *
from datetime import datetime, timedelta

class yahoo_stock(engine_interface):

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

    def sample(self):
        print("omg")
        return 0
        
    def getCompanyInfo(self):
        try:
            return self.ticker.info
        except:
            return None

    def getHistoricalByPeriod(self, period = '1mo'):
        return self.ticker.history(period)

    def getHistoricalByRange(self, start_date, end_date):
        if valid_period_format(start_date) and valid_period_format(end_date):
            end_date = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            return self.ticker.history(start = start_date, end = end_date)
        else:
            raise ValueError('date format must be YYYY-mm-dd')

    def getIntraDayData(self,start_date,end_date, interval_range):
        valid_range = ["1m", "2m", "5m", "15m", "30m", '60m', '90m', '1h']
        if interval_range not in valid_range:
            raise ValueError("Invalid interval range, valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h]")

        if valid_period_format(start_date) and valid_period_format(end_date):
            end_date = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            return self.ticker.history(start = start_date, end = end_date, interval = interval_range)
        else:
            raise ValueError('date format must be YYYY-mm-dd')

    def getActions(self):
        try:
            return self.ticker.actions
        except:
            return None

    def getDividends(self):
        try:
            return self.ticker.dividends
        except:
            return None

    def getSplits(self):
        try:
            return self.ticker.splits
        except:
            return None

    def getSustainability(self):
        try:
            return self.ticker.sustainability
        except:
            return None

    def getRecommendations(self):
        try:
            return self.ticker.recommendations
        except:
            return None

    def getNextEvent(self):
        try:
            return self.ticker.calendar
        except:
            return None

    def getAdditionalInfo(self):
        info_dict = {
            'company_info':self.getCompanyInfo(),
            'sustainability': self.getSustainability()
        }

        schedule_dict = {
            'actions': self.getActions(),
            'recommendations': self.getRecommendations(),
            'next_event': self.getNextEvent()
        }
       
        data = {
            'info': info_dict,
            'schedule': schedule_dict
            
        }
        return data