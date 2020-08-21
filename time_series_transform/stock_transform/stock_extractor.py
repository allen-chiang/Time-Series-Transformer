import pandas as pd
from time_series_transform.stock_transform.base import *
from time_series_transform.stock_transform.stock_engine._yahoo_stock import yahoo_stock
from time_series_transform.stock_transform.stock_engine._investing import investing

class Stock_Extractor(object):
    def __init__(self,symbol,engine, *args, **kwargs):
        """
        Stock_Extractor extracts data of the given symbol 
        using the selected engine   

        For investing engine: country is required.
        for example, Stock_Extractor('aapl','investing', country = 'united states')

        Parameters
        ----------
        symbol : str
            symbol of the stock
        engine : str
            engine used for data extraction
        """
        self.client = self._get_extractor(engine)(symbol, **kwargs)
        self.symbol = symbol
        self.stock = None

    def _get_extractor(self,engine):
        engineDict = {
            'yahoo': yahoo_stock,
            'investing': investing
        }
        return engineDict[engine]

    def get_stock_period(self,period):
        """
        get_stock_period extracts the stock data of the selected
        period

        Parameters
        ----------
        period : str
            period of the data
            for example, 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max 

        Returns
        -------
        stock data
            The stock data of selected period
        """
        data = self.client.getHistoricalByPeriod(period)
        data = pd.DataFrame(data.to_records())
        data['Date'] = data.Date.astype(str)
        additionalInfo = self.client.getAdditionalInfo()
        self.stock = Stock(self.symbol,data,additionalInfo,'Date')
        return self.stock

    def get_stock_date(self,start_date,end_date):
        """
        get_stock_period extracts the stock data of the selected
        period

        Parameters
        ----------
        start_date : str
            start of the data
            format: "%Y-%m-%d", eg "2020-02-20"

        end_date : str
            end of the data
         
        Returns
        -------
        stock data
            The stock data of selected period
        """
        data = self.client.getHistoricalByRange(start_date,end_date)
        data = pd.DataFrame(data.to_records())
        data['Date'] = data.Date.astype(str)
        additionalInfo = self.client.getAdditionalInfo()
        self.stock = Stock(self.symbol,data,additionalInfo,'Date')
        return self.stock

    # I/O
    @classmethod
    def get_stock_from_csv(cls, symbol, path, *args, **kwargs):
        """
        get_stock_from_csv extracts data from a local csv file

        Parameters
        ----------
        symbol : str
            symbol of the given stock data
        path : str
            path of the csv file

        Returns
        -------
        Stock
            The stock data extracted from the csv file
        """
        data = pd.read_csv(path)
        stock_data = Stock(symbol, data, *args, **kwargs)
        return stock_data


    @classmethod
    def get_stock_from_parquet(cls, symbol, path, *args, **kwargs):
        """
        get_stock_from_parquet extracts data from a local parquet file

        Parameters
        ----------
        symbol : str
            symbol of the given stock data
        path : str
            path of the parquet file

        Returns
        -------
        Stock
            The stock data extracted from the parquet file
        """
        data = pd.read_parquet(path, engin = 'pyarrow')
        stock_data = Stock(symbol, data, *args, **kwargs)
        return stock_data

class Portfolio_Extractor(object):
    def __init__(self,symbolList,engine):
        """
        Portfolio_Extractor extracts data of the given symbolList
        using the selected engine   

        Parameters
        ----------
        symbolList : list
            list of symbol 
        engine : str
            engine used for data extraction
        """
        self.engine = engine
        self.symbolList = symbolList
        self.portfolio = None

    def get_portfolio_period(self,period):
        """
        get_portfolio_period extracts the list of stock
        by the given period

        Parameters
        ----------
        period : str
            period of the data
            for example, 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max 

        Returns
        -------
        portfolio
            portfolio data of the given stock list 
        """
        stockList = []
        for symbol in self.symbolList:
            stock_data = Stock_Extractor(symbol, self.engine).get_stock_period(period)
            stockList.append(stock_data)

        self.portfolio = Portfolio(stockList)
        return self.portfolio

    def get_portfolio_date(self,start_date, end_date):
        """
        get_portfolio_date extracts the list of stock
        by the date period

        Parameters
        ----------
        start_date : str
            start of the data
            format: "%Y-%m-%d", eg "2020-02-20"

        end_date : str
            end of the data

        Returns
        -------
        portfolio
            portfolio data of the given stock list 
        """
        stockList = []
        for symbol in self.symbolList:
            stock_data = Stock_Extractor(symbol, self.engine)
            stock_data = stock_data.get_stock_date(start_date, end_date)
            stockList.append(stock_data)

        self.portfolio = Portfolio(stockList)
        return self.portfolio




