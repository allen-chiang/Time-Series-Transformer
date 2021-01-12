import threading
import numpy as np
import pandas as pd
from time_series_transform.stock_transform.base import *
from time_series_transform.stock_transform.stock_engine._investing import investing
from time_series_transform.stock_transform.stock_engine._yahoo_stock import yahoo_stock
from datetime import date, timedelta

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
        self.client = self._get_extractor(engine)(symbol, *args, **kwargs)
        self.symbol = symbol
        self.stock = None

    def _get_extractor(self,engine):
        engineDict = {
            'yahoo': yahoo_stock,
            'investing': investing
        }
        return engineDict[engine]

    def get_period(self,period):
        """
        get_period extracts the stock data of the selected
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
        self.stock = Stock(
            data,
            time_index='Date',
            symbol=self.symbol
            )
        return self.stock

    def get_date(self,start_date,end_date):
        """
        get_period extracts the stock data of the selected
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
        self.stock = Stock(
            data,
            time_index='Date',
            symbol = self.symbol
            )
        return self.stock

    def get_intra_day(self,start_date,end_date,interval = '1m'):
        """
        get_intra_day extracts the intraday stock data of the selected
        period

        Parameters
        ----------
        start_date : str
            start of the data
            format: "%Y-%m-%d", eg "2020-02-20"

        end_date : str
            end of the data
        
        interval : str
            interval of the data
            Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h]
         
        Returns
        -------
        stock data
            The stock data of selected period
        """
        data = self.client.getIntraDayData(start_date,end_date,interval)
        data = pd.DataFrame(data.to_records())
        data['Datetime'] = data.Datetime.astype(str)
        self.stock= Stock(
            data,
            time_index = 'Datetime',
            symbol = self.symbol
        )
        return self.stock

class Portfolio_Extractor(object):
    def __init__(self,symbolList,engine, *args, **kwargs):
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
        self.args = args
        self.kwargs = kwargs

    def get_period(self,period, n_threads= 8):
        """
        get_period extracts the list of stock
        by the given period

        Parameters
        ----------
        period : str
            period of the data
            for example, 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max 
        
        n_threads : int
            number of thread of multi-thread processing

        Returns
        -------
        portfolio
            portfolio data of the given stock list 
        """
        stockList = self._get_stock_list_multi(n_threads,'get_period', [period])
        self.portfolio = Portfolio(
            stockList,
            time_index='Date',
            symbolIx='symbol'
            )
        return self.portfolio

    def get_date(self,start_date, end_date, n_threads = 8):
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
        
        n_threads : int
            number of thread of multi-thread processing

        Returns
        -------
        portfolio
            portfolio data of the given stock list 
        """
        stockList = self._get_stock_list_multi(n_threads,'get_date', [start_date, end_date])
        self.portfolio = Portfolio(
            stockList,
            time_index='Date',
            symbolIx='symbol'
            )
        return self.portfolio

    def get_intra_day(self,start_date, end_date, interval = '1m', n_threads = 8):
        """
        get_intra_day extracts the intraday data of the list of stock data
        by the date period

        Parameters
        ----------
        start_date : str
            start of the data
            format: "%Y-%m-%d", eg "2020-02-20"

        end_date : str
            end of the data
        
        interval : str
            interval of the data
            Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h]
        
        n_threads : int
            number of thread of multi-thread processing

        Returns
        -------
        portfolio
            portfolio data of the given stock list 
        """
        stockList = self._get_stock_list_multi(n_threads,'get_intra_day', [start_date, end_date, interval])
        self.portfolio = Portfolio(
            stockList,
            time_index='Datetime',
            symbolIx='symbol'
            )
        return self.portfolio

    def _get_stock_list_multi(self, n_threads, func, time_val):
        stockList = []
        tasks = []
        if len(self.symbolList) < n_threads:
            n_threads = len(self.symbolList)

        bins = np.array_split(self.symbolList, n_threads)
        for bn in bins:
            thread = threading.Thread(target=self._get_stock_data, args= [stockList, bn, func, time_val])
            tasks.append(thread)
            thread.start()

        for task in tasks:
            task.join()
        
        stockDict = {}
        for i in stockList:
            stockDict.update(i)
        return stockDict

    def _get_stock_data(self, stockList, symbolList, func, time_val, *args, **kwargs):
        for i in range(len(symbolList)):
            symbol = symbolList[i]
            if self.engine == "investing":
                if 'country' not in self.kwargs:
                    raise ValueError("Country must be included while using the investing engine")
                country = self.kwargs['country'][i]
                stock_data = Stock_Extractor(symbol, self.engine, *self.args, country = country)
            else:
                stock_data = Stock_Extractor(symbol, self.engine, *self.args, **self.kwargs)
            extract_func = getattr(stock_data,func)
            stock_data = extract_func(*time_val)
            stockList.append({symbol:stock_data})
        return stockList


