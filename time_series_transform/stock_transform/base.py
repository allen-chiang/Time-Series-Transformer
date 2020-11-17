import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import ChainMap
from joblib import Parallel, delayed
import plotly.graph_objects as go
from time_series_transform.transform_core_api.util import *
from time_series_transform.transform_core_api.base import *
from time_series_transform.io import *

class Stock(Time_Series_Data):
    def __init__(self, symbol, data, additionalInfo = None, timeSeriesCol='Date'):
        """
        The class initialize data as a Stock object 
        
        Parameters
        ----------
        symbol : str 
            symbol of the stock
        data : pandas dataframe
            main data of the stock, for example, Date, High, Low, Open, Close, Volume
        additionalInfo: dict, optional
            supplemental information beside the data, by default None
        timeSeriesCol: str, optional
            time series column name for sorting data
        """
        if isinstance(data, pd.DataFrame):
            data = data.to_dict('list')
        super().__init__(data, {timeSeriesCol:data[timeSeriesCol]})
        self.symbol = symbol
        self.additionalInfo = additionalInfo
        self.timeSeriesCol = timeSeriesCol
        
    def get_data_columns(self):

        return list(self.data[0].keys())

    def get_dataFrame(self):
        """
        get_dataFrame returns the pandas dataframe of the stock data

        Returns
        -------
        pandas df
            stock data
        """
        return to_pandas(self,None,False,None)

    def plot(self, colName = 'Close', *args, **kwargs):
        """
        plot the stock data of the given column using matplot
        
        Parameters
        ----------
        colName : str, optional
            column of the data used for plotting
        """
        data = self.data[:,[colName]]
        fig, ax = plt.subplots()
        ax.plot(data[self.timeSeriesCol], data[colName])
        plt.xticks(rotation=90)
        ax.set(**kwargs)
        plt.show()

    def make_technical_indicator(self, colNames, labelNames, func, *args, **kwargs):
        """
        make_technical_indicator applies the indicatorFunction to the given columns
        
        Parameters
        ----------
        colNames : list or str 
            columns of the data used for the indicator function
        labelNames : list or str
            label name to show on the dataframe 
            if str is given, label will be in the format of {col}_{label}
        func : function
            indicator functions
        """
        if isinstance(colNames, list):
            if isinstance(labelNames, list):
                if len(labelNames) != len(colNames):
                    raise ValueError("labelNames length is different from colNames")
                else:
                    for i in range(len(colNames)):
                        self.transform(colNames[i], labelNames[i], func, *args, **kwargs)
            else:
                for col in colNames:
                    label = f'{col}_{labelNames}'
                    self.transform(col, label, func,*args, **kwargs)
        else:
            self.transform(colNames, labelNames, func,*args, **kwargs)
        
        return self

class Portfolio(Time_Series_Data_Collection):
    def __init__(self, stockList, timeSeriesCol = 'Date', categoryCol = 'symbol'):
        """
        The class initialize data as a Portfolio object, which stores multiple Stock data
        
        Parameters
        ----------
        stockList : list[Stock]
            list of Stock data

        timeSeriesCol : str
            time series column

        categoryCol : str
            column of the category
        """
        time_series_data = self._get_time_data_from_stock(stockList)
        super().__init__(time_series_data, timeSeriesCol, categoryCol)

    def _get_time_data_from_stock(self,stockList):
        res = {}
        for stock in stockList:
            res[stock.symbol] = stock
        return res

    def make_technical_indicator(self,colName, LabelName, indicator, n_job =1, verbose = 0, *args, **kwargs):
        """
        make_technical_indicator applies the indicator to the given column
        
        Parameters
        ----------
        colName : str 
            column of the data used for the indicator functions
        labelName : str
            label name to show on the dataframe
        indicator : function
            function of the indicator
        n_jobs : int
            The maximum number of concurrently running jobs, by default 1
        verbose : int
            The verbosity level: if non zero, progress messages are printed, by default 0
        """
        self.transform(colName, LabelName, indicator,n_job, verbose, *args, **kwargs)
        return self

    def get_portfolio_dataFrame(self):
        # todo symbol missing
        """
        get_portfolio_dataFrame return the portfolio overview
        
        Returns
        -------
        pandas df
            portfolio dataset including all of the applied indicators
        """
        portfolio = None
        for ix,v in enumerate(self.time_series_data_collection):
            if ix == 0:
                portfolio = self.time_series_data_collection[v].get_dataFrame()
            else:
                portfolio = portfolio.append(self.time_series_data_collection[v].get_dataFrame())
        return portfolio
        

    def weight_calculate(self,weights={}, colName = 'Close'):
        # additional info missing
        """
        generate weight index with default to weight by market cap

        Parameters
        ----------
        weights : dict, optional
            dictionary of the weight, by default {}
            for example, {'aapl':1, 'msft':1} will return average 
        colName : str, optional
            column name to be calculated, by default 'Close'

        Returns
        -------
        pandas dataframe
            columns of the original data and the index
        """
        self.remove_different_time_index()
        if len(weights) == 0 or sum(weights.values()) == 0:
            for st in self.time_series_data_collection:
                stock = self.time_series_data_collection[st]
                info = stock.additionalInfo['info']['company_info']
                outstanding_col = [i for i in list(info.keys()) if 'Outstanding' in i][0]
                outstanding_shares = info[outstanding_col]
                data = outstanding_shares*stock.df[colName]
                data = data.reset_index()[colName]
                weights[stock.symbol] = data

        total_w = sum(weights.values())
        ret = {}
        indx = []
        pd_indx = 0
        for stock in self.time_series_data_collection:
            df = self.time_series_data_collection[stock].df
            df = df.set_index('Date')
            ret[stock + '_' + colName] = df[colName]
            
            pd_indx = df[colName].index
            weight_data = df[colName].reset_index()[colName] * weights[stock]/total_w

            if len(indx) == 0:
                indx.extend(weight_data)
            else:
                indx += weight_data

        pd_ret = pd.DataFrame(ret)
        indx = pd.DataFrame(indx).set_index(pd_indx)

        pd_ret['weighted_index'] = indx
        return pd_ret

    def plot(self,stockIndicators, keyCol = 'Default' ,samePlot=False,*args,**kwargs):
        """
        plot the given stock indicator in the portfolio
        
        Parameters
        ----------
        stockIndicators : dict 
            dictionary of the stock indicator used for plotting 
        keyCol : str
            xaxis of the plot, by default "Default"
            valid value: columns of the data
        samePlot : boolean
            whether to show the stock in the same plot
        """
        df = None
        keyArr = None
        for ix,i in enumerate(stockIndicators):
            if keyCol == 'Default':
                keyArr = [i for i in range(self.time_series_data_collection[i].df.shape[0])]
            else:
                keyArr = self.time_series_data_collection[i].df[keyCol].tolist()
            
            if samePlot:
                tmp = self.time_series_data_collection[i].df[stockIndicators[i]]
                tmp.insert(0,keyCol,keyArr)
                colName = [keyCol]
                colName.extend([f'{i}_{d}' for d in stockIndicators[i]])
                tmp.columns = colName

                if ix == 0:
                    df = tmp
                else:
                    df = pd.merge(df,tmp, on = [keyCol], how = 'outer')

            else:
                self.time_series_data_collection[i].plot(stockIndicators[i], title = i+ " plot", *args,**kwargs)
        
        if samePlot:
            df = df.set_index(keyCol)
            df.plot(*args, **kwargs)
            
        plt.show()



    