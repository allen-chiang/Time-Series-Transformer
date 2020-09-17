import scipy
import numpy as np
import pandas as pd
from numpy.fft import *
import matplotlib.pyplot as plt
from collections import ChainMap
from joblib import Parallel, delayed
import plotly.graph_objects as go
from time_series_transform.transform_core_api.util import *
from time_series_transform.transform_core_api.base import *
from time_series_transform.io import *

class Stock (object):
    def __init__(self,symbol,data,additionalInfo=None,timeSeriesCol = 'Date'):
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
        self.df = data
        self.data = from_pandas(data, timeSeriesCol)
        self.symbol = symbol
        self.additionalInfo = additionalInfo
        self.timeSeriesCol = timeSeriesCol
        self.df = self.df.sort_values(timeSeriesCol,ascending = True)
        self.dateRange = self.df.sort_values(timeSeriesCol,ascending=True)[timeSeriesCol].unique().tolist()

    @property
    def dataFrame(self):
        self.df['symbol'] = self.symbol
        return self.df

    def plot(self,colName = "Close",*args,**kwargs):
        """
        plot the stock data of the given column using matplot
        
        Parameters
        ----------
        colName : str, optional
            column of the data used for plotting
        """
        # self.df[colName].plot(*args,**kwargs)
        data = self.data[:,[colName]]
        fig, ax = plt.subplots()
        ax.plot(data[self.timeSeriesCol], data[colName])
        plt.xticks(rotation=90)
        ax.set(**kwargs)
        plt.show()

    def save(self, path, format = "csv",compression = None):
        """
        save the main data locally using pandas
        
        Parameters
        ----------
        path : str 
            path of local directory
        format : str, optional
            format of the file
            valid values: csv, parquet
        compression: str, optional
            compression method, by default None
        """
        data = self.df
        download_path = path + "/" + self.symbol + "_stock_extract." + format
        if format == 'csv':
            data.to_csv(download_path,compression = compression)
        elif format == 'parquet':
            data.to_parquet(download_path,compression = compression)
        else:
            raise ValueError("invalid format value")

    def make_technical_indicator(self,colName,labelName,indicatorFunction,*args,**kwargs):
        """
        make_technical_indicator applies the indicatorFunctions to the given column
        
        Parameters
        ----------
        colNames : str 
            column of the data used for the indicator functions
        labelName : str
            label name to show on the dataframe
        indicatorFunction: dict
            dict of the indicator functions
        """
        if isinstance(colName,list):
            arr = self.df[colName]
        else:
            arr = self.df[colName].values
        indicator = indicatorFunction(arr,*args,**kwargs)
        if isinstance(indicator,dict) or isinstance(indicator,pd.DataFrame):
            for k in indicator:
                self.df[f'{labelName}_{k}'] = indicator[k]

            for func in indicatorFunction:
                self.data.transform(colName, labelName, func,*args, **kwargs)
        else:
            self.df[f'{labelName}'] = indicator

            self.data.transform(colName, labelName, indicatorFunction,*args, **kwargs)
        return self

    def get_dataFrame(self):
        return self.data.make_dataframe()

    

class Portfolio(object):
    def __init__(self,stockList):
        """
        The class initialize data as a Portfolio object, which stores multiple Stock data
        
        Parameters
        ----------
        stockList : list[Stock]
            list of Stock data
        """
        self.stockDict = self._get_stock_dict(stockList)

    def _get_stock_dict(self,stockList):
        """
        _get_stock_dict transform the stock list into dictionary
        
        Parameters
        ----------
        stockList : list[Stock] 
            list of Stock data
        """
        stockDict = {}
        for i in stockList:
            key = i.symbol
            stockDict[key] = i
        return stockDict

    def make_technical_indicator(self,colName,labelName,indicator,n_jobs =1,verbose = 0,*args,**kwargs):
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
        dctList =  Parallel(n_jobs,verbose = verbose)(delayed(self._stock_technical_indicator)(
            self.stockDict[i],
            i,colName,
            labelName,
            indicator,
            *args,
            **kwargs) for i in self.stockDict)
        self.stockDict = dict(ChainMap(*dctList))


    def _stock_technical_indicator(self,stock,symbol,colName,labelName,indicator,*args,**kwargs):
        return {symbol:stock.make_technical_indicator(colName,labelName,indicator,*args,**kwargs)}


    def get_portfolio_dataFrame(self):
        """
        get_portfolio_dataFrame return the portfolio overview
        
        Returns
        -------
        portfolio
            portfolio dataset including all of the applied indicators
        """
        portfolio = None
        for ix,v in enumerate(self.stockDict):
            if ix == 0:
                portfolio = self.stockDict[v].dataFrame
            else:
                portfolio = portfolio.append(self.stockDict[v].dataFrame)
        return portfolio


    def remove_different_date(self):
        """
        remove_different_date remove non-common date of the dataframe
        """
        timeCol = {}
        for i in self.stockDict:
            for v in self.stockDict[i].dateRange:
                if v not in timeCol:
                    timeCol[v] = 1
                else:
                    timeCol[v]+=1
        timeCol = [k for k,v in timeCol.items() if v == len(self.stockDict)]
        for i in self.stockDict:
            self.stockDict[i].dateRange = timeCol
            timeSeriesCol = self.stockDict[i].timeSeriesCol
            self.stockDict[i].df = self.stockDict[i].df[self.stockDict[i].df[timeSeriesCol].isin(timeCol)]

    def weight_calculate(self,weights = {}, colName = 'Close'):
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
        self.remove_different_date()
        if len(weights) == 0 or sum(weights.values()) == 0:
            for st in self.stockDict:
                stock = self.stockDict[st]
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
        for stock in self.stockDict:
            df = self.stockDict[stock].df
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
                keyArr = [i for i in range(self.stockDict[i].df.shape[0])]
            else:
                keyArr = self.stockDict[i].df[keyCol].tolist()
            
            if samePlot:
                tmp = self.stockDict[i].df[stockIndicators[i]]
                tmp.insert(0,keyCol,keyArr)
                colName = [keyCol]
                colName.extend([f'{i}_{d}' for d in stockIndicators[i]])
                tmp.columns = colName

                if ix == 0:
                    df = tmp
                else:
                    df = pd.merge(df,tmp, on = [keyCol], how = 'outer')

            else:
                self.stockDict[i].plot(stockIndicators[i], title = i+ " plot", *args,**kwargs)
        
        if samePlot:
            df = df.set_index(keyCol)
            df.plot(*args, **kwargs)
            
        plt.show()


