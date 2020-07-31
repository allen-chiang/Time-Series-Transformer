import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import ChainMap
from joblib import Parallel, delayed
from numpy.fft import *
from time_series_transform.util import *
import plotly.graph_objects as go

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
        self.symbol = symbol
        self.additionalInfo = additionalInfo
        self.timeSeriesCol = timeSeriesCol
        self.df = self.df.sort_values(timeSeriesCol,ascending = True)
        self.dateRange = self.df.sort_values(timeSeriesCol,ascending=True)[timeSeriesCol].unique().tolist()

    @property
    def dataFrame(self):
        self.df['symbol'] = self.symbol
        return self.df

    def plot(self,colName,*args,**kwargs):
        """
        plot the stock data of the given column using matplot
        
        Parameters
        ----------
        colName : str 
            column of the data used for plotting
        """
        self.df[colName].plot(*args,**kwargs)

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

    def make_technical_indicator(self,colNames,labelName,indicatorFunction,*args,**kwargs):
        """
        make_technical_indicator applies the indicatorFunctions to the given column
        
        Parameters
        ----------
        colName : str 
            column of the data used for the indicator functions
        labelName : str
            label name to show on the dataframe
        indicatorFunction: dict
            dict of the indicator functions
        """
        if isinstance(colNames,list):
            arr = self.df[colNames]
        else:
            arr = self.df[colNames].values
        indicator = indicatorFunction(arr,*args,**kwargs)
        if isinstance(indicator,dict) or isinstance(indicator,pd.DataFrame):
            for k in indicator:
                self.df[f'{labelName}_{k}'] = indicator[k]
        else:
            self.df[f'{labelName}'] = indicator
        return self

    

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
                self.stockDict[i].plot(stockIndicators[i], *args,**kwargs)
        
        if samePlot:
            df = df.set_index(keyCol)
            df.plot(*args, **kwargs)
            
        plt.show()


def macd(arr, return_diff = False):
    """
    Return the moving average convergence/divergence 
    
    Parameters
    ----------
    arr : array 
        data used to calculate the macd

    Returns
    -------
    df : dict
        macd of the given data, including EMA_12, EMA_26, DIF, DEM, OSC
    """
    df = {}
    df['EMA_12'] = ema(arr, span=12).flatten()
    df['EMA_26'] = ema(arr, span=26).flatten()

    df['DIF'] = df['EMA_12'] - df['EMA_26']
    df['DEM'] = ema(pd.DataFrame(df['DIF']), span=9).flatten()
    df['OSC'] = df['DIF'] - df['DEM']
    if return_diff:
        return df['OSC']
    else:
        return df

def stochastic_oscillator(arr):
    """
    Return the stochastic oscillator 
    
    Parameters
    ----------
    arr : array 
        data used to calculate the stochastic oscillator

    Returns
    -------
    ret : dict
        stochastic oscillator of the given data, including k and d values
    """
    rsv_day = 9
    ret = {}
    df = pd.DataFrame(arr)

    rsv_rolling = df.rolling(rsv_day)
    rsv_val = 100*(df - rsv_rolling.min())/(rsv_rolling.max() - rsv_rolling.min())
    ret['k_val'] = rsv_val
    ret['d_val'] = np.array(ret['k_val'].rolling(3).mean())
    ret['k_val'] = np.array(rsv_val)
    
    return ret

def rsi(arr, n_day = 14):
    """
    Return the Relative Strength Index 
    
    Parameters
    ----------
    arr : array 
        data used to calculate the Relative Strength Index

    Returns
    -------
    rsi : array
        Relative Strength Index of the given array
    """
    orgLen = len(arr)
    arr = arr[~np.isnan(arr)]

    dif = [arr[i+1]-arr[i] for i in range(len(arr)-1)]
    u_val = np.array([val if val>0 else 0 for val in dif])
    d_val = np.array([-1*val if val<0 else 0 for val in dif])

    u_ema = ema(u_val, span = n_day)
    d_ema = ema(d_val,span = n_day)
    rs = u_ema/d_ema
    rsi = 100*(1-1/(1+rs))
    rsi = rsi.reshape(-1)
    res = np.empty((int(orgLen-len(rsi))))
    res[:] = np.nan
    return np.append(res,rsi) 

def williams_r(arr, n_day=14):
    """
    Return the Williams %R index
    
    Parameters
    ----------
    arr : array 
        data used to calculate the Williams %R
    n_day : int
        window of the indicator

    Returns
    -------
    r_val : array
        Relative Strength Index of the given array
    """
    orgLen = len(arr)
    arr = arr[~np.isnan(arr)]
    df = pd.DataFrame(arr)
    r_rolling = df.rolling(n_day)
    r_val = 100*(r_rolling.max()-df)/(r_rolling.max() - r_rolling.min())
    res = np.empty((int(orgLen-len(r_val))))
    res[:] = np.nan
    return np.append(res,rsi) 