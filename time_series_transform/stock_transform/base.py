import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import ChainMap
from joblib import Parallel, delayed
from numpy.fft import *
from time_series_transform.util import *

class Stock (object):
    def __init__(self,symbol,data,additionalInfo=None):
        self.df = data
        self.symbol = symbol
        self.additionalInfo = additionalInfo


    @property
    def dataFrame(self):
        self.df['symbol'] = self.symbol
        return self.df

    def plot(self,colName,*args,**kwargs):
        self.df[colName].plot(*args,**kwargs)

    def save(self, path, format = "csv",compression = None):
        data = self.df
        download_path = path + "/" + self.symbol + "_stock_extract." + format
        if format == 'csv':
            data.to_csv(download_path,compression = compression)
        elif format == 'parquet':
            data.to_parquet(download_path,compression = compression)
        else:
            raise ValueError("invalid format value")

    def make_technical_indicator(self,colName,labelName,indicatorFunction,*args,**kwargs):
        arr = self.df[colName].values
        indicator = indicatorFunction(arr,*args,**kwargs)
        self.df[f'{colName}_{labelName}'] = indicator
        return self

    def macd_plot(self, colName):
        df = macd(self.df[colName].values)
        colList = df.columns.tolist()
        colList[0] = colName
        df.columns =colList

        fig,ax = plt.subplots(2,1,figsize=(10,10))
        plt.subplots_adjust(hspace=0.8)

        df[colName].plot(ax = ax[0])
        df['EMA_12'].plot(ax=ax[0])
        df['EMA_26'].plot(ax=ax[0])

        ax[0].legend()
        df['DIF'].plot(ax=ax[1])
        df['DEM'].plot(ax=ax[1])
        ax[1].fill_between(df.index,0,df['OSC'])
        ax[1].legend()
        plt.show()


class Portfolio(object):
    def __init__(self,stockList):
        self.stockDict = self._get_stock_dict(stockList)

    def _get_stock_dict(self,stockList):
        stockDict = {}
        for i in stockList:
            key = i.symbol
            stockDict[key] = i
        return stockDict

    def make_technical_indicator(self,colName,labelName,indicator,n_jobs =1,verbose = 0,*args,**kwargs):
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
        portfolio = None
        for ix,v in enumerate(self.stockDict):
            if ix == 0:
                portfolio = self.stockDict[v].dataFrame
            else:
                portfolio = portfolio.append(self.stockDict[v].dataFrame)
        return portfolio

    def plot(self,stockIndicators, keyCol = 'Default' ,samePlot=False,*args,**kwargs):
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


def macd(arr, targetCol = None):
    colName = 'Base'
    df = pd.DataFrame(data = arr, columns = [colName])
    df['EMA_12'] = df[colName].ewm(span=12).mean()
    df['EMA_26'] = df[colName].ewm(span=26).mean()

    df['DIF'] = df['EMA_12'] - df['EMA_26']
    df['DEM'] = df['DIF'].ewm(span=9).mean()
    df['OSC'] = df['DIF'] - df['DEM']

    if targetCol is None:
        return df
    else:
        return df[targetCol].values

def stochastic_oscillator(df):
    rsv_day = 9
    alpha = 1/3

    rsv_rolling = df['Close'].rolling(rsv_day)
    df['rsv'] = 100*(df['Close'] - rsv_rolling.min())/(rsv_rolling.max() - rsv_rolling.min())
    df = df.dropna()
    
    k_list = [50]
    for i, rsv in enumerate(list(df['rsv'])):
        k_val_prev = k_list[i]
        k_val = (1-alpha) * k_val_prev + alpha * rsv
        k_list.append(k_val)
    
    df['k_val'] = k_list[1:]
    
    d_list = [50]
    for i, k in enumerate(list(df['k_val'])):
        d_val_prev = d_list[i]
        d_val = (1-alpha) * d_val_prev + alpha * k
        d_list.append(d_val)
    
    df['d_val'] = d_list[1:]
    return df


