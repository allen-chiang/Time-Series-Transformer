import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import ChainMap
from joblib import Parallel, delayed


class Stock (object):
    def __init__(self,symbol,data,additionalInfo=None):
        self.df = data
        self.symbol = symbol
        self.additionalInfo = additionalInfo


    @property
    def dataFrame(self):
        self.df['symbol'] = self.symbol
        return self.df

    def _get_transformation_list(self):
        return {
            'moving_average':moving_average,
            'fast_fourier':fft_transform,
            'real_fast_fourier':rfft
        }

    def plot(self,colName,*args,**kwargs):
        self.df[colName].plot(*args,**kwargs)

    def make_technical_indicator(self,colName,labelName,indicator,*args,**kwargs):
        techList = self._get_transformation_list()
        arr = self.df[colName].values
        indicator = techList[indicator](arr,*args,**kwargs)
        self.df[f'{labelName}_{colName}'] = indicator
        return self


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

    def plot(self,stockIndicators,samePlot=False,*args,**kwargs):
        for ix,i in enumerate(stockIndicators):
            if samePlot:
                if ix == 0:
                    ax = self.stockDict[i].plot(stockIndicators[i],*args,**kwargs)
                else:
                    self.stockDict[i].plot(stockIndicators[i],ax = ax,*args,**kwargs)
            else:
                self.stockDict[i].plot(stockIndicators[i],*args,**kwargs)
        plt.show()


def moving_average(arr, windowSize=3) :
    ret = np.cumsum(arr, dtype=float)
    ret[windowSize:] = ret[windowSize:] - ret[:-windowSize]
    ret = ret[windowSize - 1:] / windowSize
    res = np.empty((int(len(arr)-len(ret))))
    res[:] = np.nan
    return np.append(res,ret) 

def fft_transform(arr):
    return scipy.fft.fft(arr)

def rfft(arr):
    return scipy.fft.rfft(arr)
