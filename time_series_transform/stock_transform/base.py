import scipy
import numpy as np
import pandas as pd
from functools import wraps
import matplotlib.pyplot as plt


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
        plt.show()

    def make_technical_indicator(self,colName,labelName,indicator,return_indicator,*args,**kwargs):
        techList = self._get_transformation_list()
        arr = self.df[colName].values
        indicator = techList[indicator](arr,*args,**kwargs)
        self.df[f'{labelName}_{colName}'] = indicator
        if return_indicator:
            return indicator


class Portfolio(object):
    def __init__(self,stockList):
        pass



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
