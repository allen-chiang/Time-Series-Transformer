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

    def save(self, path, format = "csv"):
        data = self.df
        download_path = path + "/" + self.symbol + "_stock_extract." + format
        if format == 'csv':
            data.to_csv(download_path)
        elif format == 'parquet':
            data.to_parquet(download_path)
        else:
            raise ValueError("invalid format value")

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

    def plot(self,stockIndicators, keyCol = 'Default' ,samePlot=False,*args,**kwargs):
        df = None
        keyArr = None
        for ix,i in enumerate(stockIndicators):
            if samePlot:
                if keyCol == 'Default':
                    keyArr = [i for i in range(self.stockDict[i].df.shape[0])]
                else:
                    keyArr = self.stockDict[i].df[keyCol]

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
                self.stockDict[i].plot(stockIndicators[i],*args,**kwargs)
        
        if samePlot:
            df = df.set_index(keyCol)
            df.plot(*args, **kwargs)
            
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

def macd(arr):
    df = pd.DataFrame(data = arr, columns = ["Origin"])
    df['EMA_12'] = df['Origin'].ewm(span=12).mean()
    df['EMA_26'] = df['Origin'].ewm(span=26).mean()

    df['DIF'] = df['EMA_12'] - df['EMA_26']
    df['DEM'] = df['DIF'].ewm(span=9).mean()
    df['OSC'] = df['DIF'] - df['DEM']
    return df