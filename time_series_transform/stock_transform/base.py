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
        if isinstance(indicator,dict):
            for k in indicator:
                self.df[f'{colName}_{labelName}_{k}'] = indicator[k]
        else:
            self.df[f'{colName}_{labelName}'] = indicator
        return self

    def macd_plot(self, colName):
        df = macd(self.df[colName])
        df[colName] = self.df[colName].values
        df = pd.DataFrame(df)

        fig,ax = plt.subplots(2,1,figsize=(10,10))
        plt.subplots_adjust(hspace=0.8)

        self.df[colName].plot(ax = ax[0])
        df['EMA_12'].plot(ax=ax[0])
        df['EMA_26'].plot(ax=ax[0])

        ax[0].legend()
        df['DIF'].plot(ax=ax[1])
        df['DEM'].plot(ax=ax[1])
        ax[1].fill_between(df.index,0,df['OSC'])
        ax[1].legend()
        plt.show()

    def candle_plot(self, *args, **kwargs):
        df = self.df
        colors = []
        INCREASING_COLOR = '#008000'
        DECREASING_COLOR = '#FF0000'

        data=[dict(type='candlestick',
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                yaxis = 'y2',
                name = self.symbol)]
        layout = dict()
        fig = dict(data = data,layout=layout)

        fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
        fig['layout']['xaxis'] = dict( rangeselector = dict( visible = True ) )
        fig['layout']['yaxis'] = dict( domain = [0, 0.2], showticklabels = False )
        fig['layout']['yaxis2'] = dict( domain = [0.2, 0.8] )
        fig['layout']['legend'] = dict( orientation = 'h', y=0.9, x=0.3, yanchor='bottom' )
        fig['layout']['margin'] = dict( t=40, b=40, r=40, l=40 )

        
        for i in range(len(df['Close'])):
            if i != 0:
                if df['Close'][i] > df['Close'][i-1]:
                    colors.append(INCREASING_COLOR)
                else:
                    colors.append(DECREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)
                
        fig['data'].append( dict( x=df['Date'], y=df['Volume'],                         
                                marker=dict( color=colors ),
                                type='bar', yaxis='y', name='Volume' ) )

        ret = go.Figure(fig)
        ret.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]), 
                dict(values=["2015-12-25", "2016-01-01"])
            ]
        )

        ret.show()


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


def macd(arr):
    df = {}
    df['EMA_12'] = ema(arr, span=12).mean().to_numpy()
    df['EMA_26'] = ema(arr, span=26).mean().to_numpy()

    df['DIF'] = df['EMA_12'] - df['EMA_26']
    df['DEM'] = ema(pd.DataFrame(df['DIF']), span=9).mean().to_numpy().reshape(df['DIF'].shape[0])
    df['OSC'] = df['DIF'] - df['DEM']
    return df

def stochastic_oscillator(arr):
    rsv_day = 9
    ret = {}
    df = pd.DataFrame(arr)

    rsv_rolling = df.rolling(rsv_day)
    rsv_val = 100*(df - rsv_rolling.min())/(rsv_rolling.max() - rsv_rolling.min())
    ret['k_val'] = rsv_val
    ret['d_val'] = np.array(ret['k_val'].rolling(3).mean())
    ret['k_val'] = np.array(rsv_val)
    
    return ret

def rsi(arr):
    dif = [arr[i+1]-arr[i] for i in range(len(arr)-1)]
    u_val = pd.DataFrame([val if val>0 else 0 for val in dif])
    d_val = pd.DataFrame([-1*val if val<0 else 0 for val in dif])

    u_ema = ema(u_val, span = 14).mean()
    d_ema = ema(d_val,span=14).mean()
    rs = u_ema/d_ema
    rsi = 100*(1-1/(1+rs))

    return rsi.to_numpy()
