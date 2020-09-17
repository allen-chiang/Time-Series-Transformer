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

class Stock(Time_Series_Data):
    def __init__(self, symbol, data,additionalInfo = None, timeSeriesCol='Date'):
        super().__init__()
        self.symbol = symbol
        self.additionalInfo = additionalInfo
        self.timeSeriesCol = timeSeriesCol
        
    def get_dataFrame(self):
        self.make_dataframe()

    def plot(self, colName = 'Close', *args, **kwargs):
        pass

    def make_technical_indicator(self, colName, labelName, indicatorFunctions, *args, **kwargs):
        pass

class Portfolio(Time_Series_Data_Colleciton):
    def __init__(self, stockList, timeSeriesCol = 'Date', categoryCol = 'symbol'):
        
        super().__init__(self,time_series_data, timeSeriesCol, categoryCol)

    def make_technical_indicator(self,colName, LabelName, indicator, n_job =1, verbose = 0, *args, **kwargs):
        pass

    def get_portfolio_dataFrame(self):
        pass

    def weight_calculate(self,weights={}, colName = 'Close'):
        pass

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
        pass


    