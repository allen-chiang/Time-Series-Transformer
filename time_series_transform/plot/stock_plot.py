import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from time_series_transform.stock_transform.base import (Stock,Portfolio)
from time_series_transform.stock_transform.util import *
from time_series_transform.transform_core_api.util import *
from time_series_transform.plot.base import plot_base
from copy import copy

class StockPlot(plot_base):
    def __init__(self,stock):
        """
        Plot uses the stock data to create various plots

        Parameters
        ----------
        stock : Stock
            stock data to create the plot
        """
        self._checkStock(stock)
        super().__init__(stock)
        self.ohlcva = self.time_series.ohlcva
        self._candleplot()
        self._plots = {
            'y' : ['candleplot'],
            'y2' : ['volume']
        }
        
        

    def _checkStock(self, object):
        if isinstance(object,(Stock,Portfolio)):
            return
        else:
            raise ValueError('object is not stock')

    def _create_candle_data(self, df, symbol):
        colors = []
        INCREASING_COLOR = '#008000'
        DECREASING_COLOR = '#FF0000'
        data=[dict(type='candlestick',
                    x=self.time_index_data,
                    open=df[self.ohlcva['Open']],
                    high=df[self.ohlcva['High']],
                    low=df[self.ohlcva['Low']],
                    close=df[self.ohlcva['Close']],
                    yaxis = 'y',
                    name = str(symbol))]

        close_data = df[self.ohlcva['Close']]
        colors = [DECREASING_COLOR if close_data[i] < close_data[i-1] else INCREASING_COLOR for i in range(1,len(close_data))]
        colors.insert(0,DECREASING_COLOR)
        
        volume_data = dict( x=self.time_index_data, y=df[self.ohlcva['Volume']],                         
                                    marker=dict( color=colors ),
                                    type='bar', yaxis='y2', name=None )
        if symbol is not None:
            volume_data['name'] = str(symbol)+'_Volume'
        data.append(volume_data)

        return data

    def _candleplot(self):
        if self.is_collection:
            data = list()
            buttonList = list()
            visible_array = np.zeros(len(self.category)*2)
            for indx in range(len(self.category)):
                cat = self.category[indx]
                stock_data = self.time_series[cat].data
                plot_data = self._create_candle_data(stock_data, cat)
                data.extend(plot_data)
                va = copy(visible_array)
                va[indx*2] = 1
                va[indx*2+1] = 1
                buttonList.append(dict(label = str(cat),
                                        method = 'update',
                                        args = [{'visible': va==1},
                                                {'title': str(cat),
                                                'showlegend':True}]))


        else:
            data = self._create_candle_data(self.time_series.data,self.time_series.symbol)
            
        layout = {
            'plot_bgcolor' : 'rgb(250, 250, 250)',
            'xaxis' : dict( anchor = 'y2', rangeselector = dict( visible = True ) ),
            'yaxis' : dict( domain = [0.2, 0.8], showticklabels = True),
            'yaxis2' : dict( domain = [0, 0.2], showticklabels = False ),
            'legend' : dict( orientation = 'h', y=0.9, x=0.3, yanchor='bottom' ),
            'margin' : dict( t=40, b=40, r=40, l=40 )
        }

        fig = dict(data = data,layout=layout)
        ret = go.Figure(fig)
        self.fig = ret
        if self.is_collection:
            self.update_layout(
                updatemenus=[go.layout.Updatemenu(
                            active=0,
                            buttons=buttonList
                            )
                        ]
            )

    def _find_next_layer(self):
        cur_max = 0
        for k in self._plots.keys():
            if len(k) > 1:
                cur_max = max(cur_max, int(k[1:]))
        
        return 'y' + str(cur_max + 1)

        
    def _add_multi_trace(self, data, colors, subplot):
        indx = 0
        for i in data:
            showLegend = True
            trace = data[i]
            if i.find('Base') >= 0 :
                showLegend = False
            self.add_line(col = None, lineType = 'scatter', color = colors[indx], legendName = i,showlegend=showLegend, subplot= subplot, data = trace)
            indx += 1


