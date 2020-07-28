import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from time_series_transform.stock_transform.base import *

class Plot(object):
    def __init__(self, stock):
        self._checkStock(stock)
        self.stock = stock
        self.fig = self._candleplot()

    def _checkStock(self, object):
        if isinstance(object,Stock):
            return
        else:
            raise ValueError('object is not stock')

    def _candleplot(self):
        df = self.stock.df
        colors = []
        INCREASING_COLOR = '#008000'
        DECREASING_COLOR = '#FF0000'

        data=[dict(type='candlestick',
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                yaxis = 'y',
                name = self.stock.symbol)]

        colors = [DECREASING_COLOR if df['Close'][i] < df['Close'][i-1] else INCREASING_COLOR for i in range(1,len(df['Close']))]
        colors.insert(0,DECREASING_COLOR)
                
        data.append( dict( x=df['Date'], y=df['Volume'],                         
                                marker=dict( color=colors ),
                                type='bar', yaxis='y2', name='Volume' ) )
        
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
        ret.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]), 
                dict(values=["2015-12-25", "2016-01-01"])
            ]
        )
        return ret


    def show(self):
        self.fig.show()

    def add_line(self, colName, color, legendName, subplot = 'y', data = None):
        if data is None:
            data = self.stock.df[colName]

        self.fig.add_trace(
            go.Scatter(
                x= self.stock.df['Date'],
                y= data,
                mode="lines",
                line=go.scatter.Line(color=color),
                showlegend= legendName is not None,
                yaxis = subplot,
                name = legendName)
            )

    def add_macd(self):
        macd_data = macd(self.stock.df['Close'])
        macd_line_data = {'DIF':macd_data['DIF'], 'DEM':macd_data['DEM'], 'ignore': np.zeros(macd_data['DEM'].shape[0])}
        
        self._add_subplot_layout()
        self._add_multi_trace(macd_line_data, ['#a0bbe8', '#ff6767', 'grey'], 'y3')
        self.fig.add_trace(dict( x=self.stock.df['Date'], y=macd_data['OSC'],                         
                                showlegend = False,
                                type='bar', yaxis='y3', name='osc' ))
        
    def _add_multi_trace(self, data, colors, subplot):
        indx = 0
        for i in data:
            trace = data[i]
            if i.find('ignore') > 0 :
                i = None
            self.add_line(colName = None, color = colors[indx], legendName = i, subplot= subplot, data = trace)
            indx += 1

    def _add_subplot_layout(self):
        axes = [self.fig.layout[e] for e in self.fig.layout if e[0:5] == 'yaxis']
        layoutNum = len(axes)
        offset = 0.05 * (layoutNum-1)  + 0.15 * (layoutNum -1) + 0.1

        layout = {
            "xaxis" + str(layoutNum): dict( domain = [0,1.0], rangeselector = dict( visible = False ) )
        }

        for num in range(1,layoutNum+2):
            if num == 1:
                layout['yaxis'] = dict( domain = [round(offset, 2), 0.85])
            elif num == 2:
                offset -= 0.1
                layout['yaxis2'] = dict( domain = [round(offset, 2), round(offset + 0.1,2)], showticklabels = False )
            else:
                offset -= 0.2
                layout['yaxis' + str(num)] = dict( domain = [round(offset, 2), round(offset + 0.15,2)])
        
        self.fig.update_layout(layout)
        return 'y' + str(layoutNum +1)


    # def _create_go_obj(self, **kwargs):
    #     ret = {}
