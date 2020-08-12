import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from time_series_transform.stock_transform.base import *
from time_series_transform.stock_transform.util import *
from time_series_transform.transform_core_api.util import *

class Plot(object):
    def __init__(self, stock):
        """
        Plot uses the stock data to create various plots

        Parameters
        ----------
        stock : Stock
            stock data to create the plot
        """
        self._checkStock(stock)
        self.stock = stock
        self.fig = self._candleplot()
        self._plots = {
            'y' : ['candleplot'],
            'y2' : ['volume']
        }
        self._subplots = {}

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


    def get_all_subplots(self):
        """
        get_all_subplots returns a list of current subplots

        Returns
        -------
        subplots
            list of the current subplots
        """
        return list(self._subplots.keys())
    
    def show(self):
        """
        show the current plot
        """
        self.fig.show()

    def add_line(self, colName, color, legendName, showLegend = True, subplot = 'y', data = None):
        """
        add_line add line to the current plot

        Parameters
        ----------
        colName : str
            the column name of the stock data
        color : str
            color of the line
        legendName : str
            legend name of the line
        showLegend : bool, optional
            show legend, by default True
        subplot : str, optional
            subplot label, by default 'y'
            for example, y, y2, y3...
        data : list, optional
            array of the data to plot, by default None
            either data or colName is required
        """
        for li in list(self._plots.values()):
            if legendName in li:
                raise ValueError("duplicated legendName or indicator")
        if subplot not in list(self._plots.keys()):
            raise ValueError("subplot does not exist")

        if data is None:
            data = self.stock.df[colName]
        
        self._plots[subplot].append(legendName)

        self.fig.add_trace(
            go.Scatter(
                x= self.stock.df['Date'],
                y= data,
                mode="lines",
                line=go.scatter.Line(color=color),
                showlegend= showLegend,
                yaxis = subplot,
                name = legendName)
            )

    def add_macd(self):
        """
        add the moving average convergence divergence plot

        Raises
        ------
        ValueError
            raise exception when the macd is already in the plot
        """
        if 'macd' in list(self._subplots.keys()):
            raise ValueError("macd already exists")
        macd_data = macd(self.stock.df['Close'])
        macd_line_data = {'DIF':macd_data['DIF'], 'DEM':macd_data['DEM'], 'macdBase1': np.zeros(macd_data['DEM'].shape[0])}
        
        axis_num = self._find_next_layer()
        self._add_subplot_layer()
        self._add_multi_trace(macd_line_data, ['#a0bbe8', '#ff6767', 'grey'], axis_num)
        self.fig.add_trace(dict( x=self.stock.df['Date'], y=macd_data['OSC'],                         
                                showlegend = False,
                                type='bar', yaxis=axis_num, name='osc' ))
        self._plots[axis_num].append('osc')
        self._subplots['macd'] = axis_num

    def add_stochastic_oscillator(self):
        """
        add the stochastic_oscillator plot

        Raises
        ------
        ValueError
            raise exception when the macd is already in the plot
        """
        if 'stochastic_oscillator' in list(self._subplots.keys()):
            raise ValueError("stochastic_oscillator already exists")
        so_data = stochastic_oscillator(self.stock.df['Close'])
        
        axis_num = self._find_next_layer()
        self._add_subplot_layer()
        self._add_multi_trace(so_data, ['red', 'grey'], axis_num)
        self._subplots['stochastic_oscillator'] = axis_num

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
            self.add_line(colName = None, color = colors[indx], legendName = i,showLegend=showLegend, subplot= subplot, data = trace)
            indx += 1

    def _add_subplot_layer(self):
        newlayer = self._find_next_layer()
        self._plots[newlayer] = list()
        self._update_layout()
        
    
    def _update_layout(self):
        layoutNum = len(self._plots.keys())
        offset = 0.05 * (layoutNum-2)  + 0.15 * (layoutNum -2) + 0.1

        layout = {}
        for num in range(1,layoutNum+1):
            if num == 1:
                layout['yaxis'] = dict( domain = [round(offset, 2), 0.85])
            elif num == 2:
                offset -= 0.1
                layout['yaxis2'] = dict( domain = [round(offset, 2), round(offset + 0.1,2)], showticklabels = False )
            else:
                offset -= 0.2
                layout['yaxis' + str(num)] = dict( domain = [round(offset, 2), round(offset + 0.15,2)])
        
        self.fig.update_layout(layout)

    def remove_line(self, legendName):
        """
        remove the line by the given legendName from the plot

        Parameters
        ----------
        legendName : str
            legend name of the line

        Raises
        ------
        ValueError
            raise when it is trying to remove the default candle plot
        """
        default_plot = [self.stock.symbol, 'volume', 'candleplot']
        if legendName in default_plot:
            raise ValueError("Cannot remove default plot")

        try:
            remove_indx = [i for i in range(len(self.fig.data)) if self.fig.data[i]['name'] == legendName][0]
            new_data = [self.fig.data[i] for i in range(len(self.fig.data)) if i != remove_indx]
            self.fig.data = new_data
            layer_loc = [ix for i, ix in enumerate(self._plots) if legendName in self._plots[ix]][0]
            if layer_loc in self._plots:
                if len(self._plots[layer_loc]) == 0:
                    self._remove_layer(layer_loc)
        except:
            print(legendName + ' not exist')
       

    def _remove_layer(self, layer_loc):
        deleted_item = self._plots[layer_loc]
        for i in deleted_item:
            self.remove_line(i)
        self._plots.pop(layer_loc)
        self.fig.layout.pop('yaxis' + layer_loc[1:])
        self._update_layout()

    def remove_subplot(self, subplotName):
        """
        remove subplot from the plot

        Parameters
        ----------
        subplotName : str
            subplot name deleted

        Raises
        ------
        ValueError
            raise when the subplot does not exist
        """
        if subplotName not in self._subplots:
            raise ValueError(subplotName + " does not exist")
        layer = self._subplots[subplotName]
        self._remove_layer(layer)
        self._subplots.pop(subplotName)


