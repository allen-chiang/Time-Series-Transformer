from time_series_transform.transform_core_api.base import (
    Time_Series_Data,
    Time_Series_Data_Collection
    )
import numpy as np
import plotly.graph_objects as go
from copy import copy




class plot_base(object):
    """plot_base is the base class for the plot engine, user is able to call 
    the plot function from the transformer or use the create_plot() function
        
    """
    def __init__(self, time_series):
        if isinstance(time_series, Time_Series_Data_Collection):
            self.category = list(time_series.time_series_data_collection.keys())
            self.time_series = time_series
            self.time_index_data = time_series[self.category[0]].time_index[time_series[self.category[0]].time_seriesIx]
            self.is_collection = True
        elif isinstance(time_series, Time_Series_Data):
            self.time_series = time_series
            self.time_index_data = time_series.time_index[time_series.time_seriesIx]
            self.is_collection = False
        else:
            raise ValueError("Input data must be Time_Series_Data")

        self.fig = go.Figure()
        self._plots = {
            'y' : []
        }

    def get_current_plots(self):
        """return a dictionary of the current plots and corresponding lines

        Returns
        -------
        dict
            current plots and lines
        """
        return self._plots
        
    def add_line(self, lineType, col = None, data = None, legendName = None, subplot = 'y', showlegend = True, color = "default",**kwargs):
        """add_line add line to the current plot

        Parameters
        ----------
        lineType : str
            type of the line, e.g. "scatter", "bar"
        col : str
            name of the column from the time_series_data, **either data or col is required**
        data : array
            array of the input data, **either data or col is required**
        legendName : str
            string of the legend name, default None
        subplot : str
            subplot of the line added, default 'y'
            format: 'y1', 'y2'....
        showlegend : bool, optional
            show the legend, by default True
        color : str, optional
            color of the line, by default None
        """
        if (data is None and col is None) or (data is not None and col is not None):
            raise ValueError("Either data or colName is required")
        for li in list(self._plots.values()):
            if legendName in li:
                raise ValueError("duplicated legendName or indicator")
        if data is not None and len(data) != len(self.time_index_data):
            raise ValueError("length of data must be the same as the length of time index")

        if legendName is None:
            if col is None:
                legendName = self._get_legend_name('default')
            else:
                legendName = self._get_legend_name(col)

        if subplot not in list(self._plots.keys()):
            subplot = self._find_next_layer()
            self._add_subplot_layer(subplot)

        fig = self.fig
        time_indx = self.time_index_data
        self._plots[subplot].append(legendName)

        if self.is_collection:
            for cat in self.category:
                if data is None:
                    data = self.time_series[cat].data[col]
                fig_data = (dict(type = lineType.lower(),
                                            x = time_indx, 
                                            y = data,
                                            name = cat+"_"+legendName,
                                            yaxis = subplot,
                                            showlegend=showlegend, **kwargs))
                fig.add_trace(fig_data)

        else:
            if data is None:
                data = self.time_series.data[col]
            if color == 'default':
                fig_data = (dict(type = lineType.lower(),
                            x = time_indx, 
                            y = data, 
                            name = legendName,
                            yaxis = subplot,
                            showlegend=showlegend,**kwargs))
            else:
                fig_data = (dict(type = lineType.lower(),
                            x = time_indx, 
                            y = data, 
                            name = legendName,
                            showlegend=showlegend,
                            yaxis = subplot,
                            line = dict(color = color),**kwargs))

            
            fig.add_trace(fig_data)

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
        try:
            remove_indx = [i for i in range(len(self.fig.data)) if self.fig.data[i]['name'] == legendName][0]
            new_data = [self.fig.data[i] for i in range(len(self.fig.data)) if i != remove_indx]
            self.fig.data = new_data
            layer_loc = [ix for i, ix in enumerate(self._plots) if legendName in self._plots[ix]][0]
            if layer_loc in self._plots:
                if len(self._plots[layer_loc]) == 0:
                    self._remove_layer(layer_loc)
        except:
            print(legendName + ' does not exist')

    def update_layout(self, **kwargs):
        """update the layout of the plot

        Returns
        -------
        plot
            it will return the plot class object with the updated layout
        """
        self.fig.update_layout(**kwargs)
        return self
    
    def _remove_layer(self, layer_loc):
        deleted_item = copy(self._plots[layer_loc])
        self._plots.pop(layer_loc)
        for i in deleted_item:
            self.remove_line(i)
        self.fig.layout.pop('yaxis' + layer_loc[1:])
        self._add_subplot_layout()

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
        if subplotName not in list(self._plots.keys()):
            raise ValueError(subplotName + " does not exist")
        self._remove_layer(subplotName)

    def add_marker(self,x,y,color,legendName,showlegend=True,marker='circle',**kwargs):
        """add_marker will add the marker of the shape on the plot 

        Parameters
        ----------
        x : array
            array of the x coordinates
        y : array
            array of the y coordinates
        color : str
            color of the marker
        legendName : str
            name of the legend
        showlegend : bool, optional
            show the legend, by default True
        marker : str, optional
            shape of the marker, by default 'circle'

        Returns
        -------
        self
            the current TimeSeriesPlot object
        """
        self.fig.add_trace(
            go.Scatter(
                mode='markers',
                x=x,
                y=y,
                name=legendName,
                marker=dict(
                    color = color,
                    symbol=marker
                ),
                showlegend=showlegend,**kwargs
            )
        )
        return self

    def _get_legend_name(self, legend):
        indx = 1
        legarr = []
        for li in list(self._plots.values()):
            legarr.extend(li)
        if legend not in legarr:
            return legend 
        while True:
            new_legend = legend+"_"+str(indx)
            if new_legend not in legarr:
                return new_legend
            else:
                indx+=1

    def _add_subplot_layer(self, layerName = None):
        if layerName is None:
            newlayer = self._find_next_layer()
        else:
            newlayer = layerName
        self._plots[newlayer] = list()
        self._add_subplot_layout()
        
    
    def _add_subplot_layout(self):
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

    def _find_next_layer(self):
        cur_max = 1
        for k in self._plots.keys():
            if len(k) > 1:
                cur_max = max(cur_max, int(k[1:]))
        
        return 'y' + str(cur_max + 1)

    def __repr__(self):
        self.fig.show()
        return ""

    def __call__(self):
        self.fig.show()

    def show(self):
        """
        show the current plot
        """
        self.fig.show()