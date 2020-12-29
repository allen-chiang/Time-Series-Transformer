from time_series_transform.transform_core_api.base import (
    Time_Series_Data,
    Time_Series_Data_Collection
    )
import numpy as np
import plotly.graph_objects as go

class plot_base(object):
    """plot_base is the base class for the plot engine, user is able to call 
    the plot function from the transformer or use the create_plot() function
        
    """
    def __init__(self, time_series):
        if isinstance(time_series, Time_Series_Data_Collection):
            self.category = list(time_series.time_series_data_collection.keys())
            self.time_series = time_series
            self.time_index_data = time_series[self.category[0]].time_index[time_series[self.category[0]].time_seriesIx]
            self.fig = go.Figure()
            self.is_collection = True
        elif isinstance(time_series, Time_Series_Data):
            self.time_series = time_series
            self.data = time_series[:]
            self.time_index_data = time_series.time_index[time_series.time_seriesIx]
            self.fig = go.Figure()
            self.is_collection = False
        else:
            raise ValueError("Input data must be Time_Series_Data")

    def add_line(self, col, lineType, showlegend = True, color = 'default',**kwargs):
        """add_line add line to the current plot

        Parameters
        ----------
        col : str
            name of the column from the time_series_data
        lineType : str
            type of the line, e.g. "scatter", "bar"
        showlegend : bool, optional
            show the legend, by default True
        color : str, optional
            color of the line, by default 'default'
        """
        fig = self.fig
        time_indx = self.time_index_data
        if self.is_collection:
            for cat in self.category:
                data = self.time_series[cat].data[col]
                fig_data = (dict(type = lineType.lower(),
                                            x = time_indx, 
                                            y = data,
                                            name = cat+"_"+col,
                                            showlegend=showlegend, **kwargs))
                fig.add_trace(fig_data)

        else:
            data = self.data[col]

            if color == 'default':
                fig_data = (dict(type = lineType.lower(),
                            x = time_indx, 
                            y = data, 
                            name = col,
                            showlegend=showlegend,**kwargs))
            else:
                fig_data = (dict(type = lineType.lower(),
                            x = time_indx, 
                            y = data, 
                            name = col,
                            showlegend=showlegend,
                            line = dict(color = color),**kwargs))

            
            fig.add_trace(fig_data)

    def update_layout(self, **kwargs):
        """update the layout of the plot

        Returns
        -------
        plot
            it will return the plot class object with the updated layout
        """
        self.fig.update_layout(**kwargs)
        return self

    def __repr__(self):
        self.fig.show()
        return ""