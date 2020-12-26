from time_series_transform.transform_core_api.base import (
    Time_Series_Data,
    Time_Series_Data_Collection
    )
import numpy as np
import plotly.graph_objects as go

class plot_base(object):
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

    def add_line(self, col, lineType, showlegend = True, color = 'default'):
        fig = self.fig
        time_indx = self.time_index_data
        if self.is_collection:
            for cat in self.category:
                data = self.time_series[cat].data[col]
                fig_data = (dict(type = lineType.lower(),
                                            x = time_indx, 
                                            y = data, 
                                            mode= 'lines', 
                                            name = cat+"_"+col,
                                            showlegend=showlegend))
                fig.add_trace(fig_data)

        else:
            data = self.data[col]

            if color == 'default':
                fig_data = (dict(type = lineType.lower(),
                            x = time_indx, 
                            y = data, 
                            mode= 'lines', 
                            name = col,
                            showlegend=showlegend))
            else:
                fig_data = (dict(type = lineType.lower(),
                            x = time_indx, 
                            y = data, 
                            mode= 'lines', 
                            name = col,
                            showlegend=showlegend,
                            line = dict(color = color)))

            
            fig.add_trace(fig_data)

    def update_layout(self, **kwargs):
        self.fig.update_layout(**kwargs)
        return self