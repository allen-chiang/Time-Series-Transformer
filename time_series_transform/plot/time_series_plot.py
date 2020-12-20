import numpy as np
from time_series_transform.plot.base import plot_base
import plotly.graph_objects as go

class TimeSeriesPlot(plot_base):
    def __init__(self, time_series_data):
        super().__init__(time_series_data)
        self.fig = None

    def create_plot(self, dataCol, type='Scatter'):
        fig = go.Figure([go.Scatter(x=self.time_index_data, y=self.data[dataCol])])
        self.fig = fig

        return self

    def __repr__(self):
        self.fig.show()
        return ""
    
def create_plot(time_series_data, dataCol, type='Scatter'):
    tsp = TimeSeriesPlot(time_series_data)
    tsp.create_plot(dataCol,type= type)
    return tsp

    