import numpy as np
from time_series_transform.plot.base import plot_base
import plotly.graph_objects as go

class TimeSeriesPlot(plot_base):
    def __init__(self, time_series_data):
        super().__init__(time_series_data)

    def create_plot(self, dataCols,title = "", type='Scatter'):
        for col in dataCols:
            self.add_line(col, type)
        
        self.fig.update_layout(
            title = title,
            xaxis_title = self.time_series.time_seriesIx,
            yaxis_title = "value",
            legend_title = "Legend"
        )

        return self

    def add_line(self, col, type):
        fig = self.fig
        data = self.data[col]
        time_indx = self.time_index_data

        fig_data = (dict(type = type,
                        x = time_indx, 
                        y = data, 
                        mode= 'lines', 
                        name = col,
                        showlegend=True))

        
        fig.add_trace(fig_data)

    def __repr__(self):
        self.fig.show()
        return ""
    
def create_plot(time_series_data, dataCols, type='Scatter'):
    tsp = TimeSeriesPlot(time_series_data)
    tsp.create_plot(dataCols,type= type)
    return tsp

    