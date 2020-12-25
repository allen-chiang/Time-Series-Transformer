import numpy as np
from time_series_transform.plot.base import plot_base
import plotly.graph_objects as go

class TimeSeriesPlot(plot_base):
    def __init__(self, time_series_data):
        super().__init__(time_series_data)

    def create_plot(self, dataCols,title = "", lineType='scatter'):
        for col in dataCols:
            self.add_line(col, lineType)
        
        self.update_layout(
            title = title,
            xaxis_title = self.time_series.time_seriesIx,
            yaxis_title = "value",
            legend_title = "Legend"
        )

        return self

    def __repr__(self):
        self.fig.show()
        return ""
    
def create_plot(time_series_data, dataCols,title = "", type='scatter'):
    tsp = TimeSeriesPlot(time_series_data)
    tsp.create_plot(dataCols,lineType= type)
    return tsp

    