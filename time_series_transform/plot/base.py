from time_series_transform.transform_core_api.base import (
    Time_Series_Data,
    Time_Series_Data_Collection
    )
import numpy as np
import plotly.graph_objects as go

class plot_base(object):
    def __init__(self, time_series):
        if isinstance(time_series, Time_Series_Data):
            self.time_series = time_series
            self.data = time_series[:]
            self.time_index_data = time_series.time_index[time_series.time_seriesIx]
            self.fig = go.Figure()
        else:
            raise ValueError("Input data is not Time_Series_Data")
