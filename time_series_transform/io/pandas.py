import pandas as pd
from time_series_transform.transform_core_api.base import Time_Series_Data, Time_Series_Data_Colleciton


def from_pandas(pandasFrame,timeSeriesCol,mainCategoryCol=None):
    tsd = Time_Series_Data()
    if timeSeriesCol is None:
        raise KeyError("time series index is required")
    data = pandasFrame.to_dict('list')
    tsd.set_time_index(data[timeSeriesCol],timeSeriesCol)
    for i in data:
        if i == timeSeriesCol:
            continue
        tsd.set_data(data[i],i)
    if mainCategoryCol is None:
        return tsd
    else:
        tsc = Time_Series_Data_Colleciton(tsd,timeSeriesCol,mainCategoryCol)
        return tsc


def to_pandas():
    pass