import numpy as np
from time_series_transform.io.base import io_base
from time_series_transform.transform_core_api import Time_Series_Data,Time_Series_Data_Collection


class Numpy_IO (io_base):
    def __init__(self, time_series, timeSeriesColIx, mainCategoryColIx):
        super().__init__(time_series, timeSeriesColIx, mainCategoryColIx)
        self.time_series = {}
        for i in range(len(time_series.T)):
            self.time_series[i] = time_series.T[i]

    def from_numpy (self):
        if self.mainCategoryCol is None:
            return self.to_single(self.time_series,self.timeSeriesCol)
        return self.to_collection(self.time_series,self.timeSeriesCol,self.mainCategoryCol)

    def to_numpy(self,expandTime,expandCategory,preprocessType):
        if isinstance(self.time_series,Time_Series_Data):
            data = self.from_single(expandTime)
        if isinstance(self.time_series,Time_Series_Data_Collection):
            data = self.from_collection(expandCategory,expandTime,preprocessType)
        return np.asarray(list(data.values()))


def from_numpy(numpyArray,timeSeriesCol,mainCategoryCol):
    if not isinstance(numpyArray,np.ndarray):
        raise ValueError('input data must be numpy array')
    numpyio = Numpy_IO(numpyArray,timeSeriesCol,mainCategoryCol)
    return numpyio.from_numpy()

def to_numpy(time_series_data,expandCategory,expandTime,preprocessType):
    if isinstance(time_series_data,Time_Series_Data):
        numpyio = Numpy_IO(time_series_data,time_series_data.time_seriesIx,None)
        return numpyio.to_numpy(expandTime,None,None)
    if isinstance(time_series_data,Time_Series_Data_Collection):
        numpyio = Numpy_IO(
            time_series_data,
            time_series_data._time_series_Ix,
            time_series_data._categoryIx
            )
    return numpyio.to_numpy(expandCategory,expandTime,preprocessType)    


__all__= [
    'from_numpy',
    'to_numpy'
]