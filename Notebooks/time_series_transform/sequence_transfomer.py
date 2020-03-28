import pandas as pd
import numpy as np
import collections

class Sequence_Transformer_Base(object):

    def Call (self):
        raise NotImplementedError()

class Moving_Average(Sequence_Transformer_Base):
    def __init__(self,window_size):
        super().__init__()
        self.window_size = window_size

    def Call(self,arr):
        return np.pad(
            self._moving_average(arr,self.window_size),
            (self.window_size-1,0),'constant',
            constant_values = np.nan
        )

    def _moving_average(self,a,n) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n