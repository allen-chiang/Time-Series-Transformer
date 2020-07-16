from numpy.fft import *
import pandas as pd
import numpy as np

def moving_average(arr, windowSize=3) :
    ret = np.cumsum(arr, dtype=float)
    ret[windowSize:] = ret[windowSize:] - ret[:-windowSize]
    ret = ret[windowSize - 1:] / windowSize
    res = np.empty((int(len(arr)-len(ret))))
    res[:] = np.nan
    return np.append(res,ret) 


def rfft_transform(arr, threshold=1e3):
    fourier = rfft(arr)
    frequencies = rfftfreq(arr.size, d=20e-3/arr.size)
    fourier[frequencies > threshold] = 0
    fourier =  irfft(fourier)
    res = np.empty((int(len(arr)-len(fourier))))
    res[:] = np.nan
    return np.append(res,fourier) 

def ema(arr, com = None, span = None, halflife = None, alpha = None, adjust = True, min_periods = 0, ignore_na = False, axis = 0):
    return arr.ewm(com = com, span = span, halflife = halflife, alpha = alpha, min_periods = min_periods, adjust = adjust, ignore_na = ignore_na, axis = axis)



