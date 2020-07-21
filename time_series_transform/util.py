from numpy.fft import *
import pandas as pd
import numpy as np
import pywt

def moving_average(arr, windowSize=3) :
    orgLen = len(arr)
    arr = arr[~np.isnan(arr)]
    ret = np.cumsum(arr, dtype=float)
    ret[windowSize:] = ret[windowSize:] - ret[:-windowSize]
    ret = ret[windowSize - 1:] / windowSize
    res = np.empty((int(orgLen-len(ret))))
    res[:] = np.nan
    return np.append(res,ret) 


def rfft_transform(arr, threshold=1e3):
    orgLen = len(arr)
    arr = arr[~np.isnan(arr)]
    fourier = rfft(arr)
    frequencies = rfftfreq(arr.size, d=20e-3/arr.size)
    fourier[frequencies > threshold] = 0
    fourier =  irfft(fourier)
    res = np.empty((int(orgLen-len(fourier))))
    res[:] = np.nan
    return np.append(res,fourier) 


def madev(d, axis=None):
    """ Mean absolute deviation """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def wavelet_denoising(arr, wavelet='db4',coeff_mode = "per",threshold_mode='hard',rec_mode='per', level=1,matchOriginLenth=True):
    orgLen = len(arr)
    arr = arr[~np.isnan(arr)]
    coeff = pywt.wavedec(arr, wavelet, mode=coeff_mode)
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(arr)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode=threshold_mode) for i in coeff[1:])
    wav = pywt.waverec(coeff, wavelet, mode=rec_mode)
    if matchOriginLenth:
        if len(wav) > orgLen:
            return wav[:len(arr)]
        else:
            res = np.empty((int(orgLen-len(wav))))
            res[:] = np.nan
            return np.append(res,wav) 
    else:
        return wav



def differencing(arr,order =1):
    for i in range(order):
        if i == 0:
            diff = np.diff(arr)
        else:
            diff = np.diff(diff)
    res = np.empty((int(len(arr)-len(diff))))
    res[:] = np.nan
    return np.append(res,diff) 
def ema(arr, com = None, span = None, halflife = None, alpha = None, adjust = True, min_periods = 0, ignore_na = False, axis = 0):
    return arr.ewm(com = com, span = span, halflife = halflife, alpha = alpha, min_periods = min_periods, adjust = adjust, ignore_na = ignore_na, axis = axis)



