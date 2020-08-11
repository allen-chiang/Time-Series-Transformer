from numpy.fft import *
import pandas as pd
import numpy as np
import pywt
from scipy.stats.mstats import gmean


def moving_average(arr, windowSize=3) :
    """
    moving_average the arithimetic moving average
    
    Given the window size, this function will perform simple moving average
    
    Parameters
    ----------
    arr : numpy array
        the input array
    windowSize : int, optional
        the grouping size, by default 3
    
    Returns
    -------
    numpy array
        the moving average array
    """
    orgLen = len(arr)
    arr = arr[~np.isnan(arr)]
    ret = np.cumsum(arr, dtype=float)
    ret[windowSize:] = ret[windowSize:] - ret[:-windowSize]
    ret = ret[windowSize - 1:] / windowSize
    res = np.empty((int(orgLen-len(ret))))
    res[:] = np.nan
    return np.append(res,ret) 


def rfft_transform(arr, threshold=1e3):
    """
    rfft_transform real fast fourier transformation
    
    Fast fourier trnasformation and ignoring the imagine number
    note: numpy implmentation

    Parameters
    ----------
    arr : numpy array
        input array
    threshold : float, optional
        the threshold used for filter frequency, by default 1e3
    
    Returns
    -------
    numpy array
        rfft array
    """
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
    """
    wavelet_denoising wavelet transformation
    
    wavelet transformation, with pywt implmentation
    
    Parameters
    ----------
    arr : numpy array
        input array
    wavelet : str, optional
        wavelet transform family, by default 'db4'
    coeff_mode : str, optional
        the coefficient mode, by default "per"
    threshold_mode : str, optional
        the threshold tye, by default 'hard'
    rec_mode : str, optional
        recover mode, by default 'per'
    level : int, optional
        sigma level for theshold, by default 1
    matchOriginLenth : bool, optional
        whether to match the input array length, by default True
    
    Returns
    -------
    numpy array
        wevelet transformed array
    """
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
    """
    differencing time series differencing
    
    it simply perform series differencing
    For example:
    order 1
        Xt, Xt+1 --> Xt+1 - Xt
    order 2
        Xt, Xt+1, Xt+2 --> Xt+1 - Xt, Xt+2-Xt+1 = a,b --> b - a 
    and so on
    
    Parameters
    ----------
    arr : numpy array
        input array
    order : int, optional
        number of differencing, by default 1
    
    Returns
    -------
    numpy array
        differenced array
    """
    for i in range(order):
        if i == 0:
            diff = np.diff(arr)
        else:
            diff = np.diff(diff)
    res = np.empty((int(len(arr)-len(diff))))
    res[:] = np.nan
    return np.append(res,diff) 

    
def ema(arr, com = None, span = None, halflife = None, alpha = None, adjust = True, min_periods = 0, ignore_na = False, axis = 0):
    """
    this is the panads ema implmentation
    """
    df = pd.DataFrame(arr)
    return df.ewm(com = com, span = span, halflife = halflife, alpha = alpha, min_periods = min_periods, adjust = adjust, ignore_na = ignore_na, axis = axis).mean().to_numpy()


def geometric_ma(arr,windowSize):
    """
    geometric_ma geometric moving average
    
    it use pandas rolling window with sicpy gmean function
    
    Parameters
    ----------
    arr : numpy array
        input arrray
    windowSize : int
        grouping size
    
    Returns
    -------
    numpy array
        geometric moving average array
    """
    return pd.Series(arr).rolling(window=windowSize).apply(gmean)