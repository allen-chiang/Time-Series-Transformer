from numpy.fft import *
import numpy as np
import pywt

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


def madev(d, axis=None):
    """ Mean absolute deviation """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def wavelet_denoising(arr, wavelet='db4', level=1):
    coeff = pywt.wavedec(arr, wavelet, mode="per")
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(arr)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')
