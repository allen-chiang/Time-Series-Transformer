from time_series_transform.transform_core_api.util import *
from time_series_transform.transform_core_api.base import *
from time_series_transform.io import *
import pandas as pd

def _arr_check(arr):
    if len(arr) == 0:
        raise ValueError("Data can not have zero length")

    
def macd(arr, return_diff = False):
    """
    Return the moving average convergence/divergence 
    
    Parameters
    ----------
    arr : array 
        data used to calculate the macd

    return_diff : bool 
        return difference between DIF and DEM if True

    Returns
    -------
    df : dict
        macd of the given data, including EMA_12, EMA_26, DIF, DEM, OSC
    """
    _arr_check(arr)
    df = {}
    df['EMA_12'] = ema(arr, span=12).flatten()
    df['EMA_26'] = ema(arr, span=26).flatten()

    df['DIF'] = df['EMA_12'] - df['EMA_26']
    df['DEM'] = ema(pd.DataFrame(df['DIF']), span=9).flatten()
    df['OSC'] = df['DIF'] - df['DEM']
    if return_diff:
        return df['OSC']
    else:
        return df

def stochastic_oscillator(arr, window = 14):
    """
    Return the stochastic oscillator 
    
    Parameters
    ----------
    arr : pandas or Time_Series_Data 
        data used to calculate the stochastic oscillator

    window : int
        window of the stochastic oscillator
    Returns
    -------
    ret : dict
        stochastic oscillator of the given data, including k and d values
    """
    if not isinstance(arr, Time_Series_Data) and not isinstance(arr, pd.DataFrame):
        raise ValueError("Input must be either Time_Series_Data or Pandas")

    if isinstance(arr, Time_Series_Data):
        df = to_pandas(arr, False, None, False)
    else:
        df = arr
    ret = {}
    df['Low_window'] = df['Low'].rolling(window = window).min()
    df['High_window'] = df['High'].rolling(window = window).max()

    ret['k_val'] = 100*((df['Close'] - df['Low_window']) / (df['High_window'] - df['Low_window']))
    ret['d_val'] = np.array(ret['k_val'].rolling(window = 3).mean()).reshape(-1)
    ret['k_val'] = np.array(ret['k_val']).reshape(-1)

    return ret

def rsi(arr, n_day = 14):
    """
    Return the Relative Strength Index 
    
    Parameters
    ----------
    arr : array 
        data used to calculate the Relative Strength Index

    Returns
    -------
    rsi : array
        Relative Strength Index of the given array
    """
    _arr_check(arr)
    arr = np.array(arr)
    orgLen = len(arr)
    arr = arr[~np.isnan(arr)]

    dif = [arr[i+1]-arr[i] for i in range(len(arr)-1)]
    u_val = np.array([val if val>0 else 0 for val in dif])
    d_val = np.array([-1*val if val<0 else 0 for val in dif])

    u_ema = ema(u_val, span = n_day)
    d_ema = ema(d_val,span = n_day)
    rs = u_ema/d_ema
    rsi = 100*(1-1/(1+rs))
    rsi = rsi.reshape(-1)
    res = np.empty((int(orgLen-len(rsi))))
    res[:] = np.nan
    return np.append(res,rsi) 

def williams_r(arr, n_day=14):
    """
    Return the Williams %R index
    
    Parameters
    ----------
    arr : array 
        data used to calculate the Williams %R
    n_day : int
        window of the indicator

    Returns
    -------
    r_val : array
        Relative Strength Index of the given array
    """
    
    _arr_check(arr)
    arr = np.array(arr)
    orgLen = len(arr)
    arr = arr[~np.isnan(arr)]
    df = pd.DataFrame(arr)
    r_rolling = df.rolling(n_day)
    r_val = 100*(r_rolling.max()-df)/(r_rolling.max() - r_rolling.min())
    res = np.empty((int(orgLen-len(r_val))))
    res[:] = np.nan
    return np.append(res,r_val) 
