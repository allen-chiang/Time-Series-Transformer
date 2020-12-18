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
    close = pd.DataFrame(arr)
    delta = close.diff() 
    up, down = delta.copy(), delta.copy()

    up[up < 0] = 0
    down[down > 0] = 0
    
    roll_up = up.ewm(com=n_day - 1, adjust=False).mean()
    roll_down = down.ewm(com=n_day - 1, adjust=False).mean().abs()
    
    rs = roll_up / roll_down

    rsi = 100-(100/(1+rs))
    rsi = np.array(rsi).reshape(-1)
    return rsi

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
    
    if not isinstance(arr, Time_Series_Data) and not isinstance(arr, pd.DataFrame):
        raise ValueError("Input must be either Time_Series_Data or Pandas")

    if isinstance(arr, Time_Series_Data):
        df = to_pandas(arr, False, None, False)
    else:
        df = arr

    r_rolling = df.rolling(n_day)
    highest = r_rolling.max()['High']
    lowest = r_rolling.min()['Low']
    r_val = -100 * (highest - df['Close']) / (highest - lowest)
    r_val = np.array(r_val).reshape(-1)

    return r_val
