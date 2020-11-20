import abc
import datetime

class engine_interface(metaclass = abc.ABCMeta):
    """
    Interface of the stock API engine

    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'getHistoricalByPeriod') and 
                callable(subclass.samgetHistoricalByPeriodple) and
                hasattr(subclass, 'getHistoricalByRange') and 
                callable(subclass.getHistoricalByRange) and
                hasattr(subclass, 'getAdditionalInfo') and 
                callable(subclass.samgetAdditionalInfople) or 
                NotImplemented)
            
    @abc.abstractmethod
    def getHistoricalByPeriod(self, period):
        """
        get historical data by period 

        Parameters
        ----------
        period : str
            period of the stock data 
            valid input sample: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max

        Returns
        -------
        stock data

        """
        return NotImplemented

    @abc.abstractmethod
    def getHistoricalByRange(self, start_date, end_date):
        """
        get historical data by date range 

        Parameters
        ----------
        start_date, end_date: str
        range of the stock, format: "%Y-%m-%d", eg "2020-02-20"


        Returns
        -------
        stock data

        """
        return NotImplemented

    @abc.abstractmethod
    def getAdditionalInfo(self):
        """
        Additional information fetched from the selected stock

        Returns
        -------
        additional data
            dictionary of additional information, e.g. company information
        """
        return NotImplemented

def valid_period_format(date_string):
    format = "%Y-%m-%d"

    try:
        datetime.datetime.strptime(date_string, format)
        return True
    except ValueError:
        return False