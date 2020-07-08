from ystock import ystock
import matplotlib.pyplot as plt

class Stock_Transformer(object):

    def __init__(self, symbol):
        """this class transform the stock data into various format
        
        Parameters
        ----------
        symbol : string
            the input symbol of desired stock data
        
        """
        self.symbol = symbol
        self.df = ystock(symbol)

    def get_company_info(self):
        print(self.df.getCompanyInfo())
        return self.df.getCompanyInfo()

    def _get_historical_data(self, period, start= None, end = None):
        result = self.df.getHistoricalByPeriod(period)
        if (start and end):
            result = self.df.getHistoricalByRange(start_date=start, end_date=end)
        return result
    
    def download_historical_data_csv(self, path, period="1y", start= None, end = None):
        res = self._get_historical_data(period, start, end)
        res.to_csv (path, index = False, header=True)

    def get_stock_data_pandas(self, period="1y", start= None, end = None):
        res = self._get_historical_data(period, start, end)
        return res[['Open', 'High', 'Low','Close', 'Volume']]
    
    def get_stock_data_np(self, period="1y", start= None, end = None):
        res = self._get_historical_data(period, start, end).reset_index()
        header = ['Date','Open', 'High', 'Low','Close', 'Volume']
        return header, res[header].to_numpy()

    
class Standard_Indicator(object):

    def __init__(self, symbol):
        """this class output the standard analysis
        
        Parameters
        ----------
        symbol : string
            the input symbol of desired stock data
        
        """
        self.symbol = symbol
        self.df = Stock_Transformer(symbol)

    def moving_average(self, window, period="1y", start= None, end = None):
        data = self.df.get_stock_data_pandas(period, start, end)
        ma = data.rolling(window).mean().dropna()
        return ma
    
    def ma_plot(self, windows, period="1y", start= None, end = None):
        for window in windows:
            ma = self.moving_average(window, period, start, end)
            plt.plot(ma['Close'])
        plt.show()
    
    