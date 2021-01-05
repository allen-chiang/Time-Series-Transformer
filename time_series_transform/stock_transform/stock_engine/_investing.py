import investpy
from datetime import date, timedelta
from dateutil.relativedelta import *
from time_series_transform.stock_transform.stock_engine.engine_interface import *
import numpy as np

class investing(engine_interface):

    """
    Fetching finance data from investing.com
    
    API Document: 
    - https://investpy.readthedocs.io/introduction.html
    ---
    Require:
    - investpy

    ---
    Citation:
        author = Alvaro Bartolome del Canto,
        title = investpy - Financial Data Extraction from Investing.com with Python,
        year = 2018-2020,
        publisher = GitHub,
        journal = GitHub Repository,
        published = https://github.com/alvarobartt/investpy
    """

    
    def __init__(self,symbol, country, product_type = 'stocks'):
        self.symbol = symbol
        self.country = country
        self.product_type = product_type
    

    # interface function implementations
    def getHistoricalByPeriod(self, period):
        end_date = date.today()
        if period =='max':
            start = '1/1/1920'
        else:
            t_delta = {
                'd' : 0,
                'mo' : 0,
                'y' : 0
            }
            if period == 'ytd':
                t_delta['y'] = 1
            else:
                indx = -1
                if 'mo' in period:
                    indx = -2
            
            t_delta[period[indx:]] = int(period[:indx])
            start_date = end_date - relativedelta(years=t_delta['y'], months=t_delta['mo'], days=t_delta['d'])
            start = start_date.strftime('%d/%m/%Y')
        end = end_date.strftime('%d/%m/%Y')
        return self.getHistoricalData(start, end)

    def getHistoricalByRange(self, start_date, end_date):
        if valid_period_format(start_date) and valid_period_format(end_date):
            start = datetime.datetime.strptime(start_date, '%Y-%m-%d').strftime('%d/%m/%Y')
            end = datetime.datetime.strptime(end_date, '%Y-%m-%d').strftime('%d/%m/%Y')
            return self.getHistoricalData(start, end)
        else:
            raise ValueError('date format must be YYYY-mm-dd')

    def getAdditionalInfo(self):
        info_dict = {
            'company_info' : self.getCompanyInfo(),
            'financial_statement': self.getFinancialSummary()
        }

        schedule_dict = {
            'dividend': self.getDividends()
        }

        data = {
            'info': info_dict,
            'schedule': schedule_dict
            
        }
        return data

    def getIntraDayData(self, start_date, end_date, interval_range):
        raise NotImplementedError("The function is not available in the investing engine, please switch to yahoo")

    def getAllStocks(self):
        return investpy.stocks.get_stocks()
    
    def getAllCountries(self):
        return investpy.stocks.get_stock_countries()

    def getCountryStockOverview(self, as_json = False, n_results = 100):
        return investpy.stocks.get_stocks_overview(self.country, as_json, n_results)


    def getCompanyInfo(self, as_json = False):
        data = investpy.stocks.get_stock_information(self.symbol, self.country, as_json = as_json)
        data = data.set_index('Stock Symbol')
        data = data.to_dict('r')
        return data[0]


    def getHistoricalData(self, start_date, end_date, as_json = False, order = 'ascending', interval = 'Daily'):
        if self.findBusinessDay(start_date,end_date)>0:
            return investpy.stocks.get_stock_historical_data(self.symbol, self.country, start_date, end_date, as_json, order, interval)
        else:
            raise ValueError("Input date are not businessday")

    def getFinancialSummary(self, summary_type = 'income_statement', period = 'annual'):
        try:
            return investpy.stocks.get_stock_financial_summary(self.symbol, self.country, summary_type, period)
        except:
            return None
        
    
    def getDividends(self):
        try:
            return investpy.stocks.get_stock_dividends(self.symbol, self.country)
        except:
            return None
    
    def findBusinessDay(self,start,end):
        start = datetime.datetime.strptime(start, '%d/%m/%Y').date()
        end = datetime.datetime.strptime(end, '%d/%m/%Y').date()
        days = np.busday_count( start, end )
        return days