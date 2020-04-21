import os
from datetime import datetime
from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_data

class stock_repo:
    """
    Fetching stock data from IEXFinance
    
    API Document: https://iexcloud.io/docs/api/
    ---
    Require:
    - iexfinance
    - pandas
    - iexfinance API KEY
    """

    def __init__(self, api_key):
        self.api_key = api_key
    
    def getStock(self, symbols):
        stock_batch = Stock(symbols,token=self.api_key, output_format='pandas')
        return stock_batch

    def getCompanyInfo(self, symbols):
        stock_batch = Stock(symbols,token=self.api_key, output_format='pandas')
        company_info = stock_batch.get_company()
        return company_info

    def getBook(self, symbols):
        stock_batch = Stock(symbols,token=self.api_key, output_format='pandas')
        book = stock_batch.get_book()
        return book

    def getAllHistorical(self, symbols):
        stock_batch = Stock(symbols,token=self.api_key, output_format='pandas')
        chart = stock_batch.get_historical_prices()
        return chart

    def getHistoricalByRange(self, symbols, start_date, end_date):
        start = datetime(int(start_date[0:4]), int(start_date[5:6]), int(start_date[7:8]))
        end = datetime(int(end_date[0:4]), int(end_date[5:6]), int(end_date[7:8]))
        df = get_historical_data(symbols, start, end,token=self.api_key, output_format='pandas')
        return df

    def getEarnings(self, symbols):
        stock_batch = Stock(symbols,token=self.api_key, output_format='pandas')
        earnings = stock_batch.get_earnings()
        return earnings

    def getKeyStats(self, symbols):
        stock_batch = Stock(symbols,token=self.api_key, output_format='pandas')
        stats = stock_batch.get_key_stats()
        return stats

    def getNews(self, symbols):
        stock_batch = Stock(symbols,token=self.api_key, output_format='pandas')
        news = stock_batch.get_news()
        return news

    def getOHLC(self, symbols):
        stock_batch = Stock(symbols,token=self.api_key, output_format='pandas')
        ohlc = stock_batch.get_ohlc()
        return ohlc

    def getQuote(self, symbols):
        stock_batch = Stock(symbols,token=self.api_key, output_format='pandas')
        quote = stock_batch.get_quote()
        return quote

    def getSector(self, symbols):
        stock_batch = Stock(symbols,token=self.api_key, output_format='pandas')
        sector = stock_batch.get_sector()
        return sector

    def getOutstandingShares(self, symbols):
        stock_batch = Stock(symbols,token=self.api_key, output_format='pandas')
        shares = stock_batch.get_shares_outstanding()     
        return shares


################################################
### Premium Account Usage ###

    # def getBalanceSheet(self, symbols):
    #     stock_batch = Stock(symbols,token=self.api_key, output_format='pandas')
    #     balance_sheet = stock_batch.get_balance_sheet()
    #     return balance_sheet

    # def getRelevantStock(self, symbols):
    #     stock_batch = Stock(symbols,token=self.api_key, output_format='pandas')
    #     relevents = stock_batch.get_relevant_stocks()
    #     return relevents

    # def getPrice(self, symbols):
    #     stock_batch = Stock(symbols,token=self.api_key, output_format='pandas')
    #     price = stock_batch.get_price()
    #     return price

    # def getLargestTrades(self, symbols):
    #     stock_batch = Stock(symbols,token=self.api_key, output_format='pandas')
    #     trade = stock_batch.get_largest_trades()
    #     return trade
    
    # def getIncomeStatement(self, symbols):
    #     stock_batch = Stock(symbols,token=self.api_key, output_format='pandas')
    #     income = stock_batch.get_income_statement()
    #     return income
    
    # def getCashFlow(self, symbols):
    #     stock_batch = Stock(symbols,token=self.api_key, output_format='pandas')
    #     cash_flow = stock_batch.get_cash_flow()
    #     return cash_flow