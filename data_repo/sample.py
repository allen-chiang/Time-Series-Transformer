import stock_repo as sd
from iexfinance.stocks import Stock
token = 'pk_5e64bc4416674c809c9c9cdb2f868cfd'
client = sd.stock_repo(api_key = token)

# Functions accept array of symbols
info = client.getCompanyInfo(symbols=['amzn','aapl'])
print(info)

df = client.getHistoricalByRange('aapl',start_date='20200102',end_date='20200401')
print(df)
