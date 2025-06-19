    # -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 01:57:21 2024

@author: HB

V1 Initial.
V1.1 Added EV/S Drawdown This avoid infinite error if margins go negative.
v1.2 Fixed YF Update
v1.3 Rate Limit Workaround

"""

import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.ticker import PercentFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.dates as mdates
import numpy as np
from scipy.optimize import minimize
from scipy.signal import argrelextrema
from statsmodels.regression.linear_model import OLS
from datetime import datetime, timedelta
import time
import datetime as dt
import os
from Edgar_Company_Facts import get_cik, get_financial_data
from Fred_API import fred_get_series
import yfinance as yf

#Inputs --------------------------------------------------------------------------
"""Inputs"""
start = dt.datetime(1985,1,1)
end = dt.datetime.now()
stock ='ROKU'
#----------------------------------------------------------------------------------

#auto to prevent botting
headers = {'User-Agent': "karl.maple.beans@gmail.com"}

#Helps get around Rate Limit Requests
from curl_cffi import requests
session = requests.Session(impersonate="chrome")

def convert_df_group(df,yaxis_text = 'Period Over Period Change',columns=0):
    list_groups = []
    for i in range(len(df.columns)+columns):
        df_group = pd.DataFrame({'Date': df.index, yaxis_text: df.iloc[:,i],'Group': df.columns[i]})
        list_groups.append(df_group)
    df_groups = pd.concat(list_groups).sort_index()
    df_groups.dropna(how = 'any', inplace=True)
    return df_groups
def get_options (stock):
    top = 3                         #top x open interest strike on each side
    list_options = []
    #Get Options Prices & Generate Graphs
    tk = yf.Ticker(stock)
    # Get Expiration dates
    exps = tk.options
    # Get options for each expiration
    list_wavg_oi = []
    list_wavg_vol = []
    list_top_oi = []
    list_top_iv = []
    list_bot_iv = []
    list_top_volume = []
    list_max_pain = []
    for e in exps:
        options = tk.option_chain(e)
        #Calculated Weighted Average Open Interest Strike
        df_call = options[0]
        df_call['Expiry'] = e
        df_call['PutCall'] = 'call'
        df_call = df_call[(df_call['ask']/df_call['strike'])>0.001] #10bps, filters out stale strikes
        df_call = df_call[(df_call['strike']/df_stock['Close'][-1])<5] #filters stalke strikes from stock splits
        df_call = df_call[(df_call['impliedVolatility'])>0.05] #10bps, filters out stale strikes
        df_call['Breakeven'] = df_call['strike'] + df_call['lastPrice']
        df_call['Weight OI'] = df_call['openInterest']/df_call['openInterest'].sum()
        df_call['Weight Vol'] = df_call['volume']/df_call['volume'].sum()
        wavg_price_oi_call = (df_call['Breakeven']*df_call['Weight OI']).sum()
        wavg_price_vol_call = (df_call['Breakeven']*df_call['Weight Vol']).sum()
        oi_call = df_call['openInterest'].sum()
        vol_call = df_call['volume'].sum()

        df_put = options[1]
        df_put['Expiry'] = e
        df_put['PutCall'] = 'put'
        df_put = df_put[(df_put['ask']/df_put['strike'])>0.001] #10bps, filters out stale strikes
        df_put = df_put[(df_put['impliedVolatility'])>0.05] #10bps, filters out stale strikes
        df_put = df_put[(df_put['strike']/df_stock['Close'][-1])<5] #filters stalke strikes from stock splits
        df_put['Breakeven'] = df_put['strike'] - df_put['lastPrice']
        df_put['Weight OI'] = df_put['openInterest']/df_put['openInterest'].sum()
        df_put['Weight Vol'] = df_put['volume']/df_put['volume'].sum()
        wavg_price_oi_put = (df_put['Breakeven']*df_put['Weight OI']).sum()
        wavg_price_vol_put = (df_put['Breakeven']*df_put['Weight Vol']).sum()
        oi_put = df_put['openInterest'].sum()
        vol_put = df_put['volume'].sum()

        #Adds all options to master list
        list_options.append(df_call)
        list_options.append(df_put)

        #
        wavg_price_oi_mat = (wavg_price_oi_call * oi_call + wavg_price_oi_put * oi_put)/(oi_call +oi_put)
        new_data = pd.DataFrame({'Expiry': e, 'WAVG Price':wavg_price_oi_mat,'WAVG Call Price': wavg_price_oi_call, 'WAVG Put Price':wavg_price_oi_put, 'OI Call':oi_call,'OI Put':oi_put}, index=[0])
        list_wavg_oi.append(new_data)
        #
        wavg_price_vol_mat = (wavg_price_vol_call * oi_call + wavg_price_vol_put * oi_put)/(oi_call +oi_put)
        new_data = pd.DataFrame({'Expiry': e, 'WAVG Price':wavg_price_vol_mat,'WAVG Call Price': wavg_price_vol_call, 'WAVG Put Price':wavg_price_vol_put, 'OI Call':oi_call,'OI Put':oi_put}, index=[0])
        list_wavg_vol.append(new_data)
        #Gets the top x number of strikes by open interest
        df_call_top_oi = df_call.sort_values(by='openInterest',ascending = False)[:top]
        df_put_top_oi = df_put.sort_values(by='openInterest',ascending = False)[:top]
        list_top_oi.append(df_call_top_oi)
        list_top_oi.append(df_put_top_oi)
        #Gets the top x number of strikes by IV, high IV = option buying some thesis whether long or short
        df_call_top_iv = df_call.sort_values(by='impliedVolatility',ascending = False)[:top]
        df_put_top_iv = df_put.sort_values(by='impliedVolatility',ascending = False)[:top]
        list_top_iv.append(df_call_top_iv)
        list_top_iv.append(df_put_top_iv)
        #Gets the bot x number of strikes by IV, low IV = options selling or bounds of the price
        df_call_bot_iv = df_call.sort_values(by='impliedVolatility',ascending = True)[:top]
        df_put_bot_iv = df_put.sort_values(by='impliedVolatility',ascending = True)[:top]
        list_bot_iv.append(df_call_bot_iv)
        list_bot_iv.append(df_put_bot_iv)
        #Gets the top x number of strikes by volume
        df_call_top_volume = df_call.sort_values(by='volume',ascending = False)[:top]
        df_put_top_volume = df_put.sort_values(by='volume',ascending = False)[:top]
        list_top_volume.append(df_call_top_volume)
        list_top_volume.append(df_put_top_volume)

        def max_pain_func(price):
            return (np.maximum(price - df_call.strike,0) + np.maximum(df_put.strike-price,0)).sum()

        max_pain = minimize(max_pain_func,df_stock['Close'][-1])
        max_pain = max_pain.x
        df_max_pain = pd.DataFrame({'Expiry':df_call.Expiry.iloc[0],'Max Pain':max_pain})
        list_max_pain.append(df_max_pain)

    df_wavg_oi =pd.concat(list_wavg_oi)
    df_wavg_oi['Expiry'] = pd.to_datetime(df_wavg_oi['Expiry'])
    df_wavg_oi.set_index('Expiry',inplace=True)
    df_wavg_oi=df_wavg_oi.dropna(axis=0,how='any')

    df_top_oi = pd.concat(list_top_oi)
    df_top_oi['Expiry'] = pd.to_datetime(df_top_oi['Expiry'])
    df_top_oi.set_index('Expiry',inplace=True)

    df_bot_iv = pd.concat(list_bot_iv)
    df_bot_iv['Expiry'] = pd.to_datetime(df_bot_iv['Expiry'])
    df_bot_iv.set_index('Expiry',inplace=True)

    df_max_pain = pd.concat(list_max_pain)
    df_max_pain['Expiry'] = pd.to_datetime(   df_max_pain['Expiry'])
    df_max_pain.set_index('Expiry',inplace=True)

    df_options=pd.concat(list_options)
    df_options['Expiry'] = pd.to_datetime(df_options['Expiry'], format='%Y-%m-%d')
    df_options['Days'] = (df_options['Expiry'] - end).dt.days
    df_options['T'] = df_options['Days']/365

    return df_wavg_oi,df_top_oi,df_bot_iv,df_max_pain,df_options
def transform_data(df):
    df['daily_pct_change'] = df.Close.pct_change(periods=1)
    df['0.25Y Return Annualized'] = (1+df.Close.pct_change(periods=63))**(4)-1
    df['0.50Y Return Annualized'] = (1+df.Close.pct_change(periods=176))**(2)-1
    df['1Y Return CAGR'] = df.Close.pct_change(periods=253)
    df['2Y Return CAGR'] = (1+df.Close.pct_change(periods=253*2))**(1/2)-1
    df['3Y Return CAGR'] = (1+df.Close.pct_change(periods=253*3))**(1/3)-1
    df['YTD_Return'] = df['Close'] / df['Close'].groupby(df.index.year).transform('first') - 1
    df['Dollar_Volume'] = df['Close'] * df['Volume']
    df['VWAP 200D'] = df['Dollar_Volume'].rolling(window=200).sum()/df['Volume'].rolling(window=200).sum()
    df['LocalMax'] = df['Close'][argrelextrema(df['Close'].values, np.greater_equal, order=5)[0]]
    df['LocalMin'] = df['Close'][argrelextrema(df['Close'].values, np.less_equal, order=5)[0]]
    df['LocalDrawdown'] = df['Close'] / df['LocalMax'].ffill() - 1
    df['LocalDrawdownMin'] = df['LocalDrawdown'][argrelextrema(df['LocalDrawdown'].values, np.less_equal, order=5)[0]]
    df['LocalDrawdownMin'] = df['LocalDrawdownMin'].bfill()
    df['rolling_max'] = df['Close'].cummax()
    df['Drawdown'] = (df['Close']-df['rolling_max']) / df['rolling_max']
    df['Volatility'] = (df['Close'].rolling(window=20).std())
    df['Intraday_Range'] = df['High'] - df['Low']
    df['Intraday_Range_Pct'] = df['High']/df['Close'] - df['Low']/df['Close']
    df['Close_intraday_range'] = (df['Close'] - df['Low'])/df['Intraday_Range']*100
    df['Close_intraday_range_smooth'] = df['Close_intraday_range'].rolling(window=5).mean()
    df['5D Return'] = df['Close']/df['Close'].shift(5)-1
    df['20D Return'] = df['Close']/df['Close'].shift(20)-1
    df['50D Return'] = df['Close']/df['Close'].shift(50)-1
    df['200D Return'] = df['Close']/df['Close'].shift(200)-1
    df['20D SMA'] = df['Close'].rolling(window=20).mean()
    df['50D SMA'] = df['Close'].rolling(window=50).mean()
    df['200D SMA'] = df['Close'].rolling(window=200).mean()
    df['Momentum 10D'] = (df['Close']-df['Close'].shift(10))/df['Close'].shift(10)
    df = df[df.index > start]
    return df

"""Get Data"""
ciklist = get_cik(stock)
list_quarterly_data = []
list_quarterly_price = []

#get a company's financial line item in a dataframe
df_financial = get_financial_data(ciklist)
#list_quarterly_data.append(df_financial[0])
#get a company's price
df_price = yf.Ticker(stock,session=session).history(start=start,end=end).tz_localize(None)
df_price.index = df_price.index.tz_localize(None)

df_price = transform_data(df_price)
df = pd.concat([df_price['Close'],df_financial[0]],axis=0,join='outer')
df=df.rename(columns={0: "Close"})
df=df.sort_index()
df=df.fillna(method='ffill') #Forward fills fundamental data for price
df=df.dropna(axis=0,subset=['Revenues']) #drops all rows where revenues is nan
df=df[~df.index.duplicated(keep='last')] #drops duplicate rows, causes an error for indexing
df['Marketcap'] = df['Close']*df['Adjusted DSO']
df['Enterprise Value'] = df['Marketcap'] + df['NetDebt']
list_quarterly_data.append(df)

#Quarterly
##Concats with 2 column indexes stock and financial data
df_quarterly= pd.concat(list_quarterly_data, axis=1,  keys = stock, names = ['Stock','Data'])
df_quarterly = df_quarterly[(df_quarterly.index > start) & (df_quarterly.index < end)]
df_quarterly.dropna(how='all',inplace=True)
##Adds CY Quarter as Rows for Barchart Later
quarterly_dates = pd.date_range(start=start, end=end, freq='Q')
df_dates = pd.DataFrame(index=quarterly_dates)
df_quarterly = pd.concat([df_quarterly,df_dates])
df_quarterly = df_quarterly[~df_quarterly.duplicated(keep='first')]
df_quarterly.sort_index(inplace=True)
df_quarterly = df_quarterly.ffill()
df_cal_quarterly = df_quarterly.groupby(pd.Grouper(freq='Q')).tail(1)
df_cal_quarterly.index = pd.to_datetime(df_cal_quarterly.index)
df_cal_ltm = df_cal_quarterly.rolling(4).sum()

##Adds industry total (Sum of companies)
df_quarterly_industry = df_cal_quarterly.groupby(level=1, axis=1).sum()
df_quarterly_industry.columns = pd.MultiIndex.from_tuples([('Industry', col) for col in df_quarterly_industry.columns])
df_stock = df_quarterly_industry.rolling(4).sum()
df_cal_quarterly = pd.concat([df_cal_quarterly,df_quarterly_industry],axis=1)
df_cal_ltm = pd.concat([df_cal_ltm,df_stock],axis=1)


#Create PDF
os.makedirs('Reports', exist_ok=True)
pdf = PdfPages(f"Reports/{stock} as of {dt.datetime.now():%Y-%m-%d}.pdf")

"""Company"""
#Sales
df_cal_quarterly_sales = df_cal_quarterly.xs('Revenues', level=1, axis=1)
df_ltm_sales = df_cal_ltm.xs('Revenues', level=1, axis=1)
df_ltm_sales_index = df_ltm_sales.dropna() / df_ltm_sales.dropna().iloc[0,:]
df_ltm_sales_yoy = df_ltm_sales_index/df_ltm_sales_index.shift(4)-1
df_ltm_sales_index.index = pd.to_datetime(df_ltm_sales_index.index)
num_years = (df_ltm_sales_index.index - df_ltm_sales_index.index[0]).days / 365.25
df_ltm_sales_cagr = df_ltm_sales_index.pow(1/num_years,axis=0) -1

df_stock_gr = pd.DataFrame(df_ltm_sales_index['Industry'])
df_stock_gr['YoY Growth'] = df_ltm_sales_yoy['Industry']
df_stock_gr['CAGR']= df_ltm_sales_cagr['Industry']
df_stock_gr = df_stock_gr.dropna(how='any',axis=0).iloc[:,1:]

#Margins
df_stock = df_cal_ltm.xs('Industry', level=0, axis=1).dropna(how='all')
df_stock_margins = df_stock.div(df_stock['Revenues'],axis=0)
df_stock_margins = df_stock_margins.loc[:,['GrossProfit','OperatingIncome','NetIncome','FCFEexSBC']]

#Returns on Capital
df_stock_bs = df_cal_quarterly.xs('Industry', level=0, axis=1).dropna(how='all') #So BS itens arent summed
df_stock_returns = pd.DataFrame()
df_stock_returns['ROA'] = df_stock['NetIncome']/df_stock_bs['TotalAssets']
df_stock_returns['ROE'] = df_stock['NetIncome']/df_stock_bs['TotalStockholderEquity']
df_stock_returns['ROCE'] = df_stock['EBIT']/(df_stock_bs['TotalAssets'] - df_stock_bs['CurrentLiabilities'])
df_stock_returns['ROIC'] = df_stock['NOPAT']/df_stock_bs['InvestedCapital']
df_stock_returns = df_stock_returns.replace([np.inf, -np.inf], [0, 0])
df_stock_returns = df_stock_returns[np.isfinite(df_stock_returns).all(axis=1)]

df_stock_expense = pd.DataFrame()
df_stock_expense['Capex to Sales'] = df_stock['Capex']/df_stock['Revenues']
df_stock_expense['SG&A to Sales'] = df_stock['SG&AExpense']/df_stock['Revenues']
df_stock_expense['R&D to Sales'] = df_stock['R&DExpense']/df_stock['Revenues']
df_stock_expense['Effective Tax Rate'] = df_stock['Tax']/df_stock['EBT']

df_stock_capital = pd.DataFrame()
df_stock_capital['NWC to Sales'] = df_stock['NetWorkingCapital']/df_stock['Revenues']
df_stock_capital['PPE to Sales'] = df_stock['PPE']/df_stock['Revenues']

#Valuation
df_quarterly_industry = df_cal_quarterly.groupby(level=1, axis=1).sum()

df_stock_val = pd.DataFrame()
df_stock_val['EV/S'] = df['Enterprise Value']/df_stock['Revenues']
df_stock_val['EV/EBITDA'] = df['Enterprise Value']/df_stock['EBITDA']
df_stock_val['P/E'] = df['Marketcap']/df_stock['NetIncome']
df_stock_val['P/FCFE'] = df['Marketcap']/df_stock['FCFEexSBC']
df_stock_val['Earn Yield'] = 1/df_stock_val['P/E']
df_stock_val['FCFE Yield'] = 1/df_stock_val['P/FCFE']
df_stock_val['COD'] = df_stock['Interest']/df_stock['NetDebt']
df_stock_val['Implied LTGR'] = (0.1 *df_stock_val['P/E'] -1)/(1+df_stock_val['P/E'])
df_stock_val['Implied COE'] = (1+0.04)/df_stock_val['P/E']+0.04 #Gr = 0.04, Earnings = 1
df_stock_val['EV/S Drawdown'] = df_stock_val['EV/S']/df_stock_val['EV/S'].cummax()-1

df_stock_val['COD'] = df_stock_val['COD'].fillna(0) #No Debt
df_stock_val = df_stock_val.replace([np.inf, -np.inf], [0, 0])
df_stock_val = df_stock_val.dropna(how='any')


#Return Decomposition
df_stock_return_decomp = pd.DataFrame()
df_stock_return_decomp['Sales Growth'] = df_stock['Revenues']/df_stock['Revenues'].shift(4)-1
df_stock_return_decomp['Margin'] = df_stock_margins['NetIncome']/df_stock_margins['NetIncome'].shift(4)-1
df_stock_return_decomp['PE'] = df_stock_val['P/E']/df_stock_val['P/E'].shift(4)-1
df_stock_return_decomp['Shares'] = -(df_stock['Adjusted DSO']/df_stock['Adjusted DSO'].shift(4)-1)
df_stock_return_decomp['Price'] = df_stock['Close']/df_stock['Close'].shift(4)-1
df_stock_return_decomp['Absolute Total'] = df_stock_return_decomp['Sales Growth'] +df_stock_return_decomp['Margin']+df_stock_return_decomp['PE']
df_stock_return_decomp = df_stock_return_decomp.dropna(how='all')
df_stock_return_decomp = df_stock_return_decomp.replace([np.nan], [0])

var_count = 4
fig = plt.figure(figsize=(8.5, 11), dpi=400)
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]
axes[0].plot(df_stock[['Revenues','GrossProfit','OperatingIncome','NetIncome','FCFEexSBC']] )
for column in df_stock[['Revenues','GrossProfit','OperatingIncome','NetIncome','FCFEexSBC']]:
    axes[0].text(df_stock[column].index[-1],df_stock[column][-1],f'{df_stock[column][-1]/1000000000:,.1f}B')
axes[0].legend(['Revenues','GrossProfit','OperatingIncome','NetIncome','FCFEexSBC'])
axes[0].set_title(stock +" Earnings")
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[1].plot(df_stock_gr)
axes[1].text(df_stock_gr['YoY Growth'].index[-1],df_stock_gr['YoY Growth'][-1],f'{df_stock_gr["YoY Growth"][-1]:+.1%}')
axes[1].text(df_stock_gr['CAGR'].index[-1],df_stock_gr['CAGR'][-1],f'{df_stock_gr["CAGR"][-1]:+.1%}')
axes[1].legend(df_stock_gr.dropna(how='any',axis=0).columns)
axes[1].set_title(stock +" LTM Revenue Growth")
axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[1].sharex(axes[0])
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[2].plot(df_stock_margins)
for column in df_stock_margins:
    axes[2].text(df_stock_margins[column].index[-1],df_stock_margins[column][-1],f'{df_stock_margins[column][-1]:.1%}')
axes[2].legend(['GrossProfit','OperatingIncome','NetIncome','FCFEexSBC'])
axes[2].set_title(stock +' LTM Margins')
axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[2].sharex(axes[0])
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[3].set_ylim(max(df_stock_returns.min().min(),-0.1), min(df_stock_returns.max().max(),0.3))
axes[3].plot(df_stock_returns)
for column in df_stock_returns:
    axes[3].text(df_stock_returns[column].index[-1],min(max(df_stock_returns[column][-1],axes[3].get_ylim()[0]),axes[3].get_ylim()[1]),f'{df_stock_returns[column][-1]:.1%}')
axes[3].legend(df_stock_returns.columns)
axes[3].set_title(stock +' Returns on Capital')
axes[3].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[3].sharex(axes[0])

plt.suptitle(stock +' Report as of '+dt.datetime.now().strftime("%Y-%m-%d"),fontweight='bold')
plt.tight_layout(h_pad=0.3)
pdf.savefig()


var_count = 4
fig = plt.figure(figsize=(8.5, 11), dpi=400)
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

df_stock_gr_dol = df_stock['Revenues'] - df_stock['Revenues'].shift(4)
axes[0].bar(df_stock_gr_dol.index, df_stock_gr_dol, width=50)
axes[0].text(df_stock_gr_dol.index[-1],df_stock_gr_dol[-1],f'{df_stock_gr_dol[-1]/1000000:,.1f}M')
axes[0].set_title(stock +' Revenue Growth')
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[1].bar(df_stock.index, df_stock['Growth Capex'], width=50)
axes[1].bar(df_stock.index, df_stock['R&DExpense'], bottom=df_stock['Growth Capex'], width=50)

if  df_stock['S&MExpense'][-1] == 0:
    sga_change = df_stock['SG&AExpense']-df_stock['SG&AExpense'].shift(1)
    inc_expense = sga_change+df_stock['R&DExpense']+df_stock['Growth Capex']
    axes[1].bar(df_stock.index, sga_change, bottom=df_stock['Growth Capex']+df_stock['R&DExpense'], width=50)
    axes[1].text(sga_change.index[-1],inc_expense[-1],f'{inc_expense[-1]/1000000:,.1f}M')
else:
    inc_expense = df_stock['S&MExpense']+df_stock['R&DExpense']+df_stock['Growth Capex']
    axes[1].bar(df_stock.index, df_stock['S&MExpense'], bottom=df_stock['Growth Capex']+df_stock['R&DExpense'], width=50)
    axes[1].text(df_stock['S&MExpense'].index[-1],df_stock['S&MExpense'][-1]+df_stock['R&DExpense'][-1]+df_stock['Growth Capex'][-1],f'{df_stock["S&MExpense"][-1]/1000000:,.1f}M')
axes[1].legend(['Growth Capex','R&DExpense','S&MExpense'])
axes[1].set_title(stock +' Growth Investments')
axes[1].sharex(axes[0])
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[2].set_ylim(df_stock_expense.iloc[:,0:3].min().min(), df_stock_expense.iloc[:,0:3].max().max())
axes[2].plot(df_stock_expense)
for column in df_stock_expense:
    axes[2].text(df_stock_expense[column].index[-1],min(max(df_stock_expense[column][-1],axes[2].get_ylim()[0]),axes[2].get_ylim()[1]),f'{df_stock_expense[column][-1]:,.1%}')
axes[2].legend(df_stock_expense.columns)
axes[2].set_title(stock +" Expenses")
axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[2].sharex(axes[0])
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[3].plot(df_stock_capital)
for column in df_stock_capital:
    axes[3].text(df_stock_capital[column].index[-1],df_stock_capital[column][-1],f'{df_stock_capital[column][-1]:,.1%}')
axes[3].legend(df_stock_capital.columns)
axes[3].sharex(axes[0])
axes[3].set_title(stock +" Capital Intensity")

plt.suptitle(stock +' Report as of '+dt.datetime.now().strftime("%Y-%m-%d"),fontweight='bold')
plt.tight_layout(h_pad=0.3)
pdf.savefig()


var_count = 4
fig = plt.figure(figsize=(8.5, 11), dpi=400)
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]
axes[0].plot(df[['Marketcap', 'Enterprise Value']] )
for column in df[['Marketcap', 'Enterprise Value']]:
    axes[0].text(df[column].index[-1],df[column][-1],f'{df[column][-1]/1000000000:,.0f}B')
axes[0].legend(['Market Capitalizaton','Enterprise Value'])
axes[0].set_title(stock +" Market Valution")
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[1].plot(df_stock_val[['EV/S','EV/EBITDA','P/E','P/FCFE']])
pe_avg = df_stock_val['P/E'].mean()
pe_std = df_stock_val['P/E'].std()
axes[1].set_ylim(min(df_stock_val['EV/S'].min(),max(pe_avg-2*pe_std,-10)),min(pe_avg+2*pe_std,50))
for column in df_stock_val[['EV/S','EV/EBITDA','P/E','P/FCFE']]:
    position = min(df_stock_val[column][-1],50)
    position = max(position,-10)
    axes[1].text(df_stock_val[column].index[-1],min(max(df_stock_val[column][-1],axes[1].get_ylim()[0]),axes[1].get_ylim()[1]),f'{df_stock_val[column][-1]:.1f}x')
axes[1].legend(['EV/S','EV/EBITDA','P/E','P/FCFE'])
axes[1].set_title(stock +" LTM Multiples")
axes[1].sharex(axes[0])
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[2].set_ylim(max(df_stock_val.min().min(),-0.05),min(df_stock_val.max().max(),0.2))
axes[2].plot(df_stock_val[['Earn Yield','FCFE Yield','COD']])
for column in df_stock_val[['Earn Yield','FCFE Yield','COD']]:
    axes[2].text(df_stock_val[column].index[-1],min(max(df_stock_val[column][-1],axes[2].get_ylim()[0]),axes[2].get_ylim()[1]),f'{df_stock_val[column][-1]:.1%}')
axes[2].legend(['Earn Yield','FCFE Yield','Cost of Debt'])
axes[2].set_title(stock +' Yields')
axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[2].sharex(axes[0])
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

coe_avg = df_stock_val['Implied COE'].mean()
coe_std = df_stock_val['Implied COE'].std()
axes[3].set_ylim(max(coe_avg-2*coe_std,-0.1),min(coe_avg+2*coe_std,0.3))
axes[3].plot(df_stock_val[['Implied LTGR','Implied COE']])
for column in df_stock_val[['Implied LTGR','Implied COE']]:
    axes[3].text(df_stock_val[column].index[-1],min(max(df_stock_val[column][-1],axes[3].get_ylim()[0]),axes[3].get_ylim()[1]),f'{df_stock_val[column][-1]:.1%}')
axes[3].plot(df_stock_returns['ROE'])
axes[3].text(df_stock_returns['ROE'].index[-1],min(max(df_stock_returns['ROE'][-1],axes[3].get_ylim()[0]),axes[3].get_ylim()[1]),f'{df_stock_returns["ROE"][-1]:.1%}')
axes[3].legend(['Implied LTGR','Implied COE','LTM COE'])
axes[3].set_title(stock +' Implied Assumptions')
axes[3].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[3].sharex(axes[0])

plt.suptitle(stock +' Report as of '+dt.datetime.now().strftime("%Y-%m-%d"),fontweight='bold')
plt.tight_layout(h_pad=0.3)
pdf.savefig()



#Price Per Share
var_count = 4
fig = plt.figure(figsize=(8.5, 11), dpi=400)
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]
axes[0].plot(df_price[['Close','50D SMA','200D SMA']] )
for column in df_price[['Close','50D SMA','200D SMA']]:
    axes[0].text(df_price[column].index[-1],df_price[column][-1],f'${df_price[column][-1]:,.2f}')
axes[0].legend(['Close','50D SMA','200D SMA'])
axes[0].set_yscale('log')
axes[0].yaxis.set_major_locator(ticker.LogLocator(base=2, numticks=20))
def fmt(x, pos):
    return '${:.2f}'.format(x)
formatter = ticker.FuncFormatter(fmt)
axes[0].yaxis.set_major_formatter(formatter)
axes[0].set_title(stock +" Price Per Share")
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


decomp_avg = df_stock_return_decomp['Absolute Total'].mean()
decomp_std = df_stock_return_decomp['Absolute Total'].std()
if decomp_avg != 0:
    axes[1].set_ylim(max(decomp_avg-3*decomp_std,-3), min(decomp_avg+3*decomp_std,3))
    #axes[1].set_ylim(df_stock_return_decomp['Sales Growth'].min(), df_stock_return_decomp['Sales Growth'].max())
axes[1].plot(df_stock_return_decomp['Price'],color='black')
axes[1].bar(df_stock_return_decomp.index, df_stock_return_decomp['Sales Growth'], bottom=df_stock_return_decomp['Shares']+df_stock_return_decomp['PE']+df_stock_return_decomp['Margin'], width=50)
axes[1].bar(df_stock_return_decomp.index, df_stock_return_decomp['Margin'], bottom=df_stock_return_decomp['Shares']+df_stock_return_decomp['PE'], width=50)
axes[1].bar(df_stock_return_decomp.index, df_stock_return_decomp['PE'], bottom=df_stock_return_decomp['Shares'], width=50)
axes[1].bar(df_stock_return_decomp.index, df_stock_return_decomp['Shares'], width=50)
axes[1].text(df_stock_return_decomp.index[-1],df_stock_return_decomp['Price'][-1],f'{df_stock_return_decomp["Price"][-1]:,.2%}')
axes[1].set_title(stock +' YoY Return Decomposition')
axes[1].legend(['Price','Sales Growth','Margin','Multiple (PE)','Shares'])
axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[1].sharex(axes[0])
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[2].plot(df_price['Drawdown'])
axes[2].set_title(stock +' Price Drawdown from Prior High')
axes[2].text(df_price['Drawdown'].index[-1],df_price['Drawdown'][-1],f'{df_price["Drawdown"][-1]:,.2%}')
axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[2].sharex(axes[0])
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

""" Can remove this if it feels to cluttered"""
axes[3].plot(df_stock_val['EV/S Drawdown'])
axes[3].set_title(stock +' EV/S Drawdown from Prior High')
axes[3].text(df_stock_val['EV/S Drawdown'].index[-1],df_stock_val['EV/S Drawdown'][-1],f'{df_stock_val["EV/S Drawdown"][-1]:,.2%}')
axes[3].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[3].sharex(axes[0])
#axes[3].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.suptitle(stock +' Report as of '+dt.datetime.now().strftime("%Y-%m-%d"),fontweight='bold')
plt.tight_layout(h_pad=0.3)
pdf.savefig()


#Options
try:
    df_wavg_oi,df_top_oi,df_bot_iv,df_max_pain,df_options = get_options(stock)

    var_count = 3
    fig = plt.figure(figsize=(8.5, 11))
    gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
    axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

    axes[0].plot(df_max_pain['Max Pain'],color='darkorange',label='Max Pain')
    axes[0].plot(df_wavg_oi['WAVG Price'],color='black',label='WAVG OI Price')
    axes[0].plot(df_wavg_oi['WAVG Call Price'],color='Green',label='WAVG OI Call Price')
    axes[0].plot(df_wavg_oi['WAVG Put Price'],color='Red',label='WAVG OI Put Price')
    axes[0].legend()
    axes[0].set_title(stock+' Options Forecast')
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    axes[1].scatter(df_top_oi[df_top_oi['PutCall']=='call'].index, df_top_oi[df_top_oi['PutCall']=='call']['Breakeven'],color='green')
    axes[1].scatter(df_top_oi[df_top_oi['PutCall']=='put'].index, df_top_oi[df_top_oi['PutCall']=='put']['Breakeven'],color='red')
    axes[1].set_title(stock+' Options Highest Open Interest')
    axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    axes[2].scatter(df_bot_iv[df_bot_iv['PutCall']=='call'].index, df_bot_iv[df_bot_iv['PutCall']=='call']['Breakeven'],color='green')
    axes[2].scatter(df_bot_iv[df_bot_iv['PutCall']=='put'].index, df_bot_iv[df_bot_iv['PutCall']=='put']['Breakeven'],color='red')
    axes[2].set_title(stock+' Options Lowest Implied Volatility')
    #axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    plt.tight_layout(h_pad=0.5)
    pdf.savefig()
    
    plt.close()
except:
    print('No Options Data for this stock')

pdf.close()
