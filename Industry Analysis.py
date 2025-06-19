# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 02:34:44 2024

@author: HB

Last Updated: 2024-10-19

"""

import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, FuncFormatter
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
import time
import datetime as dt
import os
from Edgar_Company_Facts import get_cik, get_financial_data
from Fred_API import fred_get_series
import yfinance as yf

#Inputs --------------------------------------------------------------------------
start = dt.datetime(1985,1,1)
end = dt.datetime.now()
industry_name = 'AUTO'
stocks =['JELD', 'OC']
#----------------------------------------------------------------------------------


#auto to prevent botting
headers = {'User-Agent': "karl.maple.beans@gmail.com"}

#Helps get around Rate Limit Requests
from curl_cffi import requests
session = requests.Session(impersonate="chrome")

def convert_df_group(df,yaxis_text = 'Period Over Period Change',columns=0):
    list_groups=[]
    for i in range(len(df.columns)+columns):
        df_group = pd.DataFrame({'Date': df.index,
                       yaxis_text: df.iloc[:,i],
                       'Group': df.columns[i]})
        list_groups.append(df_group)
    df_groups = pd.concat(list_groups).sort_index()
    df_groups.dropna(how = 'any', inplace=True)
    return df_groups
def get_yf_data (stock,start,end):
    df = yf.Ticker(stock,session=session).history(start=start,end=end).tz_localize(None)
    df = df.reindex(pd.date_range(start, end, freq='D')).dropna(axis=0, how = 'any')
    return df

"""Get Data"""
ciklist = [get_cik(i) for i in stocks]
list_quarterly_data = []
list_quarterly_price = []
for stock in range(0,len(stocks)):
    #get a company's financial line item in a dataframe
    df_financial = get_financial_data(ciklist[stock])
    #list_quarterly_data.append(df_financial[0])
    #get a company's price
    df_price = get_yf_data(stocks[stock], start, end)
    df = pd.concat([df_price['Close'],df_financial[0]],axis=0,join='outer')
    df=df.rename(columns={0: "Close"})
    df=df.sort_index()
    df=df.fillna(method='ffill') #Forward fills fundamental data for price
    df=df.dropna(axis=0,subset=['Revenues']) #drops all rows where revenues is nan
    df=df[~df.index.duplicated(keep='last')] #drops duplicate rows, causes an error for indexing
    df['Marketcap'] = df['Close']*df['Adjusted DSO']
    df['Enterprise Value'] = df['Marketcap'] + df['NetDebt']
    list_quarterly_data.append(df)
    time.sleep(0.2)     
    df_financial = get_financial_data(ciklist[0])

#Quarterly
##Concats with 2 column indexes stock and financial data
df_quarterly= pd.concat(list_quarterly_data, axis=1,  keys = stocks, names = ['Stock','Data'])
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
df_ltm_industry = df_quarterly_industry.rolling(4).sum()
df_cal_quarterly = pd.concat([df_cal_quarterly,df_quarterly_industry],axis=1)
df_cal_ltm = pd.concat([df_cal_ltm,df_ltm_industry],axis=1)


#Create PDF
os.makedirs('Reports', exist_ok=True)
pdf = PdfPages(f"Reports/{industry_name} Industry as of {dt.datetime.now():%Y-%m-%d}.pdf")


"""Industry"""
#Sales
df_cal_quarterly_sales = df_cal_quarterly.xs('Revenues', level=1, axis=1)
df_ltm_sales = df_cal_ltm.xs('Revenues', level=1, axis=1)
df_ltm_sales_index = df_ltm_sales.dropna() / df_ltm_sales.dropna().iloc[0,:]
df_ltm_sales_yoy = df_ltm_sales_index/df_ltm_sales_index.shift(4)-1
df_ltm_sales_index.index = pd.to_datetime(df_ltm_sales_index.index)
num_years = (df_ltm_sales_index.index - df_ltm_sales_index.index[0]).days / 365.25
df_ltm_sales_cagr = df_ltm_sales_index.pow(1/num_years,axis=0) -1

df_ltm_industry_gr = pd.DataFrame(df_ltm_sales_index['Industry'])
df_ltm_industry_gr['YoY Growth'] = df_ltm_sales_yoy['Industry']
df_ltm_industry_gr['CAGR']= df_ltm_sales_cagr['Industry']
df_ltm_industry_gr['Median Company Growth'] = df_ltm_sales_yoy.iloc[:,:-1].median(axis=1)
df_ltm_industry_gr['Average Company Growth'] = df_ltm_sales_yoy.iloc[:,:-1].mean(axis=1)
df_ltm_industry_gr = df_ltm_industry_gr.dropna(how='any',axis=0).iloc[:,1:]

#Margins
df_ltm_industry = df_cal_ltm.xs('Industry', level=0, axis=1).dropna(how='all')
df_ltm_industry_margins = df_ltm_industry.div(df_ltm_industry['Revenues'],axis=0)
df_ltm_industry_margins = df_ltm_industry_margins.loc[:,['GrossProfit','OperatingIncome','NetIncome','FCFEexSBC']]

#Returns on Capital
df_ltm_industry_returns = pd.DataFrame()
df_ltm_industry_returns['ROA'] = df_ltm_industry['NetIncome']/df_ltm_industry['TotalAssets']
df_ltm_industry_returns['ROE'] = df_ltm_industry['NetIncome']/df_ltm_industry['TotalStockholderEquity']
df_ltm_industry_returns['ROCE'] = df_ltm_industry['EBIT']/(df_ltm_industry['TotalAssets'] - df_ltm_industry['CurrentLiabilities'])
df_ltm_industry_returns['ROIC'] = df_ltm_industry['NOPAT']/df_ltm_industry['InvestedCapital']
df_ltm_industry_returns = df_ltm_industry_returns[np.isfinite(df_ltm_industry_returns).all(axis=1)]

df_ltm_industry_expense = pd.DataFrame()
df_ltm_industry_expense['Capex to Sales'] = df_ltm_industry['Capex']/df_ltm_industry['Revenues']
df_ltm_industry_expense['SG&A to Sales'] = df_ltm_industry['SG&AExpense']/df_ltm_industry['Revenues']
df_ltm_industry_expense['R&D to Sales'] = df_ltm_industry['R&DExpense']/df_ltm_industry['Revenues']
df_ltm_industry_expense['Effective Tax Rate'] = df_ltm_industry['Tax']/df_ltm_industry['EBT']

df_ltm_industry_capital = pd.DataFrame()
df_ltm_industry_capital['NWC to Sales'] = df_ltm_industry['NetWorkingCapital']/df_ltm_industry['Revenues']
df_ltm_industry_capital['PPE to Sales'] = df_ltm_industry['PPE']/df_ltm_industry['Revenues']

#Valuation
df_quarterly_industry = df_cal_quarterly.groupby(level=1, axis=1).sum()
df_ltm_industry_val = pd.DataFrame()
df_ltm_industry_val['EV/S'] = df_quarterly_industry['Enterprise Value']/df_ltm_industry['Revenues']
df_ltm_industry_val['EV/EBITDA'] = df_quarterly_industry['Enterprise Value']/df_ltm_industry['EBITDA']
df_ltm_industry_val['P/E'] = df_quarterly_industry['Marketcap']/df_ltm_industry['NetIncome']
df_ltm_industry_val['P/FCFE'] = df_quarterly_industry['Marketcap']/df_ltm_industry['FCFEexSBC']
df_ltm_industry_val['Earn Yield'] = 1/df_ltm_industry_val['P/E']
df_ltm_industry_val['FCFE Yield'] = 1/df_ltm_industry_val['P/FCFE']
df_ltm_industry_val['COD'] = df_ltm_industry['Interest']/df_ltm_industry['NetDebt']
df_ltm_industry_val['Implied LTGR'] = (0.1 *df_ltm_industry_val['P/E'] -1)/(1+df_ltm_industry_val['P/E'])
df_ltm_industry_val['Implied COE'] = (1+0.04)/df_ltm_industry_val['P/E']+0.04 #Gr = 0.04, Earnings = 1

var_count = 4
fig = plt.figure(figsize=(8.5, 11), dpi=400)
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]
axes[0].plot(df_ltm_industry[['Revenues','GrossProfit','OperatingIncome','NetIncome','FCFEexSBC']] )
for column in df_ltm_industry[['Revenues','GrossProfit','OperatingIncome','NetIncome','FCFEexSBC']]:
    axes[0].text(df_ltm_industry[column].index[-1],df_ltm_industry[column][-1],f'{df_ltm_industry[column][-1]/1000000000:,.1f}B')
axes[0].legend(['Revenues','GrossProfit','OperatingIncome','NetIncome','FCFEexSBC'])
axes[0].set_title(industry_name+" Industry Earnings")
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[1].plot(df_ltm_industry_gr)
axes[1].text(df_ltm_industry_gr['YoY Growth'].index[-1],df_ltm_industry_gr['YoY Growth'][-1],f'{df_ltm_industry_gr["YoY Growth"][-1]:+.1%}')
axes[1].text(df_ltm_industry_gr['CAGR'].index[-1],df_ltm_industry_gr['CAGR'][-1],f'{df_ltm_industry_gr["CAGR"][-1]:+.1%}')
axes[1].legend(df_ltm_industry_gr.dropna(how='any',axis=0).columns)
axes[1].set_title(industry_name+" Industry LTM Revenue Growth")
axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[1].sharex(axes[0])
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[2].plot(df_ltm_industry_margins)
for column in df_ltm_industry_margins:
    axes[2].text(df_ltm_industry_margins[column].index[-1],df_ltm_industry_margins[column][-1],f'{df_ltm_industry_margins[column][-1]:.1%}')
axes[2].legend(['GrossProfit','OperatingIncome','NetIncome','FCFEexSBC'])
axes[2].set_title(industry_name+' Industry LTM Margins')
axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[2].sharex(axes[0])
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[3].plot(df_ltm_industry_returns)
for column in df_ltm_industry_returns:
    axes[3].text(df_ltm_industry_returns[column].index[-1],df_ltm_industry_returns[column][-1],f'{df_ltm_industry_returns[column][-1]:.1%}')
axes[3].legend(df_ltm_industry_returns.columns)
axes[3].set_title(industry_name+' Returns on Capital')
axes[3].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[3].sharex(axes[0])

plt.suptitle(industry_name+' Report as of '+dt.datetime.now().strftime("%Y-%m-%d"),fontweight='bold')
plt.tight_layout(h_pad=0.3)
pdf.savefig() 

var_count = 4
fig = plt.figure(figsize=(8.5, 11), dpi=400)
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]

df_ltm_industry_gr_dol = df_ltm_industry['Revenues'] - df_ltm_industry['Revenues'].shift(4)
axes[0].bar(df_ltm_industry_gr_dol.index, df_ltm_industry_gr_dol, width=50)
axes[0].text(df_ltm_industry_gr_dol.index[-1],df_ltm_industry_gr_dol[-1],f'{df_ltm_industry_gr_dol[-1]/1000000:,.1f}M')
axes[0].set_title(industry_name+' Revenue Growth')
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[1].bar(df_ltm_industry.index, df_ltm_industry['Growth Capex'], width=50)
axes[1].bar(df_ltm_industry.index, df_ltm_industry['R&DExpense'], bottom=df_ltm_industry['Growth Capex'], width=50)
if  df_ltm_industry['S&MExpense'][-1] == 0:
    sga_change = df_ltm_industry['SG&AExpense']-df_ltm_industry['SG&AExpense'].shift(1)
    inc_expense = sga_change+ df_ltm_industry['R&DExpense']+ df_ltm_industry['Growth Capex']
    axes[1].bar( df_ltm_industry.index, sga_change, bottom= df_ltm_industry['Growth Capex']+ df_ltm_industry['R&DExpense'], width=50)
    axes[1].text(sga_change.index[-1],inc_expense[-1],f'{inc_expense[-1]/1000000:,.1f}M')
else:
    inc_expense =  df_ltm_industry['S&MExpense']+ df_ltm_industry['R&DExpense']+ df_ltm_industry['Growth Capex']
    axes[1].bar( df_ltm_industry.index,  df_ltm_industry['S&MExpense'], bottom= df_ltm_industry['Growth Capex']+ df_ltm_industry['R&DExpense'], width=50)
    axes[1].text( df_ltm_industry['S&MExpense'].index[-1], df_ltm_industry['S&MExpense'][-1]+ df_ltm_industry['R&DExpense'][-1]+ df_ltm_industry['Growth Capex'][-1],f'{ df_ltm_industry["S&MExpense"][-1]/1000000:,.1f}M')
axes[1].bar(df_ltm_industry.index, df_ltm_industry['S&MExpense'], bottom=df_ltm_industry['Growth Capex']+df_ltm_industry['R&DExpense'], width=50)
#axes[1].text(df_ltm_industry['Growth Capex'].index[-1],df_ltm_industry['Growth Capex'][-1],f'{df_ltm_industry["Growth Capex"][-1]/1000000:,.1f}M')
#axes[1].text(df_ltm_industry['R&DExpense'].index[-1],df_ltm_industry['Growth Capex'][-1]+df_ltm_industry['R&DExpense'][-1],f'{df_ltm_industry["R&DExpense"][-1]/1000000:,.1f}M')
axes[1].text(df_ltm_industry['S&MExpense'].index[-1],inc_expense[-1],f'{inc_expense[-1]/1000000:,.1f}M')
axes[1].legend(['Growth Capex','R&DExpense','S&MExpense'])
axes[1].set_title(industry_name+' Growth Investments')
axes[1].sharex(axes[0])
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[2].set_ylim(df_ltm_industry_expense.iloc[:,0:3].min().min(), df_ltm_industry_expense.iloc[:,0:3].max().max())
axes[2].plot(df_ltm_industry_expense)
for column in df_ltm_industry_expense:
    axes[2].text(df_ltm_industry_expense[column].index[-1],min(max(df_ltm_industry_expense[column][-1],axes[2].get_ylim()[0]),axes[2].get_ylim()[1]),f'{df_ltm_industry_expense[column][-1]:,.1%}')
axes[2].legend(df_ltm_industry_expense.columns)
axes[2].set_title(industry_name+" Expenses")
axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[2].sharex(axes[0])
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[3].plot(df_ltm_industry_capital)
for column in df_ltm_industry_capital:
    axes[3].text(df_ltm_industry_capital[column].index[-1],df_ltm_industry_capital[column][-1],f'{df_ltm_industry_capital[column][-1]:,.1%}')
axes[3].legend(df_ltm_industry_capital.columns)
axes[3].sharex(axes[0])
axes[3].set_title(industry_name+" Capital Intensity")

plt.suptitle(industry_name+' Report as of '+dt.datetime.now().strftime("%Y-%m-%d"),fontweight='bold')
plt.tight_layout(h_pad=0.3)
pdf.savefig() 


var_count = 4
fig = plt.figure(figsize=(8.5, 11), dpi=400)
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]
axes[0].plot(df_ltm_industry[['Marketcap', 'Enterprise Value']] )
for column in df_ltm_industry[['Marketcap', 'Enterprise Value']]:
    axes[0].text(df_ltm_industry[column].index[-1],df_ltm_industry[column][-1],f'{df_ltm_industry[column][-1]/1000000000:,.0f}B')
axes[0].legend(['Market Capitalizaton','Enterprise Value'])
axes[0].set_title(industry_name+" Market Valution")
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[1].plot(df_ltm_industry_val[['EV/S','EV/EBITDA','P/E','P/FCFE']])
pe_avg = df_ltm_industry_val['P/E'].mean()
pe_std = df_ltm_industry_val['P/E'].std()
axes[1].set_ylim(min(df_ltm_industry_val['EV/S'].min(),max(pe_avg-2*pe_std,-10)),min(pe_avg+2*pe_std,50))
for column in df_ltm_industry_val[['EV/S','EV/EBITDA','P/E','P/FCFE']]:
    position = min(df_ltm_industry_val[column][-1],50)
    position = max(position,-10)
    axes[1].text(df_ltm_industry_val[column].index[-1],min(max(df_ltm_industry_val[column][-1],axes[1].get_ylim()[0]),axes[1].get_ylim()[1]),f'{df_ltm_industry_val[column][-1]:.1f}x')
axes[1].legend(['EV/S','EV/EBITDA','P/E','P/FCFE'])
axes[1].set_title(industry_name+" Industry LTM Multiples")
axes[1].sharex(axes[0])
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[2].plot(df_ltm_industry_val[['Earn Yield','FCFE Yield','COD']])
for column in df_ltm_industry_val[['Earn Yield','FCFE Yield','COD']]:
    axes[2].text(df_ltm_industry_val[column].index[-1],df_ltm_industry_val[column][-1],f'{df_ltm_industry_val[column][-1]:.1%}')
axes[2].legend(['Earn Yield','FCFE Yield','COD'])
axes[2].set_title(industry_name+' Industry Yields')
axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[2].sharex(axes[0])
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[3].plot(df_ltm_industry_val[['Implied LTGR','Implied COE']])
for column in df_ltm_industry_val[['Implied LTGR','Implied COE']]:
    axes[3].text(df_ltm_industry_val[column].index[-1],df_ltm_industry_val[column][-1],f'{df_ltm_industry_val[column][-1]:.1%}')
axes[3].legend(['Implied LTGR','Implied COE'])
axes[3].set_title(industry_name+' Industry Implied Assumptions')
axes[3].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[3].sharex(axes[0])

plt.suptitle(industry_name+' Report as of '+dt.datetime.now().strftime("%Y-%m-%d"),fontweight='bold')
plt.tight_layout(h_pad=0.3)
pdf.savefig() 


"""Companies"""
#Revenues & Market Share
df_marketshare = df_cal_quarterly_sales.iloc[:,0:len(stocks)] #Excludes Industry (Stock only)
df_marketshare = df_marketshare.clip(lower=0) #Brute force correction changes negative numbers to zero.
df_marketshare = df_marketshare.div(df_marketshare.sum(axis=1), axis=0)

var_count = 3
fig = plt.figure(figsize=(8.5, 11), dpi=400)
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]
axes[0].plot(df_ltm_sales)
for column in df_ltm_sales:
    axes[0].text(df_ltm_sales[column].index[-1],df_ltm_sales[column][-1],f'{df_ltm_sales[column][-1]/100000000:,.0f}M')
axes[0].legend(df_ltm_sales.columns)
axes[0].set_title(industry_name+" Industry LTM Revenue")
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)


salescagr_avg = df_ltm_sales_cagr['Industry'].mean()
salescagr_std = df_ltm_sales_cagr['Industry'].std()
axes[1].set_ylim(salescagr_avg-3*salescagr_std,salescagr_avg+3*salescagr_std)
axes[1].plot(df_ltm_sales_cagr)
for column in df_ltm_sales_cagr:
    axes[1].text(df_ltm_sales_cagr[column].index[-1],min(max(df_ltm_sales_cagr[column][-1],axes[1].get_ylim()[0]),axes[1].get_ylim()[1]),f'{df_ltm_sales_cagr[column][-1]:+.1%}')
axes[1].legend(df_ltm_sales_cagr.columns)
axes[1].set_title(industry_name+" Industry LTM Revenue CAGR")
axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[1].sharex(axes[0])
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

df_marketshare.plot(kind='area', stacked=True, ax=axes[2], legend=True)
axes[2].set_title(industry_name+" Industry LTM Market Share")
axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[2].sharex(axes[0])

plt.suptitle('Competitive Analysis',fontweight='bold')
plt.tight_layout(h_pad=0.3)
pdf.savefig() 


#Margins by Companies
df_ltm_gm = df_cal_ltm.xs('GrossProfit', level=1, axis=1) / df_cal_ltm.xs('Revenues', level=1, axis=1)
df_ltm_om = df_cal_ltm.xs('OperatingIncome', level=1, axis=1) / df_cal_ltm.xs('Revenues', level=1, axis=1)
df_ltm_nm = df_cal_ltm.xs('NetIncome', level=1, axis=1) / df_cal_ltm.xs('Revenues', level=1, axis=1)
df_ltm_fcf = df_cal_ltm.xs('FCFEexSBC', level=1, axis=1) / df_cal_ltm.xs('Revenues', level=1, axis=1)

var_count = 4
fig = plt.figure(figsize=(8.5, 11), dpi=400)
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]
axes[0].plot(df_ltm_gm)
for column in df_ltm_gm:
    axes[0].text(df_ltm_gm[column].index[-1],df_ltm_gm[column][-1],f'{df_ltm_gm[column][-1]:+.1%}')
axes[0].legend(df_ltm_sales.columns)
axes[0].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[0].set_title(industry_name+" Industry Gross Margin")
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[1].plot(df_ltm_om)
for column in df_ltm_om:
    axes[1].text(df_ltm_om[column].index[-1],df_ltm_om[column][-1],f'{df_ltm_om[column][-1]:+.1%}')
axes[1].legend(df_ltm_sales_cagr.columns)
axes[1].set_title(industry_name+" Industry Operating Margin")
axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[1].sharex(axes[0])
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[2].plot(df_ltm_nm)
for column in df_ltm_nm:
    axes[2].text(df_ltm_nm[column].index[-1],df_ltm_nm[column][-1],f'{df_ltm_nm[column][-1]:+.1%}')
axes[2].legend(df_ltm_sales_cagr.columns)
axes[2].set_title(industry_name+" Industry Net Income Margin")
axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[2].sharex(axes[0])
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[3].plot(df_ltm_fcf)
for column in df_ltm_fcf:
    axes[3].text(df_ltm_fcf[column].index[-1],df_ltm_fcf[column][-1],f'{df_ltm_fcf[column][-1]:+.1%}')
axes[3].legend(df_ltm_sales_cagr.columns)
axes[3].set_title(industry_name+" Industry FCFE Margin")
axes[3].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[3].sharex(axes[0])

plt.suptitle('Competitive Analysis',fontweight='bold')
plt.tight_layout(h_pad=0.3)
pdf.savefig() 


#Returns on Capital
df_ltm_roa = df_cal_ltm.xs('NetIncome', level=1, axis=1) / df_cal_ltm.xs('TotalAssets', level=1, axis=1)
df_ltm_roe = df_cal_ltm.xs('NetIncome', level=1, axis=1) / df_cal_ltm.xs('TotalStockholderEquity', level=1, axis=1)
df_ltm_roce = df_cal_ltm.xs('EBIT', level=1, axis=1) / (df_cal_ltm.xs('TotalAssets', level=1, axis=1) - df_cal_ltm.xs('CurrentLiabilities', level=1, axis=1))
df_ltm_roic = df_cal_ltm.xs('NOPAT', level=1, axis=1) / df_cal_ltm.xs('InvestedCapital', level=1, axis=1)
df_ltm_nopat_yoy = df_cal_ltm.xs('NOPAT', level=1, axis=1) - df_cal_ltm.xs('NOPAT', level=1, axis=1).shift(4) 
df_ltm_investedcapital_yoy = df_cal_ltm.xs('InvestedCapital', level=1, axis=1)-df_cal_ltm.xs('InvestedCapital', level=1, axis=1).shift(4)
df_ltm_growthinvest = df_cal_ltm.xs('Growth Investments', level=1, axis=1)
df_ltm_roiic = df_ltm_nopat_yoy / df_ltm_investedcapital_yoy
df_ltm_roiicv2 = df_ltm_nopat_yoy / df_ltm_growthinvest

var_count = 4
fig = plt.figure(figsize=(8.5, 11), dpi=400)
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]
axes[0].plot(df_ltm_roa)
for column in df_ltm_roa:
    axes[0].text(df_ltm_roa[column].index[-1],df_ltm_roa[column][-1],f'{df_ltm_roa[column][-1]:+.1%}')
axes[0].legend(df_ltm_sales.columns)
axes[0].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[0].set_title(industry_name+" Industry ROA")
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[1].plot(df_ltm_roe)
for column in df_ltm_roe:
    axes[1].text(df_ltm_roe[column].index[-1],df_ltm_roe[column][-1],f'{df_ltm_roe[column][-1]:+.1%}')
axes[1].legend(df_ltm_sales_cagr.columns)
axes[1].set_title(industry_name+" Industry ROE")
axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[1].sharex(axes[0])
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[2].plot(df_ltm_roce)
for column in df_ltm_roce:
    axes[2].text(df_ltm_roce[column].index[-1],df_ltm_roce[column][-1],f'{df_ltm_roce[column][-1]:+.1%}')
axes[2].legend(df_ltm_sales_cagr.columns)
axes[2].set_title(industry_name+" Industry ROCE")
axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[2].sharex(axes[0])
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[3].plot(df_ltm_roic)
for column in df_ltm_roic:
    axes[3].text(df_ltm_roic[column].index[-1],df_ltm_roic[column][-1],f'{df_ltm_roic[column][-1]:+.1%}')
axes[3].legend(df_ltm_sales_cagr.columns)
axes[3].set_title(industry_name+" Industry ROIC")
axes[3].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[3].sharex(axes[0])

plt.suptitle('Competitive Analysis',fontweight='bold')
plt.tight_layout(h_pad=0.3)
pdf.savefig() 


### Return on Incremental Invested Capital
var_count = 4
fig = plt.figure(figsize=(8.5, 11), dpi=400)
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]
axes[0].plot(df_ltm_roiic)
for column in df_ltm_roiic:
    axes[0].text(df_ltm_roiic[column].index[-1],df_ltm_roiic[column][-1],f'{df_ltm_roiic[column][-1]:.1%}')
axes[0].legend(df_ltm_sales.columns)
axes[0].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[0].set_title(industry_name+" Industry ROIIC")
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[1].plot(df_ltm_investedcapital_yoy)
for column in df_ltm_investedcapital_yoy:
    axes[1].text(df_ltm_investedcapital_yoy[column].index[-1],df_ltm_investedcapital_yoy[column][-1],f'{df_ltm_investedcapital_yoy[column][-1]/1000000000:,.1f}B')
axes[1].legend(df_ltm_sales_cagr.columns)
axes[1].set_title(industry_name+" Change in Invested Capital")
axes[1].sharex(axes[0])
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[2].plot(df_ltm_roiicv2)
for column in df_ltm_roiicv2:
    axes[2].text(df_ltm_roiicv2[column].index[-1],df_ltm_roiicv2[column][-1],f'{df_ltm_roiicv2[column][-1]:.1%}')
axes[2].legend(df_ltm_sales_cagr.columns)
axes[2].set_title(industry_name+" Industry ROIIC (Growth)")
axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[2].sharex(axes[0])
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[3].plot(df_ltm_growthinvest)
for column in df_ltm_growthinvest:
    axes[3].text(df_ltm_growthinvest[column].index[-1],df_ltm_growthinvest[column][-1],f'{df_ltm_growthinvest[column][-1]/1000000000:,.1f}B')
axes[3].legend(df_ltm_sales_cagr.columns)
axes[3].set_title(industry_name+" Growth Investment")
axes[3].sharex(axes[0])

plt.suptitle('Competitive Analysis',fontweight='bold')
plt.tight_layout(h_pad=0.3)
pdf.savefig() 


### Valuation
df_ltm_evs = df_cal_quarterly.xs('Enterprise Value', level=1, axis=1) / df_cal_ltm.xs('Revenues', level=1, axis=1)
df_ltm_evebitda = df_cal_quarterly.xs('Enterprise Value', level=1, axis=1) / df_cal_ltm.xs('EBITDA', level=1, axis=1)
df_ltm_pe = df_cal_quarterly.xs('Marketcap', level=1, axis=1) / df_cal_ltm.xs('NetIncome', level=1, axis=1)
df_ltm_pfcfe = df_cal_quarterly.xs('Marketcap', level=1, axis=1) / df_cal_ltm.xs('FCFEexSBC', level=1, axis=1)
df_ltm_intrate = df_cal_ltm.xs('Interest', level=1, axis=1) / df_cal_ltm.xs('NetDebt', level=1, axis=1)
df_ltm_im_ltgr =  (0.1 *df_ltm_pe -1)/(1+df_ltm_pe)
df_ltm_im_coe = (1+0.04)/df_ltm_pe+0.04 #Gr = 0.04, Earnings = 1
df_ltm_im_1ygr = df_ltm_pe/((1+0.04)/(0.1-0.04)) #Gr = 0.04, Earnings = 1

var_count = 4
fig = plt.figure(figsize=(8.5, 11), dpi=400)
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]
axes[0].plot(df_ltm_evs)
for column in df_ltm_evs:
    axes[0].text(df_ltm_evs[column].index[-1],df_ltm_evs[column][-1],f'{df_ltm_evs[column][-1]:.1f}x')
axes[0].legend(df_ltm_sales.columns)
axes[0].set_title(industry_name+" LTM EV/Sales")
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

evebitda_avg = df_ltm_evebitda['Industry'].mean()
evebitda_std = df_ltm_evebitda['Industry'].std()
axes[1].set_ylim(evebitda_avg-3*evebitda_std,evebitda_avg+3*evebitda_std)
axes[1].plot(df_ltm_evebitda)
for column in df_ltm_evebitda:
    axes[1].text(df_ltm_evebitda[column].index[-1],min(max(df_ltm_evebitda[column][-1],axes[1].get_ylim()[0]),axes[1].get_ylim()[1]),f'{df_ltm_evebitda[column][-1]:.1f}x')
axes[1].legend(df_ltm_sales_cagr.columns)
axes[1].set_title(industry_name+" LTM EV/EBITDA")
axes[1].sharex(axes[0])
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

axes[2].plot(df_ltm_pe)
axes[2].set_ylim(min(max(pe_avg-3*pe_std,-5),50),min(pe_avg+3*pe_std,50))
for column in df_ltm_pe:
    axes[2].text(df_ltm_pe[column].index[-1],min(max(df_ltm_pe[column][-1],axes[2].get_ylim()[0]),axes[2].get_ylim()[1]),f'{df_ltm_pe[column][-1]:.1f}x')
axes[2].legend(df_ltm_sales_cagr.columns)
axes[2].set_title(industry_name+" LTM P/E")
axes[2].sharex(axes[0])
axes[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

fcfe_avg = df_ltm_pfcfe['Industry'].mean()
fcfe_std = df_ltm_pfcfe['Industry'].std()
axes[3].plot(df_ltm_pfcfe)
axes[3].set_ylim(max(fcfe_avg-3*fcfe_std,-5),min(fcfe_avg+3*fcfe_std,50))
for column in df_ltm_pfcfe:
    axes[3].text(df_ltm_pfcfe[column].index[-1],min(max(df_ltm_pfcfe[column][-1],axes[3].get_ylim()[0]),axes[3].get_ylim()[1]),f'{df_ltm_pfcfe[column][-1]:.1f}x')
axes[3].legend(df_ltm_sales_cagr.columns)
axes[3].set_title(industry_name+" LTM P/FCFE")
axes[3].sharex(axes[0])

plt.suptitle('Competitive Analysis',fontweight='bold')
plt.tight_layout(h_pad=0.3)
pdf.savefig() 


var_count = 3
fig = plt.figure(figsize=(8.5, 11), dpi=400)
gs = gridspec.GridSpec(var_count, 1, height_ratios=[1] + [1] * (var_count- 1))  # First subplot is larger
axes = [fig.add_subplot(gs[i]) for i in range(var_count)]
axes[0].plot(df_ltm_intrate)
for column in df_ltm_intrate:
    axes[0].text(df_ltm_intrate[column].index[-1],df_ltm_intrate[column][-1],f'{df_ltm_intrate[column][-1]:.1%}')
axes[0].legend(df_ltm_sales.columns)
axes[0].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[0].set_title(industry_name+" LTM Cost of Debt")
axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

coe_mean = df_ltm_im_coe['Industry'].mean()
coe_std = df_ltm_im_coe['Industry'].std()
axes[1].set_ylim(coe_mean-2*coe_std,coe_mean+2*coe_std)
axes[1].plot(df_ltm_im_coe)
for column in df_ltm_im_coe:
    axes[1].text(df_ltm_im_coe[column].index[-1],min(max(df_ltm_im_coe[column][-1],axes[1].get_ylim()[0]),axes[1].get_ylim()[1]),f'{df_ltm_im_coe[column][-1]:.1%}')
axes[1].legend(df_ltm_sales_cagr.columns)
axes[1].set_title(industry_name+" Implied COE")
axes[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[1].sharex(axes[0])
axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

ltgr_mean = df_ltm_im_ltgr['Industry'].mean()
ltgr_std = df_ltm_im_ltgr['Industry'].std()
axes[2].set_ylim(ltgr_mean-2*ltgr_std,ltgr_mean+2*ltgr_std)
axes[2].plot(df_ltm_im_ltgr)
for column in df_ltm_im_ltgr:
    axes[2].text(df_ltm_im_ltgr[column].index[-1],min(max(df_ltm_im_ltgr[column][-1],axes[2].get_ylim()[0]),axes[2].get_ylim()[1]),f'{df_ltm_im_ltgr[column][-1]:.1%}')
axes[2].legend(df_ltm_sales.columns)
axes[2].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
axes[2].set_title(industry_name+" Implied LTGR")

plt.suptitle('Competitive Analysis',fontweight='bold')
plt.tight_layout(h_pad=0.3)
pdf.savefig() 


pdf.close()
