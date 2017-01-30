# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 22:00:52 2017

@author: zhi
"""
import os
import pandas as pd
import numpy  as np



work_dir = r'C:\work\Trading'
data_file = 'factor_ZZ.xlsx'

os.chdir( work_dir )

features = pd.read_excel(data_file, sheetname=0)
returns  = pd.read_excel(data_file, sheetname=1)

features.rename(columns={'TRADE_DATE':'Date'},inplace=True)


R = returns.reset_index()
R.rename(columns={'index':'Date'},inplace=True)
R = pd.melt(R, id_vars=['Date'], var_name='market', value_name='returns')

data = pd.merge(features, R, left_on=['Date','MARKET'], right_on=['Date','market'] )

D = data.dropna().drop(['MARKET'], axis=1)
D.set_index('Date', inplace=True)

plot(D.loc[:,1], D.returns)

#for i in range(1, D.shape[1]):
#    corr = np.corrcoef(D.loc[:,i], D.returns)
#    print(corr[0,1])
    
D.    
    
    
