# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 23:01:37 2017

@author: zhi
"""



import os
import pandas as pd
import numpy  as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from Evaluator import *


work_dir = r'C:\Users\Zhi\Documents\GitHub\trading'
data_file = r'C:\work\trading\factor_ZZ.xlsx'

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


D = data.dropna().drop(['MARKET'], axis=1)
D.set_index('Date', inplace=True)

D.columns = ['feature'+str(c) if type(c) is int else 'security' if c=='market' else 'return' if c=='returns' else c for c in D.columns ]
             
             
             
up_thr_in = 0.001
up_thr_out = 0.000
down_thr_in = -up_thr_in
down_thr_out = -up_thr_out
n_period_fwd = 2
train_win_size = 250
train_freq = 22
n_sel_feature = 10            
            
evaluator = Evaluator( 'AdaBoost' )
evaluator.set_params(up_thr_in, up_thr_out, down_thr_in, down_thr_out, n_period_fwd, train_win_size, train_freq, n_sel_feature)
evaluator.set_data(data=D)
summary = evaluator.predict()

             