# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 22:00:52 2017

@author: zhi
"""
import os
import pandas as pd
import numpy  as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier


work_dir = r'C:\Users\Zhi\Documents\GitHub\trading'
data_file = r'I:\work\trading\factor_ZZ.xlsx'

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

D['returnT1'] = D.groupby('market')['returns'].shift(-1)
D['returnT2'] = D.groupby('market')['returns'].shift(-2)

threshold = 0.000

D['signal'] = D.returnT2 > threshold
D['signal2'] = D.returnT2 < -threshold



n_train_sample = 250 # training set size
train_freq     = 22


start_dt = '2007-1-1'



all_dates = D.index.unique()
all_dates.values.sort()
start = np.where(all_dates >= start_dt)[0][0]



clf = AdaBoostClassifier(n_estimators=20)
scores = cross_val_score(clf, D.iloc[:,0:76], D.signal)


D['sig_predict'] = np.nan
D['sig_predict2'] = np.nan



markets = np.unique(D.market)


for m in markets:
    Dm  = D[D.market == m]
    Dm.sort_index(inplace=True)
    for i in range(n_train_sample+2, Dm.shape[0]):
        if （i-n_train_sample-2)%2 == 0:
            # train
            D_train = Dm[(i-n_train_sample-2):(i-2)]
            clf_buy = AdaBoostClassifier(n_estimators =10)
            clf_buy = clf_buy.fit(D_train.iloc[:, 0:76], D_train.signal)
            clf_sell = AdaBoostClassifier(n_estimators=10()
            clf_sell = clf_sell.fit(D_train.iloc[:, 0:76], D_train.signal2)   
    y_predict = clf_biy                     
        
    
        
    
    
                


for i in range(start, len(all_dates)):
       
    # train_data
    if (i == start) or (all_dates[i].day < all_dates[i-1].day):  # firt date or first day of month     
        train_end_dt = all_dates[i-2]
        train_start_dt = all_dates[i-n_train_sample]
        
        idx_train = (D.index >= train_start_dt) & (D.index <= train_end_dt)
        
        clf_buy = AdaBoostClassifier (n_estimators =20)
        clf_buy = clf_buy.fit(D.iloc[idx_train, 0:76], D.signal[idx_train])
        
        clf_sell = AdaBoostClassifier(n_estimators =20)
        clf_sell = clf_sell.fit(D.iloc[idx_train, 0:76], D.signal2[idx_train])
        
    
    idx_predict = D.index == all_dates[i]
    y_predict   = clf_buy.predict( D.iloc[idx_predict, 0:76]  )    
    D.loc[idx_predict, 'sig_predict'] = y_predict    

    y_predict   = clf_sell.predict( D.iloc[idx_predict, 0:76]  )    
    D.loc[idx_predict, 'sig_predict2'] = y_predict 
    print(all_dates[i])
    

markets = np.unique( D.market )        
D['gain']  = D.sig_predict * D.returnT2


D['year'] = D.index.year
 

    
for m in markets:
    idx = (D.market == m) & (~D.sig_predict.isnull())
    returns = D.sig_predict[idx] * D.returnT2[idx]    
    
    
    
    
    
    
    
    
        
        
        
    
           
       
    # predi
    
        
    
    








#for i in range(1, D.shape[1]):
#    corr = np.corrcoef(D.loc[:,i], D.returns)
#    print(corr[0,1])









    
    
    
