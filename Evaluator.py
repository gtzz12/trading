# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 16:08:52 2017

@author: zhi
"""

import sklearn as sl
import pandas  as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier


class Evaluator:
    def __init__(self, classifier_type):
        self.name = None
        self.data = None
        self.classifier_type = classifier_type
    
        
    
    def set_params(self, up_thr_in, up_thr_out, down_thr_in, down_thr_out, n_period_fwd, train_win_size, train_freq, n_sel_feature):
        self.up_thr_in = up_thr_in
        self.up_thr_out = up_thr_out
        self.down_thr_in = down_thr_in
        self.down_thr_out = down_thr_out
        
        self.n_period_fwd = n_period_fwd
        self.train_win_size = train_win_size
        self.train_freq   = train_freq
        self.n_sel_feature = n_sel_feature

        
    def set_data(self, data ):
        data['return_fwd'] = data.groupby('security')['return'].shift(-self.n_period_fwd)
        data['signal_up_in']  = data.return_fwd > self.up_thr_in
        data['signal_up_out']  = data.return_fwd > self.up_thr_out
        data['signal_down_in']  = data.return_fwd < self.down_thr_in
        data['signal_down_out']  = data.return_fwd < self.down_thr_out
        self.cols_feature = [c for c in data.columns if c.startswith('feature')]
        self.data = data                    
         

    ## ----- Train classifier ------ 
    ## data: last column: returns
    ##       1 to n-1 column: factors
    
    ## data: column 1: Date
    ##       column 2: security
    ##       column 3: return 
    ##       column 4 - end: features
    
    
    def train_classifier(self, X, y):
        if self.classifier_type == 'AdaBoost':
            clf = AdaBoostClassifier(n_estimators = self.n_sel_feature)
            return clf.fit(X, y)
    
    def predict_security(self, data):
        
        for t in range(self.train_win_size+self.n_period_fwd, len(data)):
            
            if ( (t - self.train_win_size - self.n_period_fwd) % self.train_freq == 0) :
                train_data   = data.iloc[(t - self.n_period_fwd - self.train_win_size ):( t - self.n_period_fwd )]
                clf_up_in    = self.train_classifier(train_data.loc[:, self.cols_feature], train_data.signal_up_in)
                clf_up_out   = self.train_classifier(train_data.loc[:, self.cols_feature], train_data.signal_up_out)
                clf_down_in  = self.train_classifier(train_data.loc[:, self.cols_feature], train_data.signal_down_in)
                clf_down_out = self.train_classifier(train_data.loc[:, self.cols_feature], train_data.signal_down_out)
            
                t_range = range(t, np.minimum(len(data), t + self.train_freq) )
                data.ix[t_range, 'pred_up_in']    = clf_up_in.predict( data.ix[t_range, self.cols_feature] ) 
                data.ix[t_range, 'pred_up_out']    = clf_up_out.predict( data.ix[t_range, self.cols_feature] ) 
                data.ix[t_range, 'pred_down_in']   = clf_down_in.predict( data.ix[t_range, self.cols_feature] ) 
                data.ix[t_range, 'pred_down_out']  = clf_down_out.predict( data.ix[t_range, self.cols_feature] ) 
                print(data.index[t])

            
        for t in range(self.train_win_size+self.n_period_fwd, len(data)):
            if data.pred_up_in[t]:
                data.ix[t, 'pred_position'] = 1
            elif data.pred_down_in[t]:
                data.ix[t, 'pred_position'] = -1
            elif data.pred_position[t-1] == 1 and not data.pred_down_out[t]:
                data.ix[t, 'pred_position'] = 1
            elif data.pred_position[t-1] == -1 and not data.pred_up_out[t]:
                data.ix[t, 'pred_position'] = -1
            else:
                data.ix[t, 'pred_position'] = 0
            if (t % 100 == 0):
                print(data.index[t])

    def predict_all(self):
        
        securities = np.unique( self.data.security )
                
        start_dt = 
        for s in securities:
            Ds  = self.data[self.data.security == s]
            Ds.sort_index(inplace=True)
            for i in range(n_train_sample+2, Dm.shape[0]):
                if （i-n_train_sample-2)%2 == 0:
                    # train
                    D_train = Dm[(i-n_train_sample-2):(i-2)]
                    clf_buy = AdaBoostClassifier(n_estimators =10)
                    clf_buy = clf_buy.fit(D_train.iloc[:, 0:76], D_train.signal)
                    clf_sell = AdaBoostClassifier(n_estimators=10()
                    clf_sell = clf_sell.fit(D_train.iloc[:, 0:76], D_train.signal2)   
            y_predict = clf_biy                     
                
            
                
                
        all_dates = self.data.index.unique()
        all_dates.values.sort()
        start_dt = self.data.index
        start = np.where(all_dates >= start_dt)[0][0]
        
        
        
        clf = AdaBoostClassifier(n_estimators=20)
        scores = cross_val_score(clf, D.iloc[:,0:76], D.signal)
        
    
            
                        
        
        
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
            
        
        

    def predict(self):
        
        securities = np.unique(self.data.security)
        self.data['pred_up_in'] = np.nan
        self.data['pred_up_out'] = np.nan
        self.data['pred_down_in'] = np.nan
        self.data['pred_down_out'] = np.nan
        self.data['pred_position'] = np.nan
        
        for security in securities:
            idx = self.data.security == security
            Ds  = self.data[idx]
            self.predict_security(Ds)
            self.data.ix[idx, 'pred_position'] = Ds.pred_position
        
        self.data['port_return'] = self.data.pred_position * self.data.return_fwd
        self.data['year'] = self.data.index.year
        summary1 = self.data.groupby(['security', 'year'])['port_return'].sum()
        summary2 = self.data.groupby('year')['port_return'].sum() 
        
        print(summary1)
        print(summary2)
        
        return ([summary1, summary2])

        
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        