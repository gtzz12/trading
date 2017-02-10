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
        
        start_dt   = self.data.index.min() + pd.Timedelta(days=365*2)
                
        all_dates = self.data.index.unique()
        all_dates.values.sort()
        start = np.where(all_dates >= start_dt)[0][0]
        
        for i in range(start, len(all_dates)):
               
            # train_data
            if (i == start) or (all_dates[i].day < all_dates[i-1].day):  # firt date or first day of month     
                train_end_dt = all_dates[i-self.n_period_fwd]
                train_start_dt = all_dates[i-self.train_win_size]
                
                idx_train = (self.data.index >= train_start_dt) & (self.data.index <= train_end_dt)
                train_data = self.data.loc[idx_train]

                clf_up_in    = self.train_classifier(train_data.loc[:, self.cols_feature], train_data.signal_up_in)
                clf_up_out   = self.train_classifier(train_data.loc[:, self.cols_feature], train_data.signal_up_out)
                clf_down_in  = self.train_classifier(train_data.loc[:, self.cols_feature], train_data.signal_down_in)
                clf_down_out = self.train_classifier(train_data.loc[:, self.cols_feature], train_data.signal_down_out)
            
            idx_predict = self.data.index == all_dates[i]
            self.data.ix[idx_predict, 'pred_up_in']    = clf_up_in.predict( self.data.ix[idx_predict, self.cols_feature] ) 
            self.data.ix[idx_predict, 'pred_up_out']   = clf_up_out.predict( self.data.ix[idx_predict, self.cols_feature] ) 
            self.data.ix[idx_predict, 'pred_down_in']  = clf_down_in.predict( self.data.ix[idx_predict, self.cols_feature] ) 
            self.data.ix[idx_predict, 'pred_down_out'] = clf_down_out.predict( self.data.ix[idx_predict, self.cols_feature] ) 
            print(all_dates[i])
        
                
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

        
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        