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
        data['sigal_up_in']  = data.return_fwd > self.up_thr_in
        data['sigal_up_out']  = data.return_fwd > self.up_thr_out
        data['sigal_down_in']  = data.return_fwd < self.down_thr_in
        data['sigal_down_out']  = data.return_fwd < self.down_thr_out
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
            
            if ( (t-self.train_win_size-self.n_period_fwd) % self.train_freq == 0) :
                train_data   = data.loc[(t-self.n_period_fwd-self.train_win_size):(t-self.n_period_fwd)]
                clf_up_in    = self.train_classifier(train_data.loc[:, self.cols_feature], train_data.sig_up_in)
                clf_up_out   = self.train_classifier(train_data.loc[:, self.cols_feature], train_data.sig_up_out)
                clf_down_in  = self.train_classifier(train_data.loc[:, self.cols_feature], train_data.sig_down_in)
                clf_down_out = self.train_classifier(train_data.loc[:, self.cols_feature], train_data.sig_down_out)
            
            data.iloc[t, 'pred_up_in']    = clf_up_in.predict( data.iloc[t, self.cols_feature] ) 
            data.iloc[t, 'prd_up_out']    = clf_up_out.predict( data.iloc[t, self.cols_feature] ) 
            data.iloc[t, 'prd_down_in']   = clf_down_in.predict( data.iloc[t, self.cols_feature] ) 
            data.iloc[t, 'prd_down_out']  = clf_down_out.predict( data.iloc[t, self.cols_feature] ) 
            
        for t in range(self.train_win_size+self.n_period_fwd, len(data)):
            if data.pred_up_in[t]:
                data.iloc[t, 'pred_position'] = 1
            elif data.pred_down_in[t]:
                data.iloc[t, 'pred_position'] = -1
            elif data.pred_position[t-1] == 1 and not data.pred_down_out[t]:
                data.iloc[t, 'pred_position'] = 1
            elif data.pred_position[t-1] == -1 and not data.pred_up_out[t]:
                data.iloc[t, 'pred_position'] = -1
            else:
                data.iloc[t, 'pred_position'] = 0


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
            self.data.iloc[idx, 'pred_position'] = Ds.pred_position

        
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        