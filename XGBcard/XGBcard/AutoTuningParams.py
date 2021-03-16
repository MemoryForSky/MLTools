# -*- coding: utf-8 -*-

"""
Created at Apr. 17 2020

@author: ZhangXiaoqiang

@infos: AutoTuningParams模块，用于自动调参

==============================================================
"""

import numpy as np
import pandas as pd
from skopt import gp_minimize
from sklearn.model_selection import cross_val_score
from xgboost.sklearn import XGBClassifier
from toad.metrics import KS, F1, AUC

class AutoTuningParams(object):
    """Automatic tuning parameters.
    
    Parameters
    -----------
    X_train: array-like, shape [n_samples, n_cate_features]
        The features of training data.
    y_train: array-like, shape [n_samples,]
        The labels of Training data.
    params_fixed: dict
        The fixed parameters.
    tuning_space: dict
        The parameters to be adjusted.
    pos_weight: int or float
        Control the balance of positive and negative weights, useful for unbalanced classes. 
        A typical value to consider: sum(negative instances) / sum(positive instances).
        In addition, theparameter can used in cost-sensitive learning, for example: It is worse to class a customer as good when they are bad (5), 
        than it is to class a customer as bad when they are good (1), the parameter can set to 5.
    verbosity: int
       Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug).
    seed: int, default = 0
       Random number seed. 
    """
    def __init__(self, X_train, y_train, params_fixed={}, tuning_space={}, pos_weight=1, verbosity=1, seed=9853):
        self.X_train = X_train
        self.y_train = y_train
        self.params_fixed = params_fixed
        self.tuning_space = tuning_space
        self.pos_weight = pos_weight
        self.verbosity = verbosity
        self.seed = seed
        
    def auto_tuning(self):
        self.params = {'booster': self.params_fixed.get("booster", 'gbtree'),
                  'objective': self.params_fixed.get("objective", 'binary:logistic'),
                  'eval_metric': self.params_fixed.get("eval_metric", 'auc'),
                  'max_depth': self.params_fixed.get("max_depth", 1),
                  'learning_rate': self.params_fixed.get("learning_rate", 0.1),
                  'scale_pos_weight': self.pos_weight, 
                  'verbosity': self.verbosity,
                  'seed': self.seed,
                  }
        
        self.space = {'n_estimators': self.tuning_space.get("n_estimators", (50, 2000)),
                'min_child_weight': self.tuning_space.get("min_child_weight", (1, 30)),
                'min_samples_split': self.tuning_space.get("min_samples_split", (10, 500)),
                'min_samples_leaf': self.tuning_space.get("min_samples_leaf", (5, 100)),
                'subsample': self.tuning_space.get("subsample", (0, 1)),
                'colsample_bytree': self.tuning_space.get("colsample_bytree", (0.3, 1)),
                'reg_alpha': self.tuning_space.get("reg_alpha", (0, 20)),
                'reg_lambda': self.tuning_space.get("reg_lambda", (1, 20)),
                }
        
        self.reg = XGBClassifier(**self.params)
    
        res_gp = gp_minimize(self.objective_, self.space.values(), n_calls=50, random_state=self.seed)
        self.best_hyper_params = {k:v for k,v in zip(self.space.keys(), res_gp.x)}

        print('Best accuracy score = ', 1-res_gp.fun)
        print('Best parameters = ', self.best_hyper_params)
    
    def model_eval(self, X_test, y_test):
        """Evaluating model effects.
        
        Parameters
        -----------
        X_test: array-like
            Test set.
        y_test: array-like
            The labels of test set.
        """
        self.selected_params = self.best_hyper_params.copy()
        self.selected_params.update(self.params)
        
        # train model
        clf = XGBClassifier(**self.selected_params)
        clf.fit(self.X_train, self.y_train)
        
        # predict
        EYtr = clf.predict(self.X_train)
        EYtr_proba = clf.predict_proba(self.X_train)[:,1]
        EYts = clf.predict(X_test)
        EYts_proba = clf.predict_proba(X_test)[:,1]
        
        print('-- Training Error --')
        print('F1: %.4f' % F1(EYtr_proba, self.y_train))
        print('KS: %.4f' % KS(EYtr_proba, self.y_train))
        print('AUC: %.4f' % AUC(EYtr_proba, self.y_train))
        
        print('\n-- Test Error --')
        print('F1: %.4f' % F1(EYts_proba, y_test))
        print('KS: %.4f' % KS(EYts_proba, y_test))
        print('AUC: %.4f' % AUC(EYts_proba, y_test))
        
        print('\n-- Confusion Matrix --')
        confusion_matrix = pd.crosstab(pd.Series(y_test, name='Actual').reset_index(drop=True),
                                        pd.Series(EYts, name='Predicted').reset_index(drop=True),
                                        margins=True)
        print(confusion_matrix)
    
    def objective_(self, params):
        """"Objective function to be optimized.
        
        Parameters
        -----------
        params: dict
            Hyperparameter range to be optimized.
        """
        self.reg.set_params(**{k:p for k, p in zip(self.space.keys(), params)})
        return 1-np.mean(cross_val_score(self.reg, self.X_train, self.y_train, cv=3, n_jobs=-1, scoring='roc_auc'))
    