# -*- coding: utf-8 -*-

"""
Created at Apr. 16 2020

@author: ZhangXiaoqiang

@infos: CategoricalEncoder模块，用于将类别型变量转化为数值型变量，目前支持badrate和woe转换

==============================================================
"""

import numpy as np
import pandas as pd
import toad

class CategoricalEncoder(object):
    """Transform categorical features to numric.
    
    Parameters
    ----------
    encoding: string, belong to ['badrate', 'woe']
        The Encoder method of Categorical features.
    """
    def __init__(self, encoding='badrate'):
        self.encoding = encoding
    
    def fit(self, X, y, cate_names):
        """Fit the CategoricalEncoder to X.
        
        Parameters
        ----------
        X: array-like, shape [n_samples, n_cate_features]
            The data to determine the categories of each feature.
        y: array-like, shape [n_samples,]
            The labels of samples.
        cate_names : list, shape [n_cate_features,]
            The feature name of categories.
            
        Returns
        ----------
        self
        """
        if self.encoding not in ['badrate', 'woe']:
            template = ("encoding should be either 'badrate' or 'ordinal', but got %s")
            raise ValueError(template % self.encoding)
        
        # transform the data into np.array
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        self.cates_num_ = X.shape[1]
        self.cates_ = cate_names
        
        # combine X and y into Dataframe 
        y = y.reshape(-1, 1)
        data = pd.DataFrame(np.hstack((X, y)), columns=self.cates_+['targets'])
        data.targets = data.targets.astype(int)
        
        if self.encoding == 'badrate':
            cates_badrate_map = {}
            for cate in self.cates_:
                data_cate = data[[cate, 'targets']]
                cate_badrate_map = np.round(data_cate.groupby(cate).mean(), 6).to_dict()['targets']
                cates_badrate_map[cate] = cate_badrate_map
            self.cates_badrate = cates_badrate_map
            
        elif self.encoding == 'woe':
            # 分箱
            self.combiner_ = toad.transform.Combiner()
            self.combiner_.fit(data, y='targets', method='chi', min_samples=0.001)
            binned_data = self.combiner_.transform(data)
            
            #计算WOE
            self.transer_ = toad.transform.WOETransformer()
            self.transer_.fit(binned_data, binned_data.targets, exclude=['targets'])
        
    def transform(self, X):
        """Tranform X from categories to numeric.
        
        Parameters
        ----------
        X: array-like, shape [n_samples, n_cate_features]
            The data to determine the categories of each feature.
       
        Returns
        ----------
        data: array-like, shape [n_samples, n_cate_features]
            Data converted by badrate or woe.
        """
        if X.shape[1] != self.cates_num_:
            raise Exception("Categorical Encoder fit {} features, but get {} features".format(self.cates_num_, X.shape[1]))
        
        data = pd.DataFrame(X, columns=self.cates_)
        
        if self.encoding == 'badrate':
            # transform
            for cate in self.cates_:
                data.loc[:, cate] = data[cate].map(self.cates_badrate[cate])
            return data.values
        
        elif self.encoding == 'woe':
            data = self.transer_.transform(self.combiner_.transform(data))
            return data.values
    
    def fit_transform(self, X, y, cate_names):
        """Fit and tranform X from categories to numeric.
        
        Parameters
        ----------
        X: array-like, shape [n_samples, n_cate_features]
            The data to determine the categories of each feature.
        y: array-like, shape [n_samples,]
            The labels of samples.
        cate_names : list, shape [n_cate_features,]
            The feature name of categories.
            
        Returns
        ----------
        data: array-like, shape [n_samples, n_cate_features]
            Data converted by badrate or woe.
        """
        self.fit(X, y, cate_names)
        data = self.transform(X)
        return data