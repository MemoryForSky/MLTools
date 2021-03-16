# -*- coding: utf-8 -*-

"""
Created at Apr. 17 2020

@author: ZhangXiaoqiang

@infos: Utils模块，包括评分卡工具的通用函数

==============================================================
"""

import warnings
import numpy as np
import pandas as pd
from collections import Counter

def check_data(X):
    """检查输入数据集并提取字段名称
    
    Parameters
    ----------
    X: Dataframe
        输入数据集.

    Returns
    ----------
    headers: list
        数据集的字段名称列表.
    
    """
    if isinstance(X, pd.DataFrame):
        headers = [var for var in X.columns]
        if len(set(headers)) != len(headers):
            counts = Counter(headers)
            raise ValueError("Duplicate columns: {} were found in input data X, please check!".format(
                [k for k, v in counts.items() if v > 1]))
    else:
        raise TypeError("Only 'pandas.Dataframe'"
                        " is support for input X got{}".format(type(X)))
    return headers

def check_exclude_vars(exclude, headers):
    """检查输入的保留变量
    
    Parameters
    ----------
    exclude: str, list, tuple
        保留变量名称列表.
    headers: list
        输入数据集的字段名称列表.

    Returns
    ----------
    exclude: list
        处理后的保留变量名称列表.
    """
    if not exclude:
        exclude = []
    else:
        if isinstance(exclude, str):
            exclude = [exclude]
        elif isinstance(exclude, (tuple, list)):
            exclude = [var for var in exclude]
        else:
            warnings.warn("Expect 'str', 'list' or 'tuple' for exclude got {}".format(exclude))
            exclude = []
        if not set(list(exclude)) & set(list(headers)) == set(exclude):
            warnings.warn("'{}' was not found in input data 'X', please check 'exclude'".format(
                np.setdiff1d(exclude, headers)))
            exclude = list(set(list(exclude)) & set(list(headers)))
    return exclude

def check_intersection(lst1, lst2):
    """输入列表的冲突变量处理
    
    删除num_lst和cate_lst中的共有变量，由系统自动判断其属性
    
    Parameters
    ----------
    lst1: list
        输入变量名称列表1.
    lst2: list
        输入变量名称列表2.
    
    Returns
    ----------
    lst1: list
        删除冲突变量后的变量名称列表1.
    lst2: list
        删除冲突变量后的变量名称列表2.
    """
    inters = list(set(lst1) & set(lst2))
    if inters:
        warnings.warn("Find {0} both in num_lst and cate_lst, was removed from both!".format(inters))
        lst1 = list(set(lst1) - set(inters))
        lst2 = list(set(lst2) - set(inters))
    return lst1, lst2

def PSI_nums(dev_x, val_x): 
    """对数值型变量计算PSI
    
    计算不同时间范围内数值型数据的稳定性.
    
    Parameters
    ----------
    dev_x: array-like
        训练集中某变量的数据.
    val_x: array-like
        跨时间验证集中某变量的数据. 
        
    Return
    ----------
    PSI: float
        变量稳定性系数PSI
    """
    dev_nrows = dev_x.shape[0]  
    val_nrows = val_x.shape[0] 
    # 归一化
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    dev_x_mm = min_max_scaler.fit_transform(dev_x.reshape(-1,1)).flatten() 
    val_x_mm = min_max_scaler.transform(val_x.reshape(-1,1)).flatten() 
    dev_x_mm.sort()  
    # 等频分箱成10份  
    cutpoint = [-10] + [dev_x_mm[int(dev_nrows/10*i)] for i in range(1, 10)] + [10]  
    cutpoint = list(set(cutpoint))  
    cutpoint.sort()
    PSI = 0  
    # 每一箱之间分别计算PSI  
    for i in range(len(cutpoint)-1):  
        start_point, end_point = cutpoint[i], cutpoint[i+1]  
        dev_cnt = [p for p in dev_x_mm if start_point <= p < end_point]  
        dev_ratio = len(dev_cnt) / dev_nrows + 1e-10  
        val_cnt = [p for p in val_x_mm if start_point <= p < end_point]  
        val_ratio = len(val_cnt) / val_nrows + 1e-10  
        psi = (dev_ratio - val_ratio) * np.log(dev_ratio/val_ratio)
        PSI += psi  
    return PSI  

def PSI_cates(dev_x, val_x): 
    """对类别型变量计算PSI
    
    计算不同时间范围内类别型数据的稳定性.
    
    Parameters
    ----------
    dev_x: array-like
        训练集中某变量的数据.
    val_x: array-like
        跨时间验证集中某变量的数据.  
    
    Return
    ----------
    PSI: float
        变量稳定性系数PSI
    """
    dev_nrows = dev_x.shape[0]  
    val_nrows = val_x.shape[0] 
    dev_x_set = set(dev_x)
    PSI = 0  
    # 每一个取值分别计算PSI  
    for i in dev_x_set:   
        dev_cnt = [p for p in dev_x if p == i]  
        dev_ratio = len(dev_cnt) / dev_nrows + 1e-10  
        val_cnt = [p for p in val_x if p == i]  
        val_ratio = len(val_cnt) / val_nrows + 1e-10  
        psi = (dev_ratio - val_ratio) * np.log(dev_ratio/val_ratio)
        PSI += psi  
    return PSI  