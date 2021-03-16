# -*- coding: utf-8 -*-

"""
Created at Apr. 16 2020

@author: ZhangXiaoqiang

@infos: VarTypeClassify模块，用于区分数值型变量和类别型变量

==============================================================
"""

import warnings
from .Utils import check_data, check_exclude_vars, check_intersection

class VarTypeClassify(object):
    """判断数据的变量类型，区分数值型变量和类别型变量

    判断逻辑: 
    ----------
    （1）数值型: 变量取值全为数字
    （2）类别型: 变量取值有除了数字外的其他字符
      注: 对于类别型变量，经过数字编码后，存在无法判断为类别型变量的可能。\
      如: 数据中有一列city，取值为1,2,3...等，程序会认为为数值型，而无法自动识别为分类型。
      （自动判定后的变量列表，需要用户人工审查，如有错误，可通过cata_lst或num_lst传参矫正)

    Parameters：
    ----------
    data : Dataframe
        待区分数据集
    cate_maxBins : int, float
        最大类别数，当变量中的类别数小于或等于此值，自动判断为离散变量，大于此值， 自动判断为连续变量 ， 默认 : 10
    exclude :  str, list, tuple, 
        用户设置的不参与计算列的列名集合， 默认: 无，值得注意的是: 如果不想索引列参与计算，可将索引列参入
    cate_lst : list, str 
        可能包含'分类型变量'列名集合， 若: 用户需要指定个别变量的变量类型为分类型变量，可以对cata_lst进行传参。\
     eg: 需要'类型识别'的数据集data中有一列'city'，取值为1,2,3...等，若想要其识别结果是'分类型变量'，可以传入:  cata_lst = 'city'进行标识，
    num_lst : list, str
        能包含'连续型变量'列名集合， 使用原理与cata_lst相同
    """

    def __init__(self, data, cate_maxBins=0, exclude=None, cate_lst=None, num_lst=None):
        self.data = data
        self.cate_maxBins = cate_maxBins
        self.exclude = exclude
        self.headers = check_data(self.data)
        if isinstance(self.cate_maxBins, (int, float)):
            self.cate_maxBins = round(self.cate_maxBins)
        else:
            warnings.warn("Expect 'int', 'float' for cate_maxBins, got{}. the parameter was reset to default!".format(self.cate_maxBins))
            self.cate_maxBins = 10
        self.exclude = check_exclude_vars(self.exclude, self.headers)
        self.cate_lst = self._check_defind_vars(cate_lst)
        self.num_lst = self._check_defind_vars(num_lst)

    def classifier(self):
        """变量类型识别
        
        类别性变量：
        （1）用户指定为类别型变量；
        （2）程序识别为类别型变量；
        （3）程序识别为数值型变量，但变量类别个数小于用户设定的分类型变量类别数；
        数值型变量：
        （1）用户指定为数值型变量；
        （2）程序识别为数值型变量，且不在用户指定的类别性变量列表中；
        （3）变量类别个数大于用户设定的分类型变量类别数；
        
        returns：
        ----------
        cate_lst : list
            类别型变量名称列表
        num_lst : list
            数值型变量名称列表
        """
        trans_data = self.data[list(set(self.headers) - set(self.exclude))]
        cate_ = trans_data.select_dtypes(include=[object, bool]).columns.tolist()
        num_ = trans_data.select_dtypes(include=[int, float, 'int64', 'float64']).columns.tolist()
        self.cate_lst, self.num_lst = check_intersection(self.cate_lst, self.num_lst)
        num_ = [i for i in num_ if i not in self.cate_lst]
        self.cate_lst = list(set(self.cate_lst) | set(cate_)) if cate_ else self.cate_lst  # 2.
        uni = trans_data.apply(lambda i: i.nunique())
        cate_nunique = uni[uni <= self.cate_maxBins].index.tolist()
        num_nunique = uni[uni > self.cate_maxBins].index.tolist()
        self.cate_lst = list(set(self.cate_lst) | set(cate_nunique)) if cate_nunique else self.cate_lst
        cate_errors = list(set(self.num_lst) & set(cate_))
        if cate_errors:
            # 实际是类别型变量，用户却错分为数值型变量
            warnings.warn("Error defind! {}, expect 'Numerical type'".format(cate_errors))
            self.num_lst = list(set(self.num_lst) - set(cate_errors))
        self.num_lst = list(set(self.num_lst) | set(num_) & set(num_nunique))
        return self.cate_lst, self.num_lst

    def _check_defind_vars(self, lst):
        """检查输入参数的合法性
        """
        if lst:
            if isinstance(lst, str):
                lst = [lst]
            elif isinstance(lst, (list, tuple)):
                lst = [i for i in lst]
            else:
                warnings.warn("Expect 'str', 'list', 'tuple', got{}. was reset to default 'None'".format(type(lst)))
                lst = []
        lst = list(set(lst) & set(self.headers) - set(self.exclude)) if lst else []
        return lst