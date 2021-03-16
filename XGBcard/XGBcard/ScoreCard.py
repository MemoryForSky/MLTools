# -*- coding: utf-8 -*-

"""
Created at Apr. 24 2020

@author: ZhangXiaoqiang

@infos: ScoreCard模块，用于生成评分卡

==============================================================
"""

import re
import functools
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from toad.metrics import KS
from collections import defaultdict

class XGBcard(object):
    """集成模型评分卡
    
    Parameters
    ----------
    X_train: array-like
        训练集的特征.
    y_train: array-like
        训练集的目标变量.
    X_test: array-like
        测试集的特征.
    y_test: array-like
        测试集的目标变量.
    feature_names: list
        用于提升树显示的特征名称列表.
    """
    def __init__(self, X_train, y_train, X_test, y_test, feature_names=None):
        self.X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        self.y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
        self.X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        self.y_test = y_test.values if isinstance(y_test, pd.Series) else y_test

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
        elif feature_names is None:
            self.feature_names = ['f'+str(i) for i in range(X_train.shape[1])]
        else:
            if len(feature_names)==X_train.shape[1]:
                self.feature_names = feature_names
            else:
                ValueError("Input parameter feature_names not match requirement.")
                
    def train(self, params):
        """模型训练
        
        Parameters
        ----------
        params: dict
            自动调参获取的最优参数字典.
        
        Return
        ------
        self.bst: object
            训练完成的xgb模型.
        """
        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train, feature_names=self.feature_names)
        self.dtest = xgb.DMatrix(self.X_test, label=self.y_test, feature_names=self.feature_names)
        
        watchlist = [(self.dtrain,'train'), (self.dtest,'test')]
        self.bst = xgb.train(params, self.dtrain, num_boost_round=params['n_estimators'], evals=watchlist)
        
        # 输出模型的迭代工程
        self.bst.dump_model("./data/model_tree.txt")
        return self.bst
    
    def eval(self):
        """模型评估
        """
        EYtr_proba = self.bst.predict(self.dtrain)
        EYtr = (EYtr_proba >= 0.5)*1
        EYts_proba = self.bst.predict(self.dtest)
        EYts = (EYts_proba >= 0.5)*1
        print('-- Training Evaluation --')
        print('AUC: %.4f' % metrics.roc_auc_score(self.y_train, EYtr))
        print('KS: %.4f' % KS(EYtr_proba, self.y_train))
        print('ACC: %.4f' % metrics.accuracy_score(self.y_train, EYtr))
        print('Recall: %.4f' % metrics.recall_score(self.y_train, EYtr))
        print('F1-score: %.4f' % metrics.f1_score(self.y_train, EYtr))
        print('Precesion: %.4f' % metrics.precision_score(self.y_train, EYtr))
        print('Confusion Matrix: \n', metrics.confusion_matrix(self.y_train, EYtr))
        
        print('\n-- Test Evaluation --')
        print('AUC: %.4f' % metrics.roc_auc_score(self.y_test, EYts))
        print('KS: %.4f' % KS(EYts_proba, self.y_test))
        print('ACC: %.4f' % metrics.accuracy_score(self.y_test, EYts))
        print('Recall: %.4f' % metrics.recall_score(self.y_test, EYts))
        print('F1-score: %.4f' % metrics.f1_score(self.y_test, EYts))
        print('Precesion: %.4f' % metrics.precision_score(self.y_test, EYts))
        print('Confusion Matrix: \n', metrics.confusion_matrix(self.y_test, EYts))

    def scoring(self, y_pred, pdo=40, base_odds=0.1, base_score=600):
        """生成评分
        
        评分卡分数的计算逻辑：PDO = 40, 600 = 10:1 odds
        
        Parameters
        ----------
        y_pred: array-like
           数据集的预测概率. 
        pdo: int
           违约概率翻倍的评分. 
        base_odds: int
            客户违约相对于不违约的基础概率比值，默认设置为10.
        base_score: int
            基础分数，默认为600 = 10:1 odds.
        
        Returns
        -------
        y_pred: array-like
            客户评分.
        """
        A,B = self.get_ab_(pdo, base_odds, base_score)
        score_func = functools.partial(self.score_, a=A, b=B)
        y_pred = np.array(list(map(score_func, y_pred))).astype(int)
        return y_pred

    def score_card(self, cates_lst, cates_encoder_map, pdo=40, base_odds=0.1, base_score=600):
        """生成评分卡明细
        
        评分卡分数的计算逻辑：PDO = 40, 600 = 10:1 odds
        
        Parameters
        ----------
        cates_lst: list
            类别型变量列表.
        cates_encoder_map: dict
            类别型变量编码的字典
        pdo: int
           违约概率翻倍的评分. 
        base_odds: int
            客户违约相对于不违约的基础概率比值，默认设置为10.
        base_score: int
            基础分数，默认为600 = 10:1 odds.
        
        Returns
        -------
        xgb_score_card: DataFrame
            评分卡明细.
        """
        A,B = self.get_ab_(pdo, base_odds, base_score)
        # 解析xgboost的所有树
        with open("./data/model_tree.txt", "r") as f:  
            model_txt = f.read()  

        pattern = r'booster\[(\d+)\]:\s+0:\[(.+)<(.+)\] yes=(\d+),no=(\d+),missing=(\d+)\s+1:leaf=(.+)\s+2:leaf=(.+)'
        results = re.findall(pattern, model_txt) 
        
        tree_info = pd.DataFrame(results, columns=['tree_num', 'features', 'value', 'yes', 'no', 'missing', 'leaf_L', 'leaf_R'])
        tree_info.leaf_L = tree_info.leaf_L.astype(float)
        tree_info.leaf_R = tree_info.leaf_R.astype(float)
        
        tree_cut_info = tree_info.groupby(['features','value'])['leaf_L','leaf_R'].sum().reset_index()
        tree_cut_info.loc[:,'value_float'] = np.round(tree_cut_info.value.astype(float),4)
        
        xgb_score_card = self.get_scorecard_(tree_cut_info, cates_lst, cates_encoder_map, A, B)
        return xgb_score_card
    
    def get_score_details(self, X):
        pass
    
    def get_scorecard_(self, tree_cut_info, cates_lst, cates_encoder_map, A, B):
        """转化评分卡
        
        Parameters
        ----------
        tree_cut_info: DataFrame
            从提升树的迭代工程中解析的树结构数据.
        cates_lst: list
            类别型变量列表.
        cates_encoder_map: dict
            类别型变量编码的字典.
        A: float
            基础分值.
        B: float
            计算特征不同分箱分值的系数.
            
        Returns
        -------
        xgb_score_card: DataFrame
            评分卡明细.
        """
        xgb_score_card = pd.DataFrame()
        self.features_bin = defaultdict(list)
        for feature in tree_cut_info.features.unique():
            # 获取每个特征的划分点
            sub_feature = tree_cut_info[tree_cut_info.features==feature].sort_values(by='value_float').reset_index(drop=True)
            # 生成特征分箱及得分
            feature_score = dict()
            bin_cut = sub_feature.shape[0]
            for i in range(bin_cut+1):
                cates_badrate_bin_map = []
                feature_score['features'] = [feature]
                # 数值分箱
                if i==0:
                    feature_score['num_box'] = ['<' + sub_feature.loc[i, 'value']]
                elif i==bin_cut:
                    feature_score['num_box'] = ['>' + sub_feature.loc[i-1, 'value']]
                else:
                    feature_score['num_box'] = ['[' + sub_feature.loc[i-1, 'value'] + ',' + sub_feature.loc[i, 'value'] + ']']
                # 分类变量名称映射
                if feature not in cates_lst:
                    feature_score['box'] = feature_score['num_box']
                    # 获取数值型变量的分箱结果
                    if i < bin_cut:
                        self.features_bin[feature].append(sub_feature.loc[i, 'value_float'])
                else:
                    if i==0:
                        for key, value in cates_encoder_map[feature].items():
                            if value <= sub_feature.loc[i, 'value_float']:
                                cates_badrate_bin_map.append(key)
                    elif i==bin_cut:
                        for key, value in cates_encoder_map[feature].items():
                            if value >= sub_feature.loc[i-1, 'value_float']:
                                cates_badrate_bin_map.append(key)
                    else:
                        for key, value in cates_encoder_map[feature].items():
                            if sub_feature.loc[i-1, 'value_float'] <= value <= sub_feature.loc[i, 'value_float']:
                                cates_badrate_bin_map.append(key)
                    feature_score['box'] = [' , '.join(map(str, cates_badrate_bin_map))]
                    # 获取类别型变量的分箱结果
                    self.features_bin[feature].append(cates_badrate_bin_map)
                # 分箱得分
                feature_score['value'] = [sub_feature.leaf_L[i:].values.sum() + sub_feature.leaf_R[:i].values.sum()]
                xgb_score_card = pd.concat([xgb_score_card, pd.DataFrame(feature_score)], axis=0, ignore_index=True) 
        
        bin_score_func = functools.partial(self.bin_score_, b=B)
        xgb_score_card['pred'] = xgb_score_card.apply(lambda x: self.sigmoid_(x.value) if x.value else None, axis=1)  
        xgb_score_card['score'] = xgb_score_card.apply(lambda x: bin_score_func(x.pred) if x.pred else None ,axis=1).astype(int)  
        base_score = pd.DataFrame({'features':['BASE_SCORE'], 'num_box':[None],'box':[None],'value':[None],'pred':[None],'score':[A]})
        xgb_score_card = pd.concat([base_score, xgb_score_card], axis=0)
        return xgb_score_card
    
    def get_ab_(self, pdo, base_odds, base_score):
        """获取评分映射中的A和B
        
        Parameters
        ----------
        pdo: int
           违约概率翻倍的评分. 
        base_odds: int
            客户违约相对于不违约的基础概率比值，默认设置为10.
        base_score: int
            基础分数，默认为600 = 10:1 odds.
            
        Returns
        -------
        A: float
            基础分值.
        B: float
            计算特征不同分箱分值的系数.
        """
        B = pdo/np.log(2)
        A = base_score + B*np.log(base_odds)
        return A,B
    
    def score_(self, pred, a, b):
        """坏客户的概率转化为评分
        
        Parameters
        ----------
        pred: float
            客户违约的预测概率.
        a: float
            基础分值.
        b: float
            计算特征不同分箱分值的系数.
            
        Returns
        -------
        score: float
            评分分值.
        """
        score = a + b*(np.log((1- pred)/ pred)) 
        return score
    
    def bin_score_(self, pred, b): 
        """分箱的评分转换
        
        Parameters
        ----------
        pred: float
            客户违约的预测概率.   
        b: float
            计算特征不同分箱分值的系数.
            
        Returns
        -------
        score: float
            评分分值.
        """
        score = b*(np.log((1- pred)/ pred))  
        return score
    
    def sigmoid_(self, fx):
        """sigmoid函数: 数值转化为概率
        
        Parameters
        ----------
        fx: float
            提升树输出的预测分数.
        
        Returns
        -------
        pred: float
            客户违约的预测概率. 
        """
        pred = 1/float(1+np.exp(-fx))
        return pred  
    