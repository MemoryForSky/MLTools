# -*- coding: utf-8 -*-

"""
Created at Apr. 17 2020

@author: ZhangXiaoqiang

@infos: VarsFilter模块，用于变量筛选

==============================================================
"""

import math
import xgboost as xgb  
from xgboost import plot_importance  
  
class VarsFilter(object): 
    """变量筛选模块
    
    Parameters
    -----------
    datasets: dict
        划分后的数据集，包括训练集、测试集、时间外验证集（可选）.
    dep: string
        二分类标签的名称.
    weight: string
        样本权重变量的名称，考虑到通常建模中会对样本进行抽样，为了反映真实场景下的KS值和PSI，需要使用采样比例的倒数作为权重，进行样本量还原。因此这里权重只参与KS值和PSI的计算，不参与模型训练.
    var_names: list
        特征列表.
    params: dict
        预设的模型参数字典，未指定字段使用默认值.
    max_del_var_nums: int
        单次迭代最多删除特征的个数.
    """
    def __init__(self, datasets, dep, weight, var_names, params, max_del_var_nums=0):
        self.datasets = datasets        
        self.dep = dep     
        self.weight = weight      
        self.var_names = var_names     
        self.params = params      
        self.max_del_var_nums = max_del_var_nums    
        self.row_num = 0  
        self.col_num = 0  
        self.seed = 9853
  
    def training(self, min_score=0.0001):  
        """迭代特征筛选
        
        本初步筛选方案的精华在于：使用min_score参数控制每一次删除的特征重要性，使用max_del_var_nums控制每一次循环删除特征的个数。这在一定程度上避免了特征之间的干扰。
        
        Parameters
        -----------
        min_score: float
            预设的特征重要性阈值，删除小于该值的特征
        """
        lis = self.var_names[:]
        dev_data = self.datasets.get("dev", "")  # 训练集  
        val_data = self.datasets.get("val", "")  # 测试集  
        off_data = self.datasets.get("off", "")  # 跨时间验证集
        while len(lis) > 0:   
            # 从字典中查找参数值，没有则使用第二项作为默认值  
            model = xgb.XGBClassifier(learning_rate=self.params.get("learning_rate", 0.1),
                                      n_estimators=self.params.get("n_estimators", 1000),  
                                      max_depth=self.params.get("max_depth", 1),  
                                      min_child_weight=self.params.get("min_child_weight", 1),
                                      subsample=self.params.get("subsample", 1),  
                                      objective=self.params.get("objective", "binary:logistic"),
                                      nthread=self.params.get("nthread", 10),  
                                      scale_pos_weight=self.params.get("scale_pos_weight", 1),
                                      random_state=0,  
                                      n_jobs=self.params.get("n_jobs", 10),  
                                      reg_lambda=self.params.get("reg_lambda", 1),  
                                      missing=self.params.get("missing", None) )  
            # 模型训练  
            model.fit(X=dev_data[self.var_names], y=dev_data[self.dep])  
            # 得到特征重要性  
            scores = model.feature_importances_   
            # 清空字典  
            lis.clear()      
            ''' 
            当特征重要性小于预设值时，将特征放入待删除列表。 
            当列表长度超过预设最大值时，跳出循环。 即一次只删除限定个数的特征。 
            '''  
            for (idx, var_name) in enumerate(self.var_names):  
                # 小于特征重要性预设值则放入列表  
                if scores[idx] < min_score:    
                    lis.append(var_name)  
                # 达到预设单次最大特征删除个数则停止本次循环  
                if len(lis) >= self.max_del_var_nums:     
                    break 
            # 训练集KS  
            devks = self.sloveKS(model, dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])
            # 初始化ks值和PSI  
            valks, offks, valpsi, offpsi = 0.0, 0.0, 0.0, 0.0 
            # 测试集KS和PSI  
            if not isinstance(val_data, str):  
                valks = self.sloveKS(model, val_data[self.var_names], 
                                            val_data[self.dep], 
                                            val_data[self.weight])  
                valpsi = self.slovePSI(model, dev_data[self.var_names],
                                              val_data[self.var_names])
            # 跨时间验证集KS和PSI  
            if not isinstance(off_data, str):  
                offks = self.sloveKS(model, off_data[self.var_names],
                                            off_data[self.dep],
                                            off_data[self.weight])  
                offpsi = self.slovePSI(model, dev_data[self.var_names],
                                              off_data[self.var_names])  
            
            # 将三个数据集的KS和PSI放入字典  
            dic = {"devks": float(devks), "valks": float(valks), "offks": offks,  
                "valpsi": float(valpsi), "offpsi": offpsi}  
            print("ks: ", dic) 
            print("==> del var: ", len(self.var_names), "->", len(self.var_names) - len(lis), ", ".join(lis))  
            self.var_names = [var_name for var_name in self.var_names if var_name not in lis]
        plot_importance(model)  
    
    def auto_delete_vars(self, params):  
        """递归特征删除方案
        
        使用逐个特征删除的方案获取最优的入模特征组合，首先将一个特征从变量组合中去掉，观察模型KS值和PSI的变化：如果模型没有明显变化或者表现变好，则可以删除该特征；否则，保留该特征，继续去掉下一个特征并观察。
        """
        dev_data = self.datasets.get("dev", "")  
        val_data = self.datasets.get("val", "")   
        model = xgb.XGBClassifier(max_depth=params.get("max_depth", 1),  
                                 learning_rate=params.get("learning_rate", 0.1),
                                 n_estimators=params.get("n_estimators", 1000),
                                 min_child_weight=params.get("min_child_weight",1),
                                 subsample=params.get("subsample", 1),  
                                 scale_pos_weight=params.get("scale_pos_weight",1),
                                 reg_lambda=params.get("reg_lambda", 1),  
                                 nthread=8, n_jobs=8, random_state=7)  
        model.fit(dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight])  
        devks = self.sloveKS(model, dev_data[self.var_names], dev_data[self.dep], dev_data[self.weight]) 
        valks = self.sloveKS(model, val_data[self.var_names], val_data[self.dep], val_data[self.weight])  
        train_number = 0  
        print("train_number: %s, devks: %s, valks: %s" % (train_number, devks, valks))  
        del_list = list()  
        oldks = valks  
        while True:  
            bad_ind = True  
            for var_name in self.var_names:  
                # 遍历每一个特征  
                model=xgb.XGBClassifier(max_depth=params.get("max_depth", 1),  
                                     learning_rate=params.get("learning_rate",0.1),
                                     n_estimators=params.get("n_estimators", 1000), 
                                     min_child_weight=params.get("min_child_weight",1),
                                     subsample=params.get("subsample", 1),  
                                     scale_pos_weight=params.get("scale_pos_weight",1),
                                     reg_lambda=params.get("reg_lambda", 1),  
                                     nthread=10,n_jobs=10,random_state=7)  
                # 将当前特征从模型中去掉  
                names = [var for var in self.var_names if var_name != var]  
                model.fit(dev_data[names], dev_data[self.dep], dev_data[self.weight])  
                train_number += 1  
                devks = self.sloveKS(model, dev_data[names], dev_data[self.dep], dev_data[self.weight]) 
                valks = self.sloveKS(model, val_data[names], val_data[self.dep], val_data[self.weight])
                ''' 
                比较KS是否有提升，如果有提升或者明显变化，则可以将特征去掉 
                '''  
                if valks >= oldks:  
                    oldks = valks  
                    bad_ind = False  
                    del_list.append(var_name)  
                    self.var_names = names  
                    print("(train_n: %s, devks: %s, valks: %s del_list_vars: %s" % (train_number, devks, valks, del_list)) 
                else:  
                    continue
            if bad_ind:  
                break  
        print("(End) train_n: %s, valks: %s del_list_vars: %s" % (train_number, oldks, del_list)) 
    
    def sloveKS(self, model, X, Y, Weight):  
        """计算模型在某数据集上的KS
        """
        Y_predict = [s[1] for s in model.predict_proba(X)]  
        nrows = X.shape[0] 
        # 还原权重  
        lis = [(Y_predict[i], Y.values[i], Weight.values[i]) for i in range(nrows)]
        # 按照预测概率倒序排列  
        ks_lis = sorted(lis, key=lambda x: x[0], reverse=True)        
        KS = list()  
        bad = sum([w for (p, y, w) in ks_lis if y > 0.5])  
        good = sum([w for (p, y, w) in ks_lis if y <= 0.5])  
        bad_cnt, good_cnt = 0, 0  
        for (p, y, w) in ks_lis:  
            if y > 0.5:  
                # 1*w 即加权样本个数  
                bad_cnt += w                
            else:    
                good_cnt += w               
            ks = math.fabs((bad_cnt/bad)-(good_cnt/good))  
            KS.append(ks)  
        return max(KS) 

    def slovePSI(self, model, dev_x, val_x):  
        """计算模型在训练集与测试集/时间外验证集上的稳定度指标
        """
        dev_predict_y = [s[1] for s in model.predict_proba(dev_x)]  
        dev_nrows = dev_x.shape[0]  
        dev_predict_y.sort()  
        # 等频分箱成10份  
        cutpoint = [-100] + [dev_predict_y[int(dev_nrows/10*i)] for i in range(1, 10)] + [100]  
        cutpoint = list(set(cutpoint))  
        cutpoint.sort()
        val_predict_y = [s[1] for s in list(model.predict_proba(val_x))]  
        val_nrows = val_x.shape[0]  
        PSI = 0  
        # 每一箱之间分别计算PSI  
        for i in range(len(cutpoint)-1):  
            start_point, end_point = cutpoint[i], cutpoint[i+1]  
            dev_cnt = [p for p in dev_predict_y if start_point <= p < end_point]  
            dev_ratio = len(dev_cnt) / dev_nrows + 1e-10  
            val_cnt = [p for p in val_predict_y if start_point <= p < end_point]  
            val_ratio = len(val_cnt) / val_nrows + 1e-10  
            psi = (dev_ratio - val_ratio) * math.log(dev_ratio/val_ratio)
            PSI += psi  
        return PSI   