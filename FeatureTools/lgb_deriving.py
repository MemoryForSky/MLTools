# coding: utf-8


class LGB_DERIVING(object):
    """LGB特征衍生
    
    测试用例：
    ----------
    # 使用训练好的模型实例化转换类
    params = {'num_boost_round': 50,  
            'boosting_type': 'gbdt',  
            'objective': 'binary',  
            'num_leaves': 2,  
            'metric': 'auc',  
            'max_depth':1,  
            'feature_fraction':1,  
            'bagging_fraction':1, } 
    LD = LGB_DERIVING(params)

    # 数据
    Xtr = df_train[NUMERIC_COLS]
    Ytr = df_train['bad_ind']
    Xts = df_test[NUMERIC_COLS]
    Yts = df_test['bad_ind']

    # 训练&转换
    LD.fit(Xtr, Ytr)
    Xtr_dr = LD.transform(Xtr)
    Xts_dr = LD.transform(Xts)

    # 合并原始数据和衍生字段
    Xtr = pd.concat([Xtr, Xtr_dr], axis=1)
    Xts = pd.concat([Xts, Xts_dr], axis=1)

    # 使用psi筛选特征
    psi_remain = LD.filter_psi(Xtr, Xts)
    Xtr_psi = Xtr[psi_remain]
    Xts_psi = Xts[psi_remain]
    # LGB + LR
    LD.LR(Xtr_psi, Ytr, Xts_psi, Yts)

    # 使用LGB筛选特征
    params_filter = {'num_boost_round': 800,  
                  'boosting_type': 'gbdt',  
                  'objective': 'binary',  
                  'num_leaves': 31,  
                  'metric': 'auc',  
                  'max_depth':2,  
                  'feature_fraction':0.7,  
                  'bagging_fraction':0.7,
                  'max_features': 140,
                  'learning_rate': 0.05,
                  'min_child_weight': 50} 
    lgb_remain = LD.filter_importance(Xtr, Ytr, params_filter)
    Xtr_lgb = Xtr[lgb_remain]
    Xts_lgb = Xts[lgb_remain]
    # LGB + LR
    LD.LR(Xtr_lgb, Ytr, Xts_lgb, Yts)
    """
    def __init__(self, params):
        self.params = params
    
    def fit(self, X, y):
        lgb_train = lgb.Dataset(X, y, free_raw_data=False)  
        self.model = lgb.train(self.params, lgb_train)  
    
    def transform(self, data):
        leaf = self.model.predict(data, pred_leaf=True)  
        lgb_enc = OneHotEncoder()  
        #生成交叉特征
        lgb_enc.fit(leaf)
        #和原始特征进行合并
        data_leaf = pd.DataFrame(lgb_enc.transform(leaf).toarray())
        data_leaf.columns = [ 'fd' + str(x) for x in range(data_leaf.shape[1])]
        return data_leaf
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        data_trans = self.transform(X)
        return data_trans
    
    def filter_psi(self, data_tr, data_val, psi=0.01):
        # 计算0/1值下的好坏客户数
        psi_data_tr = self.make_psi_data(data_tr)  
        psi_data_val = self.make_psi_data(data_val) 
        # 计算PSI
        psi_dct = {}  
        for col in data_tr.columns:  
            psi_dct[col] = self.get_psi(psi_data_tr[col], psi_data_val[col]) 
        f = zip(psi_dct.keys(), psi_dct.values())  
        f = sorted(f,key = lambda x: x[1], reverse = False)  
        psi_df = pd.DataFrame(f)  
        psi_df.columns = pd.Series(['变量名', 'PSI'])  
        feature_lst = list(psi_df[psi_df['PSI'] < psi]['变量名'])
        return feature_lst
    
    def make_psi_data(self, data):  
        dftot = pd.DataFrame()  
        for col in data.columns:  
            zero= sum(data[col] == 0)  
            one= sum(data[col] == 1)  
            ftdf = pd.DataFrame(np.array([zero,one]))  
            ftdf.columns = [col]  
            if len(dftot) == 0:  
                dftot = ftdf.copy()  
            else:  
                dftot[col] = ftdf[col].copy()  
        return dftot  
    
    def get_psi(self, dev_data, val_data):  
        dev_cnt, val_cnt = sum(dev_data), sum(val_data)  
        if dev_cnt * val_cnt == 0:  
            return 0  
        PSI = 0  
        for i in range(len(dev_data)):  
            dev_ratio = dev_data[i] / dev_cnt  
            val_ratio = val_data[i] / val_cnt + 1e-10  
            psi = (dev_ratio - val_ratio) * math.log(dev_ratio/val_ratio)
            PSI += psi  
        return PSI 
    
    def filter_importance(self, X, y, params, importance=5):
        #训练模型
        model = self.lgb_train(X, y, params)  

        #模型贡献度放在feture中  
        feature = pd.DataFrame({'name': model.feature_name(),  
                                'importance': model.feature_importance()
                                }).sort_values(by = ['importance'],ascending = False) 
        feature_lst = list(feature[feature.importance>5].name)
        
        return feature_lst
    
    def lgb_train(self, X, y, params):  
        lgb_train = lgb.Dataset(X, y, free_raw_data=False)  
        clf = lgb.train(params, lgb_train)   
        return clf
    
    def LR(self, Xtr, Ytr, Xts, Yts):
        lgb_lm = LogisticRegression(C=0.3, class_weight='balanced', solver='liblinear')
        lgb_lm.fit(Xtr, Ytr)  

        y_pred_lgb_lm_train = lgb_lm.predict_proba(Xtr)[:, 1]  
        fpr_lgb_lm_train, tpr_lgb_lm_train, _ = roc_curve(Ytr, y_pred_lgb_lm_train)

        y_pred_lgb_lm = lgb_lm.predict_proba(Xts)[:, 1]  
        fpr_lgb_lm, tpr_lgb_lm, _ = roc_curve(Yts, y_pred_lgb_lm)  

        plt.figure(1)  
        plt.plot([0, 1], [0, 1], 'k--')  
        plt.plot(fpr_lgb_lm_train, tpr_lgb_lm_train, label='LGB + LR train')  
        plt.plot(fpr_lgb_lm, tpr_lgb_lm, label='LGB + LR test')  
        plt.xlabel('False positive rate')  
        plt.ylabel('True positive rate')  
        plt.title('ROC curve')  
        plt.legend(loc='best')  
        plt.show()  
        print('LGB+LR train ks:',abs(fpr_lgb_lm_train - tpr_lgb_lm_train).max(),
              'LGB+LR AUC:', metrics.auc(fpr_lgb_lm_train, tpr_lgb_lm_train))  
        print('LGB+LR test ks:',abs(fpr_lgb_lm - tpr_lgb_lm).max(),'LGB+LR AUC:', 
              metrics.auc(fpr_lgb_lm, tpr_lgb_lm))  
