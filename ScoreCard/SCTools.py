"""
@author: Zhang Xiaoqiang

@infos：评分卡模块化程序
程序包含五个模块：（1）数据预处理；（2）特征筛选；（3）构建模型；（4）验证模型；（5）持久化模型；

@logs:

# TODO 1、分箱单调性过滤；2、分箱U型过滤
"""


import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import toad
from toad.metrics import KS, AUC
from toad.plot import badrate_plot, bin_plot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.rcParams['font.sans-serif'] = ['simHei']


class DataPreprocessing(object):
    """数据预处理"""
    def __init__(self, train_data, y, feats_info, oot_data=None):
        self.train_data = train_data  # 训练数据
        self.oot_data = oot_data  # 跨期验证数据
        self.feats_info = feats_info
        self.y = y
        self.index_name_map = dict(zip(self.feats_info.feat_id, self.feats_info.chn_name))

    def feat_filter(self, empty=0.5, iv=0.02, corr=0.7, exclude=None):
        """特征粗筛选"""
        self.selected_data, drop_dict = toad.selection.select(self.train_data, target=self.y,
                                                              empty=empty, iv=iv, corr=corr,
                                                              exclude=exclude, return_drop=True)
        return drop_dict

    def split_data(self, test_size=0.25, random_state=2020, **args):
        """划分数据集"""
        Xtr, Xts, Ytr, Yts = train_test_split(self.selected_data.drop(self.y, axis=1),
                                              self.selected_data[self.y], test_size=test_size,
                                              random_state=random_state, **args)
        self.data_tr = pd.concat([Xtr, Ytr], axis=1)
        self.data_tr['type'] = 'train'
        self.data_ts = pd.concat([Xts, Yts], axis=1)
        self.data_ts['type'] = 'test '
        self.data_all = pd.concat([self.data_tr, self.data_ts], axis=0)

    def bins_woe(self, combiner, binned_data_tr):
        """卡方分箱和wOE转换"""
        # 分箱
        self.combiner = combiner
        self.bins = self.combiner.export()
        # WOE转换
        self.transer = toad.transform.WOETransformer()
        data_tr_woe = self.transer.fit_transform(binned_data_tr, binned_data_tr[self.y],
                                                 exclude=[self.y, 'type'])
        data_ts_woe = self.transer.transform(self.combiner.transform(self.data_ts))
        return data_tr_woe, data_ts_woe

    def adj_bins_woe(self, adj_bins={}, empty_separate=True):
        """调整分箱和wOE转换"""
        # 调整分箱
        self.combiner.set_rules(adj_bins)
        binned_data_tr = self.combiner.transform(self.data_tr)
        self.bins = self.combiner.export()
        # WOE转换
        self.transer = toad.transform.WoETransformer()
        data_tr_woe = self.transer.fit_transform(binned_data_tr, binned_data_tr[self.y],
                                                                exclude=[self.y, 'type'])
        data_ts_woe = self.transer.transform(self.combiner.transform(self.data_ts))
        return data_tr_woe, data_ts_woe

    def rebins_woe(self, selected_vars, oot=False):
        """全量训练数据重新建模"""
        # 分箱
        self.data_all2 = self.data_all[selected_vars + [self.y]]
        combiner = toad.transform.Combiner()
        binned_data = combiner.fit_transform(self.data_all2, y=self.y, method='chi', min_samples=0.05,
                                             empty_separate=True)
        feats_bins = combiner.export()
        # WOE编码
        transer = toad.transform.WOETransformer()
        data_woe = transer.fit_transform(binned_data, binned_data[self.y], exclude=[self.y])
        bins_woe = transer.export()
        if oot:
            self.oot = self.oot_data[selected_vars+[self.y]]
            oot_data_woe = transer.transform(combiner.transform(self.oot))
            return data_woe, oot_data_woe, feats_bins, bins_woe
        else:
            return data_woe, feats_bins, bins_woe

    def bins_badrate(self, remain_vars):
        """不同分箱的坏样本率"""
        temp_data = self.combiner.transform(self.data_tr, labels=True)
        for feat in remain_vars:
            bin_plot(temp_data, x=feat, target=self.y)
            if feat in self.index_name_map:
                plt.title(self.index_name_map[feat])

    def train_test_badrate(self, remain_vars):
        """训练集和测试集不同分箱的badrate是否有交叉，交叉影响泛化能力"""
        temp_data = self.combiner.transform(self.data_all, labels=True)
        for feat in remain_vars:
            badrate_plot(temp_data, target='target', x='type', by=feat)
            if feat in self.index_name_map:
                plt.title(self.index_name_map[feat])


class VarsFilter(object):
    """特征筛选"""
    def __init__(self, data_tr, data_ts, target='target'):
        self.data_tr = data_tr
        self.data_ts = data_ts
        self.target = target
        self.remain_vars = []
        self.del_vars={}
        self.vars_iv = None

    def vars_stability(self, psi=0.1):
        """删除psi大于0.1的变量"""
        for var in self.data_tr.columns:
            if var == 'type':
                continue
            if var == 'target':
                self.remain_vars.append(var)
                continue
            real_psi = toad.metrics.PSI(self.data_tr[var], self.data_ts[var])
            if real_psi < psi:
                self.remain_vars.append(var)
            else:
                self.del_vars[var] = "psi > {psi}({real_psi})".format(psi=psi, real_psi=real_psi)

        self.data_tr = self.data_tr[self.remain_vars]
        self._print_logs('PSI筛选')

    def filter_iv(self, iv=0.02):
        """n保留iv值大于0.02的变量"""
        data_tr_iv = toad.quality(self.data_tr, self.target)
        self.remain_vars = list(data_tr_iv[data_tr_iv.iv > iv].index.values)
        del_vars = list(data_tr_iv[data_tr_iv.iv <= iv].index.values)
        keep_vars = self.remain_vars + [self.target]
        self.data_tr = self.data_tr[keep_vars]
        for var in del_vars:
            self.del_vars[var] = 'iv < {iv}({real_iv})'.format(iv=iv, real_iv=data_tr_iv.loc[var, 'iv'])
        self.vars_iv = data_tr_iv.loc[self.remain_vars, :]
        self._print_logs('IV值筛选')

    def filter_corr(self, corr=0.8):
        """删除变量之间的相关性大于0.8的变量"""
        self.data_tr, drop_lst = toad.selection.drop_corr(self.data_tr, target=self.target,
                                                          threshold=corr, by='IV', return_drop=True)
        for var in drop_lst:
            self.del_vars[var] = 'corr>0.8'
            self.remain_vars.remove(var)
        self.vars_iv = self.vars_iv.loc[self.remain_vars, :]
        self._print_logs('变量相关性筛选')

    def filter_stepwise(self, data_tr, drop_lst):
        """删除不显著的变量"""
        self.data_tr = data_tr
        for var in drop_lst:
            self.del_vars[var] = 'stepwise'
            self.remain_vars.remove(var)
        self.vars_iv = self.vars_iv.loc[self.remain_vars, :]
        self._print_logs('逐步回归筛选')

    def filter_vif(self, vif=3):
        """删除共线性大于3的变量"""
        self.data_tr, drop_lst = toad.selection.drop_vif(self.data_tr, threshold=vif, return_drop=True,
                                                         exclude=[self.target])
        for var in drop_lst:
            self.del_vars[var] = 'vif > 3'
            self.remain_vars.remove(var)
        self.vars_iv = self.vars_iv.loc[self.remain_vars, :]
        self._print_logs('共线性筛选')

    def _print_logs(self, method):
        print("{}：删除{}个变量，剩余{}个变量".format(method, len(self.del_vars), len(self.remain_vars)))


class BuildModel(object):
    """构建模型"""
    def __init__(self, data, y='target'):
        self.data = data
        self.y = y
        self.sorted_feats = list(toad.quality(self.data, target=y).index)
        self.remain_vars = []
        self.del_vars = {}

    def filter_coef_pos(self, feat_nums_init=5):
        """筛选符号为正的变量"""
        self.remain_vars = self.sorted_feats[:feat_nums_init]
        lr = LogisticRegression(class_weight='balanced')
        lr.fit(self.data[self.remain_vars], self.data[self.y])

        # 首先选择IV最高的少量基本变量（系数均为正）
        while np.any(lr.coef_ < 0):
            if feat_nums_init <= 1:
                raise ("Error:LR cannot execute.")
            feat_nums_init -= 1
            self.remain_vars = self.sorted_feats[:feat_nums_init]
            lr.fit(self.data[self.remain_vars], self.data[self.y])

        # 按优先级从高到低逐渐添加变量，若使系数为负，舍弃该变量
        for feat in self.sorted_feats[feat_nums_init:]:
            self.remain_vars.append(feat)
            lr.fit(self.data[self.remain_vars], self.data[self.y])
            if np.any(lr.coef_ < 0):
                self.remain_vars.pop()
                self.del_vars[feat] = 'coef < 0'
        self.data = self.data[self.remain_vars + [self.y]]
        self._print_logs()

    def filter_p_value(self, p_thred=0.1):
        """筛选p值小于0.1的变量"""
        X = self.data.drop(self.y, axis=1)
        Y = self.data[self.y]
        X_c = sm.add_constant(X)
        self.est = sm.Logit(Y, X_c)
        self.est = self.est.fit()
        self.remain_vars = list(X.columns[self.est.pvalues.values[1:] < p_thred])
        self.data = self.data[self.remain_vars + [self.y]]
        del_vars = list(X.columns[self.est.pvalues.values[1:] >= p_thred])
        for var in del_vars:
            self.del_vars[var] = 'p value>0.1'.format(str(p_thred))
        self._print_logs()

    def _print_logs(self):
        print("删除{}个变量，剩余{}个变量".format(len(self.del_vars), len(self.remain_vars)))

    def train(self, lr):
        """训练LR模型"""
        Xtr = self.data.drop(self.y, axis=1)
        Ytr = self.data[self.y]
        lr.fit(Xtr, Ytr)
        print("最终入模变量{}个.".format(Xtr.shape[1]))
        return lr


class ValidationModel(object):
    """验证模型"""
    def __init__(self, data_tr, data_ts, remain_vars, y='target'):
        # 验证数据
        self.remain_vars = remain_vars
        self.y = y
        self.Xtr = data_tr[remain_vars]
        self.Ytr = data_tr[y]
        self.Xts = data_ts[remain_vars]
        self.Yts = data_ts[y]

    def ks_auc(self, clf):
        """评估KS和AUC"""
        print("------训练集评估------")
        self.EYtr_proba = clf.predict_proba(self.Xtr)[:, 1]
        print("KS:", KS(self.EYtr_proba, self.Ytr))
        print("AUC:", AUC(self.EYtr_proba, self.Ytr))
        print("-----------测试集评看---------")
        self.EYts_proba = clf.predict_proba(self.Xts)[:, 1]
        print("KS:", KS(self.EYts_proba, self.Yts))
        print("AUC:", AUC(self.EYts_proba, self.Yts))

    @staticmethod
    def plot_roc_curve(labels, preds, co_labels=None, co_preds=None, figsize=(6, 4)):
        """plot roc curve.

        Parameters
        ----------
        labels: array, shape=[n samples]
            True binary labels.
        preds: array, shape=[n samples]
            Target scores, probability estimates of the positive class.
        co_labels: array, shape=[m samples]
            True binary labels of evaluation data.
        co_preds: array, shape=[m samples]
            Target scores of evaluation data, probability estimates of the positive class.
        figsize: tuple
            The figure size of roc curve.
        """
        fpr_xgb, tpr_xgb, _ = roc_curve(labels, preds)
        auc_ = auc(fpr_xgb, tpr_xgb)
        Flag = False
        if co_preds is not None and co_labels is not None:
            co_fpr_xgb, co_tpr_xgb, _ = roc_curve(co_labels, co_preds)
            co_auc_ = auc(co_fpr_xgb, co_tpr_xgb)
            Flag = True

        # plot roc curve
        _ = plt.figure(figsize=figsize)
        if Flag:
            plt.plot(fpr_xgb, tpr_xgb, label='train')
            plt.plot(co_fpr_xgb, co_tpr_xgb, label='eval')
        else:
            plt.plot(fpr_xgb, tpr_xgb)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        if Flag:
            plt.title('ROC Curve \n (Train AUC: %.3f   Test AUC: %.3f)' % (auc_, co_auc_))
            plt.legend(loc='best')
        else:
            plt.title('ROC Curve \n (AUC: %.3f)' % auc_)
        plt.show()

    @staticmethod
    def plot_ks_curve(labels, preds, bins=10, figsize=(6, 4), title='KS Curve'):
        """Calculate ks value and plot ks curve.

        Parameters
        ----------
        labels: array, shape=[n samples]
            True binary labels.
        preds: array, shape=[n samples]
            Probability estimates of the positive class.
        bins: int
            Divide the input data set into several parts for KS calculation.
        figsize: tuple
            The figure size of roc curve.
        title: str
            The title of figure.
        """
        def n0(x): return sum(x == 0)
        def n1(x): return sum(x == 1)
        pred = preds
        bad = labels
        n = bins
        data = pd.DataFrame({'bad': bad, 'pred': pred})
        df_ks = data.sort_values(by='pred', ascending=False).reset_index(drop=True)  \
                    .assign(group=lambda x: np.ceil((x.index + 1) / (len(x.index) / n)))  \
                    .groupby('group')['bad'].agg([n0, n1])  \
                    .reset_index().rename(columns={'n0': 'good', 'n1': 'bad'}) \
                    .assign(group=lambda x: (x.index+1)/len(x.index),
                            cumgood=lambda x: np.cumsum(x.good)/sum(x.good),
                            cumbad=lambda x: np.cumsum(x.bad)/sum(x.bad)
                            ).assign(ks=lambda x: abs(x.cumbad - x.cumgood))
        df_ks = pd.concat([pd.DataFrame({'group': 0, 'good': 0, 'bad': 0, 'cumgood': 0,
                                         'cumbad': 0, 'ks': 0}, index=np.arange(1)),
                           df_ks], ignore_index=True)
        seri_ks = df_ks.loc[lambda x: x.ks == max(x.ks)].sort_values('group').iloc[0]

        # plot ks curve
        fig, ax = plt.subplots(figsize=figsize)
        l1, = plt.plot(df_ks.group, df_ks.ks, color='blue', linestyle='-')  # 绘制ks曲线
        l2, = plt.plot(df_ks.group, df_ks.cumgood, color='green', linestyle='-')
        l3, = plt.plot(df_ks.group, df_ks.cumbad, 'k-')
        l4, = plt.plot([seri_ks['group'], seri_ks['group']], [0, seri_ks['ks']], 'r--')
        plt.text(seri_ks['group'], max(df_ks['ks']), 'KS= %0.3f' % max(df_ks['ks']))
        plt.legend(handles=[11, 12, 13, 14],
                   labels=['ks-curve', 'fpr-curve', 'tpr-curve'],
                   loc='upper left')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title(title)
        ax.set_xlabel('population ratio')
        ax.set_ylabel('total Good/Bad ratio')

    @staticmethod
    def plot_divergence(labels, preds, plot_type='kde', figsize=(6, 4)):
        """plot divergence of Good/Bad.

        Parameters
        ----------
        labels: array, shape=[n samples]
            True binary labels.
        preds: array, shape=[n samples]
            Target scores, probability estimates of the positive class.
        plot_type: str
            The type of figure.
        plot type: string, value=[' kde',' dist']
            The type of plotting univariate distributions.
        figsize: tuple
            The figure size of roc curve.
        """
        pred_0 = preds[labels.values == 0]
        pred_1 = preds[labels.values == 1]

        fig, ax = plt.subplots(figsize=figsize)

        if plot_type == 'dist':
            sns.distplot(pred_0, label='Good')
            sns.distplot(pred_1, label='Bad')
        elif plot_type == 'kde':
            sns.kdeplot(pred_0, label='Good')
            sns.kdeplot(pred_1, label='Bad')

        ax.set_title('divergence of Good/Bad')
        ax.set_xlabel('probability estimates of the positive')
        ax.set_ylabel('probability density')
        plt.legend(loc='best')

    @staticmethod
    def plot_ks_badrate(data, plot_lst=None):
        """绘制ks和badrate曲线"""
        plt.rcParams['font.sans-serif']=['Microsoft YaHei']
        yticks = mtick.FormatStrFormatter("%.2f%%")  # 设置百分比形式的坐标轴
        xaxis_data = [i for i in range(data.shape[0])]
        yaxis0_data = list(data[plot_lst[0]].values * 100)
        yaxis1_data = list(data[plot_lst[1]].values)

        # badrate
        fig = plt.figure(figsize=(15, 8))
        ax1 = fig.add_subplot(111)
        ax1.plot(xaxis_data, yaxis0_data, 'or-', label='bad_ratio')
        ax1.yaxis.set_major_formatter(yticks)
        for i, (_x, _y) in enumerate(zip(xaxis_data, yaxis0_data)):
            # 将数值显示在图形上
            plt.text(_x, _y, np.round(yaxis0_data[i], 1).astype(str)+'%', color='black', fontsize=10)
        ax1.legend(loc=1)
        ax1.set_ylim([0, 100])
        ax1.set_ylabel('bad_ratio')
        plt.legend(prop={'family': 'SimHei', 'size': 18}, loc="upper left")

        # KS
        ax2 = ax1.twinx()  # this is the important function
        ax2.plot(xaxis_data, yaxis1_data, 'b-', label='ks')
        for i, (x, y) in enumerate(zip(xaxis_data, yaxis1_data)):
            plt.text(x, y, np.round(yaxis1_data[i], 2), color='black', fontsize=10)  # 将数值显示在图形上
        ax2.legend(loc=2)
        ax2.set_ylim([0, 1])  # 设置y轴取值范围
        ax2.set_ylabel('KS')
        plt.legend(prop={'family': 'simHei', 'size': 18}, loc="upper right")

    @staticmethod
    def plot_badrate_count(data, plot_lst=None):
        """绘制ks和样本占比曲线"""
        plt.rcParams["font.sans-serif"] = ['Microsoft YaHei']
        yticks = mtick.FormatStrFormatter("%.2f%%")  # 设置百分比形式的坐标轴
        xaxis_data = [i for i in range(data.shape[0])]
        yaxis0_data = list(data[plot_lst[0]].values * 100)
        yaxis1_data = list(data[plot_lst[1]].values)

        # count
        fig=plt.figure(figsize=(15, 8))
        ax1 = fig.add_subplot(111)
        ax1.bar(xaxis_data, yaxis1_data, label='count')
        ax1.legend(loc=2)
        ax1.set_ylim([0, max(yaxis1_data) * 1.5])  # 设置y翰取值范围
        ax1.set_ylabel('count')
        plt.legend(prop={'family': 'SimHei', 'size': 18}, loc="upper left")

        # badrate
        ax2 = ax1.twinx()
        ax2.plot(xaxis_data, yaxis0_data, 'or-', label='bad_ratio')
        ax2.yaxis.set_major_formatter(yticks)
        for i, (x, y) in enumerate(zip(xaxis_data, yaxis0_data)):
            plt.text(x, y, np.round(yaxis0_data[i], 1).astype(str) + '%', color='black', fontsize=10)
        ax2.legend(loc=1)
        ax2.set_ylim([0, 100])
        ax2.set_ylabel('bad ratio')
        plt.legend(prop={'family': 'SimHei', 'size': 18}, loc="upper right")
        plt.grid(b=None)

    @staticmethod
    def plot_score_badrate_count(data):
        """绘制ks和样本占比曲线"""
        plt.rcParams['font.sans-serif']=['Microsoft YaHei']
        yticks = mtick.FormatStrFormatter("%.2f%%")  # 设置百分比形式的坐标轴
        xaxis_data = [i for i in range(data.shape[0])]
        xaxis_data_bins = list(data['bins'])
        yaxis0_data = list(data['mean'].values*100)
        yaxis1_data = list(data['count'].values)

        # count
        fig = plt.figure(figsize=(18, 8))
        ax1 = fig.add_subplot(111)
        ax1.bar(xaxis_data, yaxis1_data, label='count')
        ax1.legend(loc=2)
        ax1.set_ylim([0, max(yaxis1_data) * 1.5])  # 设置y翰取值范围
        ax1.set_ylabel('count')
        plt.legend(prop={'family': 'SimHei', 'size': 18}, loc="upper left")

        # badrate
        ax2 = ax1.twinx()
        ax2.plot(xaxis_data, yaxis0_data, 'or-', label='bad_ratio')
        ax2.yaxis.set_major_formatter(yticks)
        for i, (x, y) in enumerate(zip(xaxis_data, yaxis0_data)):
            plt.text(x, y, np.round(yaxis0_data[i], 1).astype(str) + '%', color='black', fontsize=10)
        ax2.legend(loc=1)
        ax2.set_ylim([0, 100])
        ax2.set_ylabel('bad_ratio')
        plt.legend(prop={' family': 'SimHei', ' size': 18}, loc="upper right")
        plt.grid(b=None)
        plt.xticks(xaxis_data, xaxis_data_bins)


class PersistModel(object):
    """持久化模型

    Parameters
    ----------
    data：
    oot：
    y：
    selected_vars：
    feats_info：
    """
    def __init__(self, data, y, selected_vars, feats_info):
        self.Xtr = data[selected_vars]
        self.Ytr = data[y]
        self.selected_vars = selected_vars
        self.feats_info = feats_info
        self.y = y
        self.feats_info()

    def retrain(self):
        """全部数据重新训练LR模型"""
        self.clf = LogisticRegression(class_weight='balanced')
        self.clf.fit(self.Xtr, self.Ytr)

    def save_model(self, feats_bins, bins_woe, model_tab):
        """保存模型参数表"""
        # 模型分箱参数
        df_feats_bins = self.bin_card(feats_bins, bins_woe)
        feats_coef = pd.DataFrame(zip(self.selected_vars, list(self.clf.coef_[0])),
                                  columns=['feat id', 'coef'])
        model_info = pd.merge(df_feats_bins, feats_coef, on='feat id', how='inner')
        intercept_info = pd.DataFrame({'feat id': ['intercept'], 'feat_bin': ['-1'],
                                       'bin_index': [0], 'feat_str': [''], 'woe': [1],
                                       'coef': [self.clf.intercept_[0]]})
        model_info = pd.concat([model_info, intercept_info], axis=0)
        model_info = pd.merge(self.feats_info_add, model_info, on='feat id', how='right')
        model_info['theta_x'] = model_info.coef * model_info.woe

        return model_info

    def _feats_info(self):
        # 入模变量信息
        feats_iv = toad.quality(self.Xtr, self.Ytr)[['iv']]
        df_feats_iv = feats_iv.reset_index()
        df_feats_iv.columns = ['feat id', 'iv']
        feats_selected_info = pd.merge(df_feats_iv, self.feats_info, on='feat_id', how='inner')
        self.feats_info_add = feats_selected_info[['feat id', 'chn_name', 'eng_name', 'iv', 'feat_type']]
        self.feats_cat = list(self.feats_info_add[self.feats_info_add.feat_type == '类别型'].feat_id)
        self.feats_num = list(self.feats_info_add[self.feats_info_add.feat_type == '数值型'].feat_id)

    def bin_card(self, feats_bins, bins_woe):
        """构建分箱明细表"""
        df_feats_bins = pd.DataFrame()
        for feat, bins in feats_bins.items():
            if len(bins) == 0:
                continue
            # 类别型特征
            if feat in self.feats_cat:
                for i, bin_i in enumerate(bins):
                    item = pd.DataFrame([feat, bin_i, i, ','.join(bin_i), bins_woe[feat][i]]).T
                    df_feats_bins = df_feats_bins.append(item)
            # 数值型特征
            elif feat in self.feats_num:
                if math.isnan(bins[-1]):
                    bins_str = ','.join(map(str, [-np.inf]+bins[:-1]+[np.inf]+bins[-1:]))
                    item = pd.DataFrame([feat, 'nan', len(bins), bins_str, bins_woe[feat][len(bins)]]).T
                    df_feats_bins = df_feats_bins.append(item)
                    bins = bins[:-1]
                else:
                    bins_str = ','.join(map(str, [-np.inf] + bins + [np.inf]))
                bins_cnt = len(bins)
                for i in range(bins_cnt + 1):
                    if i == 0:
                        item = pd.DataFrame([feat, '[' + str(-np.inf) + ', ' + str(bins[i]) + ')', i,
                                             bins_str, bins_woe[feat][i]]).T
                    elif i == bins_cnt:
                        item = pd.DataFrame([feat, '[' + str(bins[i-1]) + ',' + str(np.inf) + ')', i,
                                             bins_str, bins_woe[feat][i]]).T
                    else:
                        item = pd.DataFrame([feat, '[' + str(bins[i-1]) + ',' + str(bins[i]) + ')', i,
                                             bins_str, bins_woe[feat][i]]).T
                    df_feats_bins = df_feats_bins.append(item)
        df_feats_bins.columns = ['feat_id', 'feat_bin', 'bin_index', 'feat_str', 'woe']

        return df_feats_bins


class PredictScore(object):
    """模型部署：对新数据预测评分
    
    Parameters
    ----------
    predict_data: Dataframe
        待预测客户的数据.
    model_param: Dataframe
        用于预测的模型参数.
    """
    def __init__(self, predict_data, model_param, feats_info):
        self.predict_data = predict_data
        self.model_param = model_param
        self.feats_info = feats_info

    def predict_score(self, pdo=40, base_odds=10, base_score=600):
        """预测评分

        Returns
        ---------
        data_pred: Dataframe
            客户的预测概率。
        data_score: Dataframe
            客户在不同特征上的分数。
        vars lst: list
            用于预测的模型特征列表.
        """
        # 入模特征
        feats_selected = self.model_param[['feat id']].drop_duplicates()
        feats_info = pd.merge(self.feats_info, feats_selected, on='feat_id', how='inner')
        feats_lst = list(feats_info.feat_id)
        feats_cat = list(feats_info[feats_info.feat_type == '类别型'].feat_id)
        feats_num = list(feats_info[feats_info.feat_type == '数值型'].feat_id)

        # 待预则客户
        data = self.predict_data.copy()
        for feat in feats_cat:
            data[feat].replace('', 'empty', inplace=True)
            data[feat] = data[feat].fillna('missing')

        # 映射分箱：数值型
        null_index = False
        for feat in feats_num:
            # 变量值映射到分箱
            feat_cut = self.model_param.loc[self.model_param.feat_id == feat, 'feat_str'].values[0].split(',')
            feat_cut_float = list(map(float, feat_cut))
            if math.isnan(feat_cut_float[-1]):
                feat_cut_float = feat_cut_float[:-1]
                null_index = True
            data[feat] = pd.cut(data[feat], bins=feat_cut_float, right=False, labels=False)
            if null_index:
                data[feat] = data[feat].fillna(len(feat_cut_float) - 1).astype(int)
            # 分箱获取转换theta_x
            model_info_feat = self.model_param[self.model_param.feat_id == feat]
            feat_map = dict(zip(model_info_feat.bin_index, model_info_feat.theta_x))
            data[feat] = data[feat].map(feat_map)

        # 映射分箱：类别型
        for feat in feats_cat:
            # 变量值映射到分箱
            feat_bins = self.model_param.loc[self.model_param.feat_id == feat, ['feat bin',
                                             'feat_str']].values
            feat_map = {}
            for values, keys in feat_bins:
                key_lst = keys.split(',')
                for key in key_lst:
                    feat_map[key] = values
            data[feat] = data[feat].map(feat_map)
            # 分箱获取转英theta_x
            model_info_feat = self.model_param[self.model_param.feat_id == feat]
            feat_map = dict(zip(model_info_feat.feat_bin, model_info_feat.theta_x))
            data[feat] = data[feat].map(feat_map)

        # 分箱转换结果校验：特征值缺先率小于万分之一
        self._verify_feat_missing(self.predict_data, data)

        # 特征分，仅输出数值型变量
        data_score = data.copy()
        A, B = self._get_ab(pdo, base_odds, base_score)
        for feat in feats_lst:
            data_score[feat] = B * data_score[feat]

        # 预测评分
        data['intercept'] = self.model_param.loc[self.model_param.feat_id == 'intercept', 'coef'].values[0]
        data['log_odds'] = data.sum(axis=1)
        data['p_pred'] = data.log_odds.apply(lambda wx: 1 /(1 + np.exp(-wx)))
        data['score'] = (A + B * np.log(data.p_pred / (1 - data.p_pred))).astype(int)
        data_score['p_pred'] = data['p pred']
        data_score['score'] = data['score']
        return data_score

    @staticmethod
    def _verify_feat_missing(data_org, data, missing_ratio=0.01):
        """校验分箱映射后的缺失率"""
        feat_cols = data.isnull().sum(axis=0) / data.shape[0]
        feats_missing = feat_cols[feat_cols > missing_ratio]
        if len(feats_missing) > 0:
            for feat, missing_ratio in feats_missing.items():
                missing_val = dict(data_org[feat][data[feat].isnull()].value_counts())
                print("特征{}分箱映射的缺率：{}，缺失值和缺失个数：{}.".format(feat, missing_ratio, str(missing_val)))

    @staticmethod
    def _get_ab(pdo, base_odds, base_score):
        """计算用于评分映射的参数A，B

        Parameters
        ----------
        pdo: int
            客户预测概率翻倍的加分。
        base_odds: int
            对数几率pred / (1 - pred).
        base_score: int
            某个特定违约概率下的预期评分。
        Returns
        ----------
        A: float
            基础分.
        B: float
            分数放大系数.
        """
        B = pdo / np.log(2)
        A = base_score + B * np.log(base_odds)
        return A, B

    @staticmethod
    def get_score(pred, a, b):
        """计算评分

        Parameters
        ----------
        pred: float
            预测概率.
        a: float
            基础分.
        b: float
            分数放大系数.

        Returns
        ----------
        score: float
            预测分数.
        """
        score = a - b * (np.log(pred / (1 - pred)))
        return score

