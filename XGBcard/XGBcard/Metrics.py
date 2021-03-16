# -*- coding: utf-8 -*-

"""
Created at May. 06 2020

@author: ZhangXiaoqiang

@infos: metrics模块，用于评估和监控模型效果

==============================================================
"""

import json
import functools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pyecharts.charts.base import Base


def plot_roc_curve(labels, preds, co_labels=None, co_preds=None, figsize=(6, 4)):
    """plot roc curve.
    
    Parameters
    ----------
    labels: array, shape = [n_samples]
        True binary labels.
    preds: array, shape = [n_samples]
        Target scores, probability estimates of the positive class.
    co_labels: array, shape = [m_samples]
        True binary labels of evaluation data.
    co_preds: array, shape = [m_samples]
        Target scores of evaluation data, probability estimates of the positive class.
    figsize: tuple
        The figure size of roc curve.
    """
    fpr_xgb, tpr_xgb, _ = roc_curve(labels, preds)  
    auc_ = auc(fpr_xgb, tpr_xgb)
    if co_preds is not None and co_labels is not None:
        co_fpr_xgb, co_tpr_xgb, _ = roc_curve(co_labels, co_preds)  
        co_auc_ = auc(co_fpr_xgb, co_tpr_xgb)

    # plot roc curve
    fig = plt.figure(figsize=figsize)
    if co_preds is not None and co_labels is not None:
        plt.plot(fpr_xgb, tpr_xgb, label = 'train')  
        plt.plot(co_fpr_xgb, co_tpr_xgb, label = 'eval')  
    else:
        plt.plot(fpr_xgb, tpr_xgb)  
    plt.plot([0,1],[0,1],'k--')  
    plt.xlabel('False positive rate')  
    plt.ylabel('True positive rate')  
    if co_preds is not None and co_labels is not None:
        plt.title('ROC Curve \n (Train AUC: %.3f    Test AUC: %.3f)' % (auc_, co_auc_))  
        plt.legend(loc = 'best')  
    else:
        plt.title('ROC Curve \n (AUC: %.3f)' % (auc_))
    plt.show()  
    
def plot_ks_curve(labels, preds, bins=10, figsize=(6,4), title='KS Curve'):
    """Calculate ks value and plot ks curve.
    
    Parameters
    ----------
    labels: array, shape = [n_samples]
        True binary labels.
    preds: array, shape = [n_samples]
        Probability estimates of the positive class.
    bins: int
        Divide the input data set into several parts for KS calculation.
    figsize: tuple
        The figure size of roc curve.
    """
    def n0(x):
        return sum(x == 0)

    def n1(x):
        return sum(x == 1)
    
    pred = preds  
    bad = labels 
    n = bins
    data = pd.DataFrame({'bad': bad, 'pred': pred})
    
    df_ks = data.sort_values(by='pred', ascending=False).reset_index(drop=True) \
        .assign(group=lambda x: np.ceil((x.index + 1) / (len(x.index) / n))) \
        .groupby('group')['bad'].agg([n0, n1]) \
        .reset_index().rename(columns={'n0': 'good', 'n1': 'bad'}) \
        .assign(group=lambda x: (x.index + 1) / len(x.index),
                cumgood=lambda x: np.cumsum(x.good) / sum(x.good),
                cumbad=lambda x: np.cumsum(x.bad) / sum(x.bad)
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
    
    plt.text(seri_ks['group'], max(df_ks['ks']), 'KS = %0.3f' % max(df_ks['ks']))
    plt.legend(handles=[l1, l2, l3, l4], labels=['ks-curve', 'fpr-curve', 'tpr-curve'], loc='upper left')
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title(title)
    ax.set_xlabel('population ratio')
    ax.set_ylabel('total Good/Bad ratio')

def plot_divergence(labels, preds, plot_type='kde', figsize=(6,4)):
    """plot divergence of Good/Bad.
    
    Parameters
    -----------
    labels: array, shape = [n_samples]
        True binary labels.
    preds: array, shape = [n_samples]
        Target scores, probability estimates of the positive class.
    plot_type: string, value = ['kde', 'dist']
        The type of plotting univariate distributions.
    figsize: tuple
        The figure size of roc curve.
    """
    pred_0 = preds[labels.values==0]
    pred_1 = preds[labels.values==1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if plot_type=='dist':
        sns.distplot(pred_0, label='Good')
        sns.distplot(pred_1, label='Bad')
    elif plot_type=='kde':
        sns.kdeplot(pred_0, label='Good')
        sns.kdeplot(pred_1, label='Bad')
      
    ax.set_title('divergence of Good/Bad')
    ax.set_xlabel('probability estimates of the positive')
    ax.set_ylabel('probability density')
    plt.legend(loc = 'best')  

def report_old(labels, preds, bins=20, bin_type='cut'):
    """Get model report.
    
    Parameters
    ----------
    labels: array, shape = [n_samples]
        True binary labels.
    preds: array, shape = [n_samples]
        Target scores, probability estimates of the positive class.
    bins: int
        Divide the input data set into several parts for statistics of prediction results.
        
    Returns
    -------
    dct_report: dataframe
        model report.
    """
    row_num, col_num = 0, 0  
    Y_predict = preds
    Y = labels
    nrows = Y.shape[0]  
    lis = [(Y_predict[i], Y[i]) for i in range(nrows)]  
    ks_lis = sorted(lis, key=lambda x: x[0], reverse=True)  
    bin_num = int(nrows/bins+1)  
    bad = sum([1 for (p, y) in ks_lis if y > 0.5])  
    good = sum([1 for (p, y) in ks_lis if y <= 0.5])  
    bad_cnt, good_cnt = 0, 0  
    KS = []  
    BAD = []  
    GOOD = []  
    BAD_CNT = []  
    GOOD_CNT = []  
    BAD_PCTG = []  
    BADRATE = []  
    dct_report = {}  
    for j in range(bins):  
        ds = ks_lis[j*bin_num: min((j+1)*bin_num, nrows)]  
        bad1 = sum([1 for (p, y) in ds if y > 0.5])  
        good1 = sum([1 for (p, y) in ds if y <= 0.5])  
        bad_cnt += bad1  
        good_cnt += good1  
        bad_pctg = round(bad_cnt/sum(labels),3)  
        badrate = round(bad1/(bad1+good1),3)  
        ks = round(np.abs((bad_cnt / bad) - (good_cnt / good)),3)  
        KS.append(ks)  
        BAD.append(bad1)  
        GOOD.append(good1)  
        BAD_CNT.append(bad_cnt)  
        GOOD_CNT.append(good_cnt)  
        BAD_PCTG.append(bad_pctg)  
        BADRATE.append(badrate)  
        dct_report['KS'] = KS  
        dct_report['BAD'] = BAD  
        dct_report['GOOD'] = GOOD  
        dct_report['BAD_CNT'] = BAD_CNT  
        dct_report['GOOD_CNT'] = GOOD_CNT  
        dct_report['BAD_PCTG'] = BAD_PCTG  
        dct_report['BADRATE'] = BADRATE  
    dct_report = pd.DataFrame(dct_report)  
    return dct_report

def report(labels, preds, bins=10, bin_type='cut'):
    """Get model report.
    
    Parameters
    ----------
    labels: array, shape = [n_samples]
        True binary labels.
    preds: array, shape = [n_samples]
        Target scores, probability estimates of the positive class.
    bins: int
        Divide the input data set into several parts for statistics of prediction results.
    bin_type: string
        The way the sample is divided.
        
    Returns
    -------
    dct_report: dataframe
        model report.
    """
    def n0(x):
        return sum(x == 0)

    def n1(x):
        return sum(x == 1)

    data = pd.DataFrame({'bad': labels, 'pred': preds})
    if bin_type == 'cut':
        bins_cut = functools.partial(pd.cut, precision=2)
    elif bin_type == 'qcut':
        bins_cut = functools.partial(pd.qcut, precision=2)

    df_report = data.sort_values(by='pred', ascending=False).reset_index(drop=True) \
        .assign(bins=lambda x: bins_cut(x.pred, bins)) \
        .groupby('bins')['bad'].agg(['count', n0, n1]) \
        .reset_index().rename(columns={'n0': 'good', 'n1': 'bad', 'count': 'bin_cnt'}) \
        .assign(good_cnt=lambda x: np.cumsum(x.good),
                bad_cnt=lambda x: np.cumsum(x.bad),
                good_pctg=lambda x: np.round(np.cumsum(x.good) / sum(x.good), 2),
                bad_pctg=lambda x: np.round(np.cumsum(x.bad) / sum(x.bad), 2),
                bad_rate=lambda x: np.round(x.bad / x['bin_cnt'], 3)
        ).assign(ks=lambda x: abs(x.bad_pctg - x.good_pctg))
    
    return df_report

def plot_chart(data, legend='', plot_lst=[], plot_type=[]):
    """plot the performance of the model.
    
    plot_chart supports the following graphics:
    - ks and badrate figure for different bins;
    - data distribution and badrate figure for different bins;
    - the predicted and actual values of badrate;
    
    Parameters
    ----------
    data: DataFrame
       The data to be plotted. 
    legend: string
       The name of legend.
    plot_lst: list
       The name list of two figures.
    plot_type: list
       The type list of figures to be plotted.
        
    Returns
    -------
    chart.render_notebook(): HTML
       The chart to be plotted.
    """
    xaxis_data = list(data[legend].values.astype(str))
    
    if plot_type[0] == 'bar':
        yaxis0_data = [int(i) for i in list(data[plot_lst[0]].values)]
    else:
        yaxis0_data = list(data[plot_lst[0]].values)
    
    if plot_type[1] == 'bar':
        yaxis1_data = [int(i) for i in list(data[plot_lst[1]].values)]
    else:
        yaxis1_data = list(data[plot_lst[1]].values)
    
    option = {'tooltip': {'trigger': 'axis',
                          'axisPointer': {'type': 'cross',
                                          'crossStyle': {'color': '#999'}
                                          }
                         },
              'toolbox': {'feature': {'dataView': {'show': True, 'readOnly': False},
                                      'magicType': {'show': True, 'type': plot_type},
                                      'restore': {'show': True},
                                      'saveAsImage': {'show': True}
                                      }
                         },
              'legend': {'data': plot_lst},
              'xAxis': [{'type': 'category',
                         'data': xaxis_data,
                         'axisPointer': {'type': 'shadow'}
                        }],
              'yAxis': [{'name': plot_lst[0],},
                        {'name': plot_lst[1],}
                        ],
              'series': [{'name': plot_lst[0],
                          'type': plot_type[0],
                          'data': yaxis0_data,
                         },
                         {'name': plot_lst[1],
                          'type': plot_type[1],
                          'yAxisIndex': 1,
                          'data': yaxis1_data}
                        ]
                } 
    
    chart = Base()
    chart.options = option
    chart.render(path='./img/{}_{}.html'.format(*plot_lst))
    
    return chart.render_notebook()