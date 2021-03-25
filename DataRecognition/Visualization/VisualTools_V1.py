
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from utils import resize_image,add_mark,set_grid

sns.palplot(sns.color_palette('deep'))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.edgecolor'] = '#080808'


@set_grid
@add_mark
def plot_dist(x, xlabel='', ylabel='', title='', labels=None, figsize=(9,6), **kwargs):
    """Flexibly plot a univariate distribution of observations.
    
    Parameters:
    -----------
    x: Series, 1d-array, 2d-array or DataFrame
        Observed data.
    xlabel: string, optional
        x label for the relevent component of the plot.
    ylabel: string, optional
        y label for the relevent component of the plot.
    title: string, optional
        title for the figure.
    labels: list, optional
        the name of legend for the figure.
    figsize: tuple, optional
        size of the figure.
    kwargs: key, value pairings
        Other keyword arguments are passed to sns.distplot, more parameters available.
    """
    if not isinstance(x, (np.ndarray, pd.Series, pd.DataFrame)):
        raise Exception("Expected data type is ndarray, Series or DataFrame, but get {}.".format(type(x)))
    
    if x.ndim not in (1, 2):
        raise Exception("Expected dim of y in 1 or 2, but get dim = {}, input dim error.".format(x.ndim))
    
    fig = plt.figure(figsize=figsize)
    
    if isinstance(x, pd.DataFrame):
        labels = x.columns
        x = x.T.values   
    
    if x.ndim == 1:
        sns.distplot(x, **kwargs)
    else:
        for i in range(x.shape[0]):
            if labels is None:
                sns.distplot(x[i], **kwargs)
            else:
                sns.distplot(x[i], label=labels[i], **kwargs)
                plt.legend()
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

@set_grid
@add_mark
def plot_kde(x, xlabel='', ylabel='', title='', labels=None, figsize=(9,6), **kwargs):
    """Fit and plot a univariate or bivariate kernel density estimate.
    
    Parameters:
    -----------
    x: Series, 1d-array, 2d-array or DataFrame
        Observed data.
    xlabel: string, optional
        x label for the relevent component of the plot.
    ylabel: string, optional
        y label for the relevent component of the plot.
    title: string, optional
        title for the figure.
    labels: list, optional
        the name of legend for the figure.
    figsize: tuple, optional
        size of the figure.
    kwargs: key, value pairings
        Other keyword arguments are passed to sns.kdeplot, more parameters available.
    """
    if not isinstance(x, (np.ndarray, pd.Series, pd.DataFrame)):
        raise Exception("Expected data type is ndarray, Series or DataFrame, but get {}.".format(type(x)))
    
    if x.ndim not in (1, 2):
        raise Exception("Expected dim of y in 1 or 2, but get dim = {}, input dim error.".format(x.ndim))
    
    fig = plt.figure(figsize=figsize)
    
    if isinstance(x, pd.DataFrame):
        labels = x.columns
        x = x.T.values   
    
    if x.ndim == 1:
        sns.kdeplot(x, **kwargs)
    else:
        for i in range(x.shape[0]):
            if labels is None:
                sns.kdeplot(x[i], **kwargs)
            else:
                sns.kdeplot(x[i], label=labels[i], **kwargs)
                plt.legend()
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

@add_mark
def plot_box(xlabel='', title='', figsize=(9,6), **kwargs):
    """Draw a box plot to show distributions with respect to categories.
    
    Parameters:
    -----------
    x: Series, 1d-array, 2d-array or DataFrame
        Observed data.
    xlabel: string, optional
        x label for the relevent component of the plot.
    ylabel: string, optional
        y label for the relevent component of the plot.
    title: string, optional
        title for the figure.
    figsize: tuple, optional
        size of the figure.
    kwargs: key, value pairings
        Other keyword arguments are passed to sns.boxplot, more parameters available.
    """
    fig = plt.figure(figsize=figsize)
    sns.boxplot(**kwargs)
    if isinstance(kwargs['x'], np.ndarray):
        plt.xlabel(xlabel)
    plt.title(title)

@add_mark
def plot_scatter(title='', figsize=(9,6), **kwargs):
    """Draw a scatter plot with possibility of several semantic groupings.
    
    Parameters:
    -----------
    title: string, optional
        title for the figure.
    figsize: tuple, optional
        size of the figure.
    kwargs: key, value pairings
        Other keyword arguments are passed to sns.scatterplot, more parameters available.
    """
    fig = plt.figure(figsize=figsize)
    sns.scatterplot(**kwargs)
    plt.title(title)

@add_mark
def plot_heatmap(title='', figsize=(9,6), **kwargs):
    """Plot rectangular data as a color-encoded matrix.
    
    Parameters:
    -----------
    title: string, optional
        title for the figure.
    figsize: tuple, optional
        size of the figure.
    kwargs: key, value pairings
        Other keyword arguments are passed to sns.scatterplot, more parameters available.
    """
    fig = plt.figure(figsize=figsize)
    sns.heatmap(**kwargs)
    plt.title(title)

@set_grid
@add_mark    
def plot_line(x, y, xlabel='', ylabel='', title='', labels=None, figsize=(9,6), **kwargs):
    """Draw a line plot.
    
    parameter:
    ----------
    x: Series, 1d-array
        Observed data.
    y: Series, 1d-array, 2d-array
        Observed data.
    xlabel: string, optional
        x label for the relevent component of the plot.
    ylabel: string, optional
        y label for the relevent component of the plot.
    title: string, optional
        title for the figure.
    labels: list, optional
        the name of legend for the figure.
    figsize: tuple, optional
        size of the figure.
    """
    if not isinstance(x, np.ndarray):
        raise Exception("Expected x type is ndarray, but get {}.".format(type(x)))
    
    if not isinstance(y, (np.ndarray)):
        raise Exception("Expected y type is ndarray, but get {}.".format(type(y)))
    
    if x.ndim != 1:
        raise Exception("Expected dim of x is 1, but get dim = {}, input dim error.".format(x.ndim))
    
    if y.ndim not in (1, 2):
        raise Exception("Expected dim of y in 1 or 2, but get dim = {}, input dim error.".format(y.ndim))
    
    fig = plt.figure(figsize=figsize)
    
    if y.ndim == 1:
        plt.plot(x, y, **kwargs)
    else:
        for i in range(y.shape[0]):
            if labels is None:
                plt.plot(x, y[i], **kwargs)
            else:
                plt.plot(x, y[i], label=labels[i], **kwargs)
                plt.legend()
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

@add_mark    
def plot_line_twinx(x, y1, y2, xlabel='', y1_label='', y2_label='', title='', y1_legend=None, y2_legend=None, rot=0, figsize=(9,6), **kwargs):
    """Draw a line plot with double coordinate.
    
    parameter:
    ----------
    x: Series, 1d-array
        Observed data.
    y1: Series, 1d-array, 2d-array
        Observed data of axis y1.
    y2: Series, 1d-array, 2d-array
        Observed data of axis y2.
    xlabel: string, optional
        x label for the relevent component of the plot.
    y1_label: string, optional
        y1 label for the relevent component of the plot.
    y2_label: string, optional
        y2 label for the relevent component of the plot.
    title: string, optional
        title for the figure.
    y1_legend: list, optional
        the name of legend for y1.
    y2_legend: list, optional
        the name of legend for y2.
    rot: int
        the angle of rotate x lable.
    figsize: tuple, optional
        size of the figure.
    """
    if not isinstance(x, np.ndarray):
        raise Exception("Expected x type is ndarray, but get {}.".format(type(x)))
    
    if not isinstance(y1, (np.ndarray))&isinstance(y2, (np.ndarray)):
        raise Exception("Expected y type is ndarray, but get {} for y1, {} for y2.".format(type(y1), type(y2)))
    
    if x.ndim != 1:
        raise Exception("Expected dim of x is 1, but get dim = {}, input dim error.".format(x.ndim))
    
    if y1.ndim not in (1, 2) or y2.ndim not in (1, 2):
        raise Exception("Expected dim of y in 1 or 2, but get y1 dim = {}, y2 dim = {} input dim error.".format(y1.ndim, y2.ndim))
    
    y1 = y1.reshape(1,-1) if y1.ndim == 1 else y1      
    y2 = y2.reshape(1,-1) if y2.ndim == 1 else y2
        
    if y1_legend is not None and len(y1_legend) != y1.shape[0]:
        raise Exception("the dim of y1 is {}, not match the legend {} of y1.".format(y1.shape[0], len(y1_legend)))
    
    if y2_legend is not None and len(y2_legend) != y2.shape[0]:
        raise Exception("the dim of y2 is {}, not match the legend {} of y2.".format(y2.shape[0], len(y2_legend)))
    
    fig = plt.figure( figsize=figsize)
    ax1 = fig.add_subplot(1,1,1)
    
    for i in range(y1.shape[0]):
        if y1_legend is None:
            ax1.plot(y1[i], linewidth=2, linestyle='-', **kwargs)
        else:
            ax1.plot(y1[i], linewidth=2, linestyle='-', label=y1_legend[i], **kwargs)
            ax1.legend(loc='upper left', frameon=True)

    ax1.set_xlim(-1, len(x)+1)
    ax1.set_ylim(y1[~np.isnan(y1)].min()*0.8, y1[~np.isnan(y1)].max()*1.2)

    x_tick = range(0, len(x), 40)
    ax1.set_xticks(x_tick)
    ax1.set_xticklabels(x, rotation=rot)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1_label)

    # Add a second axes
    ax2 = ax1.twinx()
    for i in range(y2.shape[0]):
        if y2_legend is None:
            ax2.plot(y2[i], linewidth=2, linestyle='--', alpha=0.8, **kwargs)
        else:
            ax2.plot(y2[i], linewidth=2, linestyle='--', label=y2_legend[i], alpha=0.8, **kwargs)
            ax2.legend(loc='upper right', frameon=True)
    
    ax2.set_ylim(y2[~np.isnan(y2)].min()*0.8, y2[~np.isnan(y2)].max()*1.2)
    ax2.set_ylabel(y2_label)
    ax2.spines['right'].set_visible(True)
    ax2.set_title(title)

@add_mark    
def plot_clustermap(X, y, method='t-SNE', title='', figsize=(9, 6), load_img=False):
    """Plot a matrix dataset as a t-SNE for the visualization of high-dimensional datasets.
    
    parameter:
    ----------
    x: 2d-array
        Observed data.
    y: 1d-array
        clustered target.
    method：string, optional
        if method='t-SNE', dimension reduction by t-SNE, the default is t-SNE;
        if method='PCA', dimension reduction by PCA.
    title: string, optional
        title for the figure.
    figsize: tuple, optional
        size of the figure.
    load_img：Bool, optional
        if True, load image into current path, the default is False.
    """
    if not isinstance(X, np.ndarray):
        raise Exception("Expected X type is ndarray, but get {}.".format(type(X)))
    
    if not isinstance(y, np.ndarray):
        raise Exception("Expected y type is ndarray, but get {}.".format(type(y)))
    
    if method == 't-SNE':
        X_dim = TSNE(n_components=2, random_state=2020).fit_transform(X)
    elif method == 'PCA':
        X_dim = PCA(n_components=2).fit_transform(X)
    else:
        raise Exception("Expected method is t-SNE or PCA, but get method: {}.".format(method))
    
    plt.figure(figsize=figsize)
    plt.scatter(X_dim[:, 0], X_dim[:, 1], c=y, label=method)
    plt.legend()
    plt.title(title)
    
    if load_img:
        img_dir = "img"
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        plt.savefig('img/digits_{}.png'.format(method), dpi=120)

@set_grid
@add_mark
def plot_bar(x, y, xlabel='', ylabel='', title='', figsize=(9,6), hlines=None):
    """Make a vertical bar plot.
    
    parameter:
    ----------
    x: Series, 1d-array
        categorical levels.
    y: Series, 1d-array
        values for each category.
    xlabel: string, optional
        x label for the relevent component of the plot.
    y_label: string, optional
        y1 label for the relevent component of the plot.
    title: string, optional
        title for the figure.
    figsize: tuple, optional
        size of the figure.
    hlines: float, optional
        horizontal line, if mean is wanted, set this parameter.
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        raise Exception("Expected data type of x is ndarray or Series, but get {}.".format(type(x)))
    
    if not isinstance(y, (np.ndarray, pd.Series)):
        raise Exception("Expected data type of y is ndarray or Series, but get {}.".format(type(y)))
    
    if x.ndim != 1 or y.ndim != 1:
        raise Exception("Expected dim of x and y must be 1, but get dim: {} for x, {} for y, input dim error.".format(x.ndim, y.ndim))
    
    fig = plt.figure(figsize=figsize)
    sns.barplot(x, y)
    
    if hlines:
        plt.hlines(hlines, 0, len(x)-1, colors='gray', linewidth=2, linestyles='--')
    
    for x, y in enumerate(y):
        plt.text(x, y, '%s' % y, fontsize=15, horizontalalignment='center')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

@set_grid
@add_mark
def plot_barh(x, y, xlabel='', ylabel='', title='', xlim=None, figsize=(9,6), vlines=None):
    """Make a horizontal bar plot.
    
    parameter:
    ----------
    x: Series, 1d-array
        values for each category.
    y: Series, 1d-array
        categorical levels.
    xlabel: string, optional
        x label for the relevent component of the plot.
    y_label: string, optional
        y1 label for the relevent component of the plot.
    title: string, optional
        title for the figure.
    figsize: tuple, optional
        size of the figure.
    vlines: float, optional
        vertical line, if mean is wanted, set this parameter.
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        raise Exception("Expected data type of x is ndarray or Series, but get {}.".format(type(x)))

    if not isinstance(y, (np.ndarray, pd.Series)):
        raise Exception("Expected data type of y is ndarray or Series, but get {}.".format(type(y)))
    
    if x.ndim != 1 or y.ndim != 1:
        raise Exception("Expected dim of x and y must be 1, but get dim: {} for x, {} for y, input dim error.".format(x.ndim, y.ndim))
    
    fig = plt.figure(figsize=figsize)
    sns.barplot(x, y)

    if vlines:
        plt.vlines(vlines, 0, len(y)-1, colors='gray', linewidth=2, linestyles='--')
        
    for i, values in enumerate(x):
        plt.text(values, i, '%s' % values, fontsize=15, verticalalignment='center')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

@set_grid
@add_mark
def plot_multi_bar(x, y, cates, xlabel='', ylabel='', title='', figsize=(9,6)):
    """Make a multiple bar plot.
    
    parameter:
    ----------
    x: Series, 1d-array
        categorical levels.
    y: Series, 1d-array
        values for each category.
    cates: Series, 1d-array
        categorical levels.
    xlabel: string, optional
        x label for the relevent component of the plot.
    y_label: string, optional
        y1 label for the relevent component of the plot.
    title: string, optional
        title for the figure.
    figsize: tuple, optional
        size of the figure.
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        raise Exception("Expected data type of x is ndarray or Series, but get {}.".format(type(x)))
    
    if not isinstance(y, (np.ndarray, pd.Series)):
        raise Exception("Expected data type of y is ndarray or Series, but get {}.".format(type(y)))
    
    if not isinstance(cates, (np.ndarray, pd.Series)):
        raise Exception("Expected data type of y is ndarray or Series, but get {}.".format(type(cates)))
    
    if x.ndim != 1 or y.ndim != 1 or cates.ndim != 1:
        raise Exception("Expected dim of x/y/cates must be 1, but get dim: {} for x, {} for y, {} for cates, input dim error.".format(x.ndim, y.ndim, cates.ndim))
    
    fig = plt.figure(figsize=figsize)
    xx = np.arange(len(set(x)))
    
    sns.barplot(x=x, y=y, hue=cates)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)    

@add_mark
def plot_pie(x, y, xlabel='', figsize=(8,8)):
    """Make a pie plot.
    
    parameter:
    ----------
    x: 1d-array
        categorical levels.
    y: 1d-array
        values for each category.
    xlabel: string, optional
        x label for the relevent component of the plot.
    figsize: tuple, optional
        size of the figure.
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        raise Exception("Expected data type of x is ndarray or Series, but get {}.".format(type(x)))

    if not isinstance(y, (np.ndarray, pd.Series)):
        raise Exception("Expected data type of y is ndarray or Series, but get {}.".format(type(y)))
    
    if x.ndim != 1 or y.ndim != 1:
        raise Exception("Expected dim of x and y must be 1, but get dim: {} for x, {} for y, input dim error.".format(x.ndim, y.ndim))
    
    fig = plt.figure(figsize=figsize)
    idx = np.argsort(y)
    patches, l_text, p_text = plt.pie(y[idx], labels=x[idx], autopct='%1.2f%%', startangle=90)
    
    for t in l_text:
        t.set_size(16)
    for t in p_text:
        t.set_size(14)
    
    plt.xlabel(xlabel)
    plt.legend()