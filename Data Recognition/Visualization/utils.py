
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functools
from PIL import Image
import matplotlib.image as image


def resize_image(file_in, file_out, resize=1):
    """resize picture.
    
    file_in: str
        the path of origin picture.
    file_out: 
        the path of target picture.
    resize: float, optional
        change the size of origin picture.
    """
    image = Image.open(file_in)
    width, height = image.size
    width = int(width*resize)
    height = int(height*resize)
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    resized_image.save(file_out)
    
def add_mark(func):
    @functools.wraps(func)
    def wrapper(*args, mark=False, mark_dir='./img/ccx.png', mark_resize=1, mark_xo=10, mark_yo=10, mark_alpha=.7, **kwargs):
        """add watermark for figure.
        
        Parameters:
        -----------
        mark: bool, optional
            if True, add watermark for figure, the default is False.
        mark_dir: string, optional
            the path of watermark picture.
        mark_resize: float, optional
            change the size of origin picture.
        mark_xo: int, optional
            the x image offset in pixels.
        mark_yo: int, optional
            the y image offset in pixels.
        mark_alpha: float, optional
            opacity of the watermark.
        """
        func(*args, **kwargs)
        if mark:
            if mark_resize != 1:
                _, file_type =os.path.splitext(mark_dir)
                target_dir = mark_dir.replace(file_type, '_1'+file_type)
                resize_image(mark_dir, target_dir, resize=mark_resize)
            else:
                target_dir = mark_dir
            im = image.imread(target_dir)
            plt.figimage(im, xo=mark_xo, yo=mark_yo, alpha=mark_alpha, zorder=3)
    return wrapper

def set_grid(func):
    @functools.wraps(func)
    def wrapper(*args, grid_visible=True, grid_axis='y', **kwargs):
        """add watermark for figure.
        
        Parameters:
        -----------
        grid_visible: bool, optional
            if True, show grid in the figure, the default is True.
        grid_axis: string, optional
            axis displayed in figure, the value can be 'both', 'x', 'y'.
        """
        func(*args, **kwargs)
        if grid_visible:
            plt.grid(axis=grid_axis) 
    return wrapper