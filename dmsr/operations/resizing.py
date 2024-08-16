#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 18:27:40 2024

@author: brennan

This file defines functions for resizing tensors.
"""

import numpy as np
import tensorflow as tf


def scale_up_data(data, scale, batch=False):
    """
    Increases the size of the given data tensor by the given scale factor by
    repeating values in each direction.
    """
    data = tf.repeat(data, scale, axis=2)
    data = tf.repeat(data, scale, axis=3)
    data = tf.repeat(data, scale, axis=4)
    return data


def cut_field(field, cut_size, grid_size, step=0, pad=0):
    """
    
    """
    cuts = []
    if not step:
        step = cut_size
    
    for i in range(0, grid_size, step):
        for j in range(0, grid_size, step):
            for k in range(0, grid_size, step):
                
                slice_x = [n % grid_size for n in range(i-pad, i+cut_size+pad)]
                slice_y = [n % grid_size for n in range(j-pad, j+cut_size+pad)]
                slice_z = [n % grid_size for n in range(k-pad, k+cut_size+pad)]
                
                patch = np.take(field, slice_x, axis=2)
                patch = np.take(patch, slice_y, axis=3)
                patch = np.take(patch, slice_z, axis=4)
                
                cuts.append(patch)
    
    return np.concatenate(cuts)


@tf.function
def crop_to_match(large_tensor, small_tensor):
    """
    Crops the larger tensor to match the size of the smalled tensor.
    """
    large_shape = tf.shape(large_tensor)
    small_shape = tf.shape(small_tensor)
    offsets = (large_shape - small_shape) // 2
    offset = offsets[-1]
    return crop_edge(large_tensor, offset)


@tf.function
def crop_edge(tensor, size):
    """
    Crop the spacial dimensions of the given tensor by the given amount.
    """
    return tensor[:, :, size:-size, size:-size, size:-size]
