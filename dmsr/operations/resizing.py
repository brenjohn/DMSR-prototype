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
    

@tf.function
def trilinear_scaling(data, scale) -> tf.Tensor:
    """
    Returns a tensor constructed by rescaling the given data tensor using
    trilinear interpolation.
    
    Note: No anti-aliasing is used either before or after the interpolation.
    
    Arguments:
        - data   : Tensor with shape (batch_size, channels, N, N, N), where N
                   is the original grid size.
        - scale  : The scale factor to increase the spacial resolution of the
                   given data tensor by.
    """
    
    # Get number of batches, channels and the old and new sizes of the data.
    data_shape = tf.shape(data)
    batch_size = data_shape[0]
    channels   = data_shape[1]
    old_size   = data_shape[2]
    new_size   = tf.cast(tf.cast(old_size, tf.float32) * scale, tf.int32)
    
    # Create a tensor containing the new points to draw samples for.
    new_points = tf.linspace(0, old_size-1, new_size)
    new_points = tf.meshgrid(new_points, new_points, new_points, indexing='ij')
    new_points = tf.stack(new_points, axis=-1)
    
    # Get the interger points above and below the new points.
    lower_points = tf.floor(new_points)
    upper_points = lower_points + 1
    
    # For integer points outside the bounardy, we sample values on the 
    # boundary. This is like assuming the boundary values are repeated beyond
    # the boundary.
    upper_bound = tf.cast(old_size-1, tf.float64)
    upper_sample_points = tf.clip_by_value(upper_points, 0, upper_bound)
    
    # Here we build a tensor of indices for the data tensor to pull out 
    # relevant values. The indices tensor should be a 
    # (batch_size, channels, new_size, new_size, new_size, 8) tensor of indices
    # into the data tensor, corresponding to the 8 neighbours of each point in
    # the new array. The indices, ignoring batch indices are 3 dimensional so
    # the final shape of the indices tensor is
    # (batch_size, channels, new_size, new_size, new_size, 8, 3)
    lower_index = tf.cast(lower_points, tf.int32)
    upper_index = tf.cast(upper_sample_points, tf.int32)
    
    lower_index = tf.expand_dims(lower_index, 0)
    lower_index = tf.expand_dims(lower_index, 0)
    lower_index = tf.repeat(lower_index, batch_size, axis=0)
    lower_index = tf.repeat(lower_index, channels, axis=1)
    
    upper_index = tf.expand_dims(upper_index, 0)
    upper_index = tf.expand_dims(upper_index, 0)
    upper_index = tf.repeat(upper_index, batch_size, axis=0)
    upper_index = tf.repeat(upper_index, channels, axis=1)
    
    x0, y0, z0 = tf.unstack(lower_index, axis=-1)
    x1, y1, z1 = tf.unstack(upper_index, axis=-1)
    
    indices_x = tf.stack([x0, x1, x0, x1, x0, x1, x0, x1], axis=-1)
    indices_y = tf.stack([y0, y0, y1, y1, y0, y0, y1, y1], axis=-1)
    indices_z = tf.stack([z0, z0, z0, z0, z1, z1, z1, z1], axis=-1)
    indices = tf.stack([indices_x, indices_y, indices_z], axis=-1)
    
    # Draw the relevant values from the data tensor to construct the new
    # data tensor. (ie the values for the 9 neighbouring points of each new 
    # point in the new tensor)
    content = tf.gather_nd(data, indices, batch_dims=2)
    
    # Compute the weights to be used when computing the new values as a
    # weights sum of the 8 neighbouring values.
    distance_to_lower = new_points - lower_points
    distance_to_upper = upper_points - new_points
    dx0, dy0, dz0 = tf.unstack(distance_to_lower, axis=-1)
    dx1, dy1, dz1 = tf.unstack(distance_to_upper, axis=-1)
    wx = tf.stack([dx1, dx0, dx1, dx0, dx1, dx0, dx1, dx0], axis=-1)
    wy = tf.stack([dy1, dy1, dy0, dy0, dy1, dy1, dy0, dy0], axis=-1)
    wz = tf.stack([dz1, dz1, dz1, dz1, dz0, dz0, dz0, dz0], axis=-1)
    weights = wx * wy * wz
    weights = tf.cast(weights, dtype=tf.float32)
    
    # Compute the new data tensor.
    new_data = weights * content
    new_data = tf.add_n(tf.split(new_data, 8, -1))
    new_data = tf.squeeze(new_data, axis=-1)
    return new_data