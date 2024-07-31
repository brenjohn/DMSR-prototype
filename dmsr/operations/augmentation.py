#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:05:57 2024

@author: brennan

This file defines functions for randomly flipping and permuting the dimensions
of a tensor containing particle data.

The random flips and permutations compose to generate a random element of the
octahedral group (the symmetry group of a 3D cube). This group has 48 elements.
"""

import tensorflow as tf


@tf.function
def random_transformation(LR_field, HR_field):
    """
    Randomly selects a composition of axis flips and permutations and applies
    the transformation to both the given LR and HR fields.
    """
    random_flip, permutation = random_flip_permutation()
    
    LR_field = flip(LR_field, random_flip)
    HR_field = flip(HR_field, random_flip)
    
    LR_field = permute(LR_field, permutation)
    HR_field = permute(HR_field, permutation)

    return LR_field, HR_field


@tf.function
def random_flip_permutation():
    """
    Generates a triplet of random boolean values and a random permutation.
    
    The boolean values represent a composition of axis flips and the 
    permutation represent a permutation of x, y, z dimensions.
    """
    flip = tf.random.uniform((3, ))
    flip = tf.math.greater(flip, 0.5)
    permutation = tf.reshape(tf.random.shuffle([0, 1, 2]), [-1])
    return flip, permutation


@tf.function
def flip(tensor, flip_axes):
    """
    Flips the given tensor along the specified axes.
    """
    flip_x, flip_y, flip_z = tf.split(flip_axes, 3, axis=0)
    x, y, z = tf.split(tensor, 3, axis=0)
    tensor = tf.concat(
        (-1 * x if flip_x else x, 
         -1 * y if flip_y else y, 
         -1 * z if flip_z else z), 
        axis=0
    )
    
    flip_axes = tf.reshape(tf.where(flip_axes) + 1, [-1])
    tensor = tf.reverse(tensor, flip_axes)
    
    return tensor


@tf.function
def permute(tensor, permutation):
    """
    Permutes the dimensions of the given tensor.
    """
    shape = tensor.shape
    
    tensor = tf.gather(tensor, permutation, axis=0)
    permutation = tf.concat(([0], permutation + 1), axis=0)
    
    tensor = tf.transpose(tensor, permutation)
    return tf.ensure_shape(tensor, shape)