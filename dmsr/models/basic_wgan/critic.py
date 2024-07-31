#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 19:14:25 2024

@author: brennan
"""

from tensorflow.keras import layers, models

def Critic(critic_in_channels):
    
    inputs = layers.Input(shape=(64, 64, 64, critic_in_channels))
    
    # Initial convolutional layer
    x = layers.Conv3D(64, 7, padding='same')(inputs)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 128)
    
    # Global average pooling and output layer
    x = layers.GlobalMaxPooling3D()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model(inputs, outputs)



def residual_block(x, filters, kernel_size=3):
    xi = x

    # First convolutional layer
    x = layers.Conv3D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    # Second convolutional layer
    x = layers.Conv3D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)

    # Add the shortcut (input) to the output
    xi = layers.Conv3D(filters, 1, strides=1, padding='same')(xi)

    x = layers.add((x, xi))
    
    # Downscale with interpolation
    xi = layers.Conv3D(filters, 1, strides=2, padding='same')(xi)

    return x