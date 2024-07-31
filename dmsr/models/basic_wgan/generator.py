#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 13:22:53 2024

@author: brennan
"""

import keras

from keras import layers

def Generator(generator_in_channels):

    # Create the generator.
    generator = keras.Sequential(
        [
            keras.layers.InputLayer((32, 32, 32, generator_in_channels)),
            
            layers.Conv3D(128, (7, 7, 7), padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv3D(128, (7, 7, 7), padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            
            layers.Conv3DTranspose(128, (4, 4, 4), strides=(2, 2, 2), padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            
            layers.Conv3D(3, (7, 7, 7), padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
        ],
        name="generator",
    )
    
    return generator