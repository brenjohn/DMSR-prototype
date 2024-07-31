#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 13:25:53 2024

@author: brennan
"""

import keras

from keras import layers

def Discriminator(discriminator_in_channels):

    # Create the discriminator.
    discriminator = keras.Sequential(
        [
            keras.layers.InputLayer((64, 64, 64, discriminator_in_channels)),
            layers.Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv3D(128, (3, 3, 3), strides=(2, 2, 2), padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv3D(128, (3, 3, 3), strides=(2, 2, 2), padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv3D(128, (3, 3, 3), strides=(2, 2, 2), padding="same"),
            layers.LeakyReLU(negative_slope=0.2),
            layers.GlobalMaxPooling3D(),
            layers.Dense(1, activation='sigmoid'),
        ],
        name="discriminator",
    )
    
    return discriminator