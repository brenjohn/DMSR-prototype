#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 18:56:23 2024

@author: brennan

This file defines functions for building the generator model for a DMSRGAN.

It also has functions for creating a latent space sampler for the generator.
"""

from tensorflow import keras
from keras.random import normal
from keras.layers import Conv3D, PReLU, UpSampling3D, Cropping3D, Concatenate


def build_generator(N, scale_factor, channels=256):
    """
    Returns a generator model to be used in a DMSRGAN. The model consists of
    a convolutional layer to increase the number of channels to the specified
    value and then and sequence of H-blocks to both upscale the data and add
    stocastic details by injecting noise. Each H-block increases the size of
    data by a factor of two with some cropping due to the convolutional layers.
    
            LR_data          noiseA/B         noiseA/B
           y|     |x            |                |
            | Convoltion        |                |
            |     |x            |                |
            H-block <-----------|                |
           y|     |x                             |
            H-block <----------------------------|
            :     :
            :     :
            H-block <---- ...
           y|     |x
            |
         Output
         
    Parameters
    ----------
    N            : The number of cells in each spatial dimension of the LR data
    scale_factor : The scale factor to enhance the LR data by. Determines the
                   number of H-blocks used in the model.
    channels     : The number of channels to use in each H-block. The number of
                   channels decreases by a factor of 2 in each subsequent 
                   H-block.
            
    """
    inputs = create_input_layers(N, scale_factor)
    
    # Create initial convolutional layer.
    x = inputs[0]
    y = Cropping3D(1, data_format='channels_first')(x)
    x = Conv3D(channels, 3, data_format='channels_first')(x)
    x = PReLU(shared_axes=(2, 3, 4))(x)
    
    # Add H-blocks.
    noise_inputs = iter(inputs[1:])
    scale = 1
    while scale < scale_factor:
        scale *= 2
        noiseA = next(noise_inputs)
        noiseB = next(noise_inputs)
        
        x, y = HBlock(x, y, noiseA, noiseB,
            in_channels = 2 * channels // scale,
            out_channels = channels // scale
        )
    
    # y = Cropping3D(2, data_format='channels_first')(y)
    generator = keras.Model(
        inputs=inputs,
        outputs=y, 
        name='dmsr_generator'
    )
    
    return generator


def create_input_layers(N, scale_factor):
    """
    Create the LR data and noise input layers for the generator.
    """
    
    # Create the input layer for the LR data to be enhanced.
    input_shape = (3, N, N, N)
    LR_input = keras.Input(shape=input_shape, name='input_layer')
    inputs = (LR_input, )
    
    # Create the input layers for the noise to be added to each H-block. 
    N -= 2
    scale = 1
    while scale < scale_factor:
        noiseA_shape = (1, N, N, N)
        noiseA = keras.Input(shape=noiseA_shape, name=f'noiseA_{scale}')
        
        scale *= 2; N *= 2; N -= 2
        
        noiseB_shape = (1, N, N, N)
        noiseB = keras.Input(shape=noiseB_shape, name=f'noiseB_{scale}')
        
        N -= 2
        inputs += (noiseA, noiseB)
        
    return inputs


def HBlock(x_p, y_p, noiseA, noiseB, in_channels, out_channels):
    """
    The "H" block of the StyleGAN2 generator.

     noiseA  noiseB           x_p                     y_p
       |       |               |                       |
       >------->---------->convolution           linear upsample
                               |                       |
                                >--- projection ------>+
                               |                       |
                               v                       v
                              x_n                     y_n

    See Fig. 7 (b) upper in https://arxiv.org/abs/1912.04958

    Parameters
    ----------
    x
    y
    noiseA       : noise for noise layer A in convolutional block
    noiseB       : noise for noise layer B in convolutional block
    in_channels  : number of channels of x_p
    out_channels : number of channels of x_n

    Notes
    -----
    x_p and y_p should have the same spatial dimensions 
    y_p and y_n always have 3 channels
    next_size = 2 * prev_size - 4
    """
    x_n = HBlock_conv(x_p, noiseA, noiseB, in_channels, out_channels)
    p = HBlock_projection(x_n)
    
    y_n = UpSampling3D(size=2, data_format='channels_first')(y_p)
    y_n = Cropping3D(2, data_format='channels_first')(y_n)
    
    return x_n, y_n + p


def HBlock_conv(x, noiseA, noiseB, in_channels, out_channels):
    """
    The convolutional block of the H-block.
    
            x     noiseA        noiseB
            |       |             |
            +---Convolution       |
            |                     |
        Upsample x2               |
            |                     |
        Convolution               |
            |                     |
            +----------------Convolution
            |
        Convolution
            |
    
    Parameters
    ----------
    x
    noiseA       : noise for noise layer A
    noiseB       : noise for noise layer B
    in_channels  : number of channels of x_p
    out_channels : number of channels of output
    
    Notes
    -----
    next_size = 2 * prev_size - 4
    noiseA_size = prev_size
    noiseB_size = 2 * prev_size - 2
    """
    
    # Add Noise A.
    noiseA = Conv3D(in_channels, 1, data_format='channels_first')(noiseA)
    # x = x + noiseA
    x = Concatenate(axis=1)([x, noiseA])
    
    # Upsample.
    x = UpSampling3D(size=2, data_format='channels_first')(x)
    x = Conv3D(out_channels, 3, data_format='channels_first')(x)
    x = PReLU(shared_axes=(2, 3, 4))(x)

    # Add Noise B.
    noiseB = Conv3D(out_channels, 1, data_format='channels_first')(noiseB)
    # x = x + noiseB
    x = Concatenate(axis=1)([x, noiseB])
    
    # Final convolution.
    x = Conv3D(out_channels, 3, data_format='channels_first')(x)
    x = PReLU(shared_axes=(2, 3, 4))(x)
    
    return x


def HBlock_projection(x):
    """
    The H-block projection operation.
    """
    x = Conv3D(3, 1, data_format='channels_first')(x)
    x = PReLU(shared_axes=(2, 3, 4))(x)
    return x


def build_latent_space_sampler(input_shapes):
    noise_shapes = [tuple(shape[1:]) for shape in input_shapes[1:]]
    
    def sampler(batch_size):
        batch_size = (batch_size,)
        samples = tuple(normal(batch_size + shape) for shape in noise_shapes)
        return samples
    
    return sampler


def generator_supervised_dataset(dataset, batch_size, sampler):
    """
    Returns a new dataset with latent space samples added to the LR data
    from the given dataset. The new dataset can be used for supervised
    training of a generator to learn to predict the HR data from the LR
    data.
    """
    def add_latent_sample(LR_fields, HR_fields):
        lantent_variables = sampler(batch_size)
        LR_fields = (LR_fields, ) + lantent_variables
        return LR_fields, HR_fields
    
    return dataset.map(add_latent_sample)