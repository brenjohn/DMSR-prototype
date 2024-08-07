#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:50:27 2024

@author: brennan
"""

import tensorflow as tf

from tensorflow import keras
from keras.random import normal
from keras.layers import Input, Conv3D, PReLU, Cropping3D, Flatten
# from keras.initializers import HeNormal


def build_critic(generator, channels=16):
    """
    Returns a critic model to be used in a DMSRGAN. The model consists of
    a number of residual blocks to downsample the data and a final 
    convolutional layer to reduce the output to a single number.
    
            (SR_data, SR_density, LR_data, LR_density)
                                |
                          Residual Block
                                |
                          Residual Block
                                :
                                :
                          Convolutional
                                |
                          Classification
           
         
    Parameters
    ----------
    generator    : The generator model to create a critic model for. The output
                   shape is used to determine the number of residual blocks and
                   convolutional layers to use.
    channels     : The number of channels to use in each layer.
    """
    
    # 8 channels = 1 LR density + 3 LR position + 1 HR density + 3 HR position
    critic_input_shape = (8,)
    critic_input_shape += generator.output.shape[2:]
    critic_inputs = Input(shape=critic_input_shape, name='critic_input_layer')
    
    num_cells = critic_input_shape[-1]
    
    # kwargs = {
    #     'data_format'        : 'channels_first',
    #     'kernel_initializer' : HeNormal()
    # }
    
    # Initial convolutional layers.
    x = Conv3D(channels, 3, data_format='channels_first')(critic_inputs)
    x = PReLU(shared_axes=(2, 3, 4))(x)
    num_cells -= 2
    channels *= 2
    
    # Add residual blocks to downsample the data.
    while num_cells > 5:
        if num_cells % 2 == 0:
            num_cells = (num_cells - 4) // 2
            channels *= 2
            x = residual_block(x, channels)
            
        else:
            num_cells -= 1
            x = Conv3D(channels, 2, data_format='channels_first')(x)
            x = PReLU(shared_axes=(2, 3, 4))(x)
      
    # Final layer to output a sigle number.
    output = Conv3D(
        1, 
        num_cells,
        activation='sigmoid',
        data_format='channels_first'
    )(x)
    output = Flatten(data_format='channels_first')(output)
    
    critic = keras.Model(
        inputs=critic_inputs, 
        outputs=output, 
        name='dmsr_critic'
    )
    
    return critic
    

def residual_block(x, channels):
    """
    Adds a residual block to x
    
                 x
                 |------------>|
                 |             |
            Convolution   Convolution
                 |             |
            Convolution        |
                 |             |
                 + <-----------|
                 |
             Downsample
                 |
                 x
                  
    next_size = (prev_size - 4) / 2
    """
    
    # Skip connection.
    y = Conv3D(channels, 1, data_format='channels_first')(x)
    y = Cropping3D(2, data_format='channels_first')(y)
    
    # Convolutional block.
    x = Conv3D(channels, 3, data_format='channels_first')(x)
    x = PReLU(shared_axes=(2, 3, 4))(x)
    x = Conv3D(channels, 3, data_format='channels_first')(x)
    x = PReLU(shared_axes=(2, 3, 4))(x)
    
    # Downsample.
    x = x + y
    x = Conv3D(channels, 2, strides=(2, 2, 2), data_format='channels_first')(x)
    
    return x


#%%
class CriticNoiseSampler:
    
    def __init__(self, critic=None, initial_std=0, epochs=1):
        if critic:
            self.shape = critic.input.shape[2:]
        self.std_decrement = initial_std / epochs
        self.current_std = tf.Variable(initial_std, dtype=tf.float32)
    
    
    def __call__(self, batch_size):
        shape = (batch_size, 3) + self.shape
        noise = self.current_std * normal(shape)
        return noise
    
    
    def update(self):
        self.current_std.assign_sub(self.std_decrement)
        self.current_std.assign(tf.maximum(self.current_std, 0.0))
        
        
    def get_config(self):
        config = {
            'shape'         : self.shape,
            'std_decrement' : self.std_decrement,
            'current_std'   : self.current_std.numpy()
        }
        return config
    
    @classmethod
    def from_config(cls, config):
        noise_sampler = cls()
        noise_sampler.set_config(**config)
        return noise_sampler
    
    def set_config(self, shape, std_decrement, current_std):
        self.shape = tuple(shape)
        self.std_decrement = std_decrement
        self.current_std = tf.Variable(current_std, dtype=tf.float32)