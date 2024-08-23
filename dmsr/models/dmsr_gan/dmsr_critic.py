#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:50:27 2024

@author: brennan

This file defines functions for building the critic model for a DMSRGAN.
"""

import tensorflow as tf

from tensorflow import keras
from keras.random import normal
from keras.layers import Input, Conv3D, PReLU, Cropping3D, Flatten, GlobalAveragePooling3D
from keras.initializers import HeNormal

from ...operations.particle_density import ngp_density_field
from ...operations.resizing import scale_up_data, crop_to_match


KWARGS = {
    'data_format'        : 'channels_first',
    'kernel_initializer' : HeNormal()
}


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
    
    # Initial convolutional layers.
    x = Conv3D(channels, 1, **KWARGS)(critic_inputs)
    x = PReLU(shared_axes=(2, 3, 4))(x)
    # num_cells -= 2
    channels *= 2
    
    # Add residual blocks to downsample the data.
    while num_cells > 5:
        if (num_cells - 4) % 2 == 0:
            x = residual_block(x, channels)
            num_cells = (num_cells - 4) // 2
            channels *= 2
            
        else:
            # print('Warning: Adding 2x2x2 convolution between residual blocks')
            # num_cells -= 1
            # x = Conv3D(channels, 2, **KWARGS)(x)
            # x = PReLU(shared_axes=(2, 3, 4))(x)
            print('Breaking at', num_cells)
            break
      
    # Final layer to output a sigle number.
    output = Conv3D(
        1,
        1,
        # activation='sigmoid',
        use_bias=False,
        **KWARGS
    )(x)
    # output = Flatten(data_format='channels_first')(output)
    
    output = GlobalAveragePooling3D(data_format='channels_first')(output)
    
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
    y = Conv3D(channels, 1, use_bias=False, **KWARGS)(x)
    y = Cropping3D(2, data_format='channels_first')(y)
    
    # Convolutional block.
    x = Conv3D(channels, 3, **KWARGS)(x)
    x = PReLU(shared_axes=(2, 3, 4))(x)
    x = Conv3D(channels, 3, **KWARGS)(x)
    x = PReLU(shared_axes=(2, 3, 4))(x)
    
    # Downsample.
    x = x + y
    x = Conv3D(channels, 2, strides=(2, 2, 2), use_bias=False, **KWARGS)(x)
    
    return x



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
        


class NoiseController(keras.callbacks.Callback):
    
    def on_epoch_end(self, epoch, logs=None):
        self.model.noise_sampler.update()
        print(
            "Noise std is now", 
            self.model.noise_sampler.current_std.read_value()
        )



# TODO: this might not work so well with the WGAN critic anymore since the
# critic should learn an unbounded score for real and fake data. The labels
# should be removed and a WGAN loss function should be used for supervised
# learning now. 
def critic_supervised_dataset(LR_data, HR_data, box_size, generator, sampler):
    """
    Returns a new dataset with both HR samples and SR samples generated by the
    given generator. The new dataset can be used for supervised
    training of a critic to learn to distinguish between HR and SR data.
    """
    gen_inputs = [(field[None, ...], sampler(1)) for field in LR_data]
    SR_data = [generator(x) for x in gen_inputs]
    SR_data = tf.concat(SR_data, axis=0)
    
    US_data = scale_up_data(LR_data, scale=2)
    US_data = crop_to_match(US_data, HR_data)
    US_density = ngp_density_field(US_data, box_size)
    US_data = tf.concat((US_density, US_data), axis=1)
    
    HR_density = ngp_density_field(HR_data, box_size)
    real_data = tf.concat((HR_density, HR_data, US_data), axis=1)
    real_labels = tf.ones(real_data.shape[0],)
    
    SR_density = ngp_density_field(SR_data, box_size)
    fake_data = tf.concat((SR_density, SR_data, US_data), axis=1)
    fake_labels = tf.zeros(fake_data.shape[0],)
    
    data = tf.concat((real_data, fake_data), axis=0)
    labels = tf.concat((real_labels, fake_labels), axis=0)
    
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    return dataset.shuffle(len(dataset))