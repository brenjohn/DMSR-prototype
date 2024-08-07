#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:52:42 2024

@author: brennan
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import glob
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

from dmsr.operations.resizing import crop_edge
from dmsr.models.dmsr_gan.dmsr_gan import DMSRGAN
from dmsr.models.dmsr_gan.dmsr_gan import DMSRMonitor
from dmsr.operations.augmentation import random_transformation


#%%

def load_dataset(data_directory):
    """
    """
    LR_data = np.load(data_directory + 'LR_fields.npy')
    LR_data = tf.cast(LR_data, dtype=tf.float32)
    
    HR_data = np.load(data_directory + 'HR_fields.npy')
    HR_data = tf.cast(HR_data, dtype=tf.float32)
    
    meta_file = data_directory + 'metadata.npy'
    box_size, LR_grid_size, HR_grid_size = np.load(meta_file)
    
    return LR_data, HR_data, box_size, LR_grid_size, HR_grid_size


data_directory = '../../data/dmsr_training/'
data = load_dataset(data_directory)
LR_data, HR_data, box_size, LR_grid_size, HR_grid_size = data

batch_size = 1
HR_crop_size = 4
HR_box_size = (HR_grid_size - 2 * HR_crop_size) * box_size / HR_grid_size
scale_factor = int(HR_grid_size / LR_grid_size)

HR_data = crop_edge(HR_data, size=HR_crop_size)
dataset = tf.data.Dataset.from_tensor_slices((LR_data, HR_data))
dataset = dataset.map(random_transformation)
dataset = dataset.shuffle(len(dataset))
dataset = dataset.batch(batch_size)


#%% Create the GAN.
gan_args = { 
    'LR_grid_size'       : int(LR_grid_size),
    'scale_factor'       : scale_factor,
    'HR_box_size'        : HR_box_size,
    'generator_channels' : 256,
    'critic_channels'    : 16,
    'critic_steps'       : 10,
    'gp_weight'          : 10.0,
    'gp_rate'            : 16,
    'noise_std'          : 0.0,
    'noise_epochs'       : 70
}

gan = DMSRGAN(**gan_args)


#%%
supervised_dataset = gan.supervised_dataset(dataset, batch_size)

gan.generator.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.0),
    loss      = keras.losses.MSE
)


#%%
history_supervised = gan.generator.fit(supervised_dataset, epochs = 50)


#%%
gan.compile(
    critic_optimizer    = keras.optimizers.Adam(
        learning_rate=0.000001, beta_1=0.0, beta_2=0.99
    ),
    generator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0000001, beta_1=0.0, beta_2=0.99
    ),
)


#%% Train the GAN.
generator_noise = gan.sampler(2)
LR_samples = LR_data[1:3, ...]
HR_samples = HR_data[1:3, ...]
cbk = DMSRMonitor(generator_noise, LR_samples, HR_samples)

#%%
history_gan = gan.fit(dataset, epochs=2048, callbacks=[cbk])


#%%
plt.figure()
plt.plot(history_gan.epoch, history_gan.history['critic_loss'], label='critic')
plt.plot(history_gan.epoch, history_gan.history['generator_loss'], label='gen')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss-history.png', dpi=300)


#%%
plt.figure()

plt.plot(
    cbk.critic_batch_loss, 
    linewidth=0.1, color='black', alpha=0.4
)
plt.plot(
    cbk.generator_batches, cbk.generator_batch_loss, 
    linewidth=0.1, color='red', alpha=0.4
)

plt.plot(
    cbk.critic_epochs, cbk.critic_epoch_loss, 
    label='critic', linewidth=2, color='black'
)
plt.plot(
    cbk.generator_epochs, cbk.generator_epoch_loss, 
    label='generator', linewidth=2, color='red'
)

plt.xlabel('Batches')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss-batch-history.png', dpi=300)
