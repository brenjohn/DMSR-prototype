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
from dmsr.models.dmsr_gan.dmsr_gan import build_dmsrgan
from dmsr.models.dmsr_gan.dmsr_monitor import DMSRMonitor
from dmsr.operations.augmentation import random_transformation
from dmsr.models.dmsr_gan.dmsr_checkpoint import DMSRGANCheckpoint


#%%

def load_dataset(data_directory):
    """
    """
    LR_data = np.load(data_directory + 'LR_fields.npy')
    LR_data = tf.cast(LR_data, dtype=tf.float32)
    
    HR_data = np.load(data_directory + 'HR_fields.npy')
    HR_data = tf.cast(HR_data, dtype=tf.float32)
    
    meta_file = data_directory + 'metadata.npy'
    meta_data = np.load(meta_file)
    box_size, HR_patch_size, LR_size, HR_size, LR_mass, HR_mass = meta_data
    
    return LR_data, HR_data, HR_patch_size, LR_size, HR_size



data_directory = '../../data/dmsr_training/'
data = load_dataset(data_directory)
LR_data, HR_data, box_size, LR_grid_size, HR_grid_size = data
# LR_data, HR_data = LR_data[:30, ...], HR_data[:30, ...]

batch_size = 4
HR_crop_size = 0
HR_box_size = (HR_grid_size - 2 * HR_crop_size) * box_size / HR_grid_size
scale_factor = 2

# HR_data = crop_edge(HR_data, size=HR_crop_size)
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
    'critic_channels'    : 32
}

gan = build_dmsrgan(**gan_args)


#%%
gan_training_args = {
    'critic_optimizer'    : keras.optimizers.Adam(
        learning_rate=0.00002, beta_1=0.0, beta_2=0.99 , weight_decay=0.000001
    ),
    'generator_optimizer' : keras.optimizers.Adam(
        learning_rate=0.00001, beta_1=0.0, beta_2=0.99
    ),
    'critic_steps' : 2,
    'gp_weight'    : 10.0,
    'gp_rate'      : 1,
}

gan.compile(**gan_training_args)

#%%

checkpoint_prefix = './data/checkpoints/dmsr_gan_checkpoint_{epoch}'
gan_checkpoint = DMSRGANCheckpoint(gan, checkpoint_prefix, checkpoint_rate=100)


#%% Train the GAN.
generator_noise = gan.sampler(2)
LR_samples = LR_data[1:3, ...]
HR_samples = HR_data[1:3, ...]
monitor = DMSRMonitor(generator_noise, LR_samples, HR_samples)

#%%
# callbacks = [monitor, gan_checkpoint]
callbacks = [monitor, gan_checkpoint]

#%%
history_gan = gan.fit(dataset, epochs=6144, callbacks=callbacks)


#%%
gan.critic_optimizer.learning_rate.assign(0.00001)
gan.generator_optimizer.learning_rate.assign(0.000005)


# #%%
# history_gan = gan.fit(dataset, epochs=1024, callbacks=callbacks)


#%%
checkpoint_dir = './data/checkpoints'
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

#%%
gan_checkpoint.restore(latest_checkpoint)

#%%
plt.figure()

plt.plot(
    monitor.critic_batches, monitor.critic_batch_loss, 
    linewidth=0.1, color='black', alpha=0.4
)
plt.plot(
    monitor.generator_batches, monitor.generator_batch_loss, 
    linewidth=0.1, color='red', alpha=0.4
)

plt.plot(
    monitor.critic_epochs, monitor.critic_epoch_loss, 
    label='critic', linewidth=2, color='black'
)
plt.plot(
    monitor.generator_epochs, monitor.generator_epoch_loss, 
    label='generator', linewidth=2, color='red'
)

# plt.ylim(-10000, 4000)
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss-batch-history.png', dpi=300)


#%%
plt.figure()

plt.plot(
    monitor.critic_batches, -1 * np.asarray(monitor.critic_batch_loss), 
    linewidth=0.1, color='black', alpha=0.4
)

plt.plot(
    monitor.critic_epochs, -1 * np.asarray(monitor.critic_epoch_loss), 
    linewidth=2, color='black'
)

plt.yscale('log')
plt.ylim(500, 4000)
plt.xlabel('Batches')
plt.ylabel('Wasserstein Distance Approximation')
plt.savefig('Wasserstein-Distance-history.png', dpi=300)