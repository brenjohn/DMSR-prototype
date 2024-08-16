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
from dmsr.models.dmsr_gan.dmsr_critic import NoiseController
from dmsr.models.dmsr_gan.dmsr_monitor import DMSRMonitor
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
    meta_data = np.load(meta_file)
    box_size, patch_size, LR_size, HR_size, LR_mass, HR_mass = meta_data
    
    return LR_data, HR_data, patch_size, LR_size, HR_size


data_directory = '../../data/dmsr_training/'
data = load_dataset(data_directory)
LR_data, HR_data, box_size, LR_grid_size, HR_grid_size = data

batch_size = 1
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
    'generator_channels' : 512,
    'critic_channels'    : 64,
    'critic_steps'       : 5,
    'generator_steps'    : 1,
    'gp_weight'          : 10.0,
    'gp_rate'            : 1,
    'noise_std'          : 0.0,
    'noise_epochs'       : 1
}

gan = DMSRGAN(**gan_args)


#%%
generator_dataset = gan.generator_supervised_dataset(dataset, batch_size)

gan.generator.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.0),
    loss      = keras.losses.MSE
)


#%%
generator_history = gan.generator.fit(generator_dataset, epochs = 5)


# #%%
# critic_dataset = gan.critic_supervised_dataset(LR_data, HR_data)
# critic_dataset = critic_dataset.batch(7)

# gan.critic.compile(
#     optimizer = keras.optimizers.Adam(learning_rate=0.000005, beta_1=0.0),
#     loss      = keras.losses.MSE
# )


# #%%
# critic_history = gan.critic.fit(critic_dataset, epochs = 21)


#%%
gan.compile(
    critic_optimizer    = keras.optimizers.Adam(
        learning_rate=0.00002, beta_1=0.0, beta_2=0.99 , weight_decay=0.000001
    ),
    generator_optimizer = keras.optimizers.Adam(
        learning_rate=0.00001, beta_1=0.0, beta_2=0.99
    ),
)


#%% Train the GAN.
generator_noise = gan.sampler(2)
LR_samples = LR_data[1:3, ...]
HR_samples = HR_data[1:3, ...]
monitor = DMSRMonitor(generator_noise, LR_samples, HR_samples)
noise_controller = NoiseController()

callbacks = [monitor]
# callbacks = [monitor, noise_controller]

#%%
history_gan = gan.fit(dataset, epochs=6144, callbacks=callbacks)


#%%
gan.critic_optimizer.learning_rate.assign(0.000001)
gan.generator_optimizer.learning_rate.assign(0.000001)

#%%
model_filename = 'my_gan'
gan.save(model_filename)

#%%
loaded_model = tf.keras.models.load_model('my_gan')


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

plt.ylim(-5000, 5000)
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss-batch-history.png', dpi=300)


#%%
critic_weights = np.asarray([])
for weights in gan.critic.get_weights():
    weights = weights.reshape(-1)
    critic_weights = np.concatenate((critic_weights, weights), axis=0)
    

generator_weights = np.asarray([])
for weights in gan.generator.get_weights():
    weights = weights.reshape(-1)
    generator_weights = np.concatenate((generator_weights, weights), axis=0)

plt.figure()
plt.hist(critic_weights, bins=500)
# plt.show()
# plt.close()

# plt.figure()
plt.hist(generator_weights, bins=500)
plt.yscale('log')
plt.show()
plt.close()