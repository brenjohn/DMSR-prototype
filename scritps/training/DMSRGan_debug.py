#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 19:16:28 2024

@author: brennan
"""

import os
import sys
sys.path.append("..")
sys.path.append("../..")

import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from dmsr.operations.resizing import crop_edge
from dmsr.models.dmsr_gan.dmsr_gan import DMSRGAN
from dmsr.models.dmsr_gan.dmsr_critic import DMSRCritic
from dmsr.models.dmsr_gan.dmsr_generator import DMSRGenerator
from dmsr.models.dmsr_gan.dmsr_gan import DMSRMonitor


#%%

def load_dataset(LR_directory, HR_directory):
    """
    """
    
    LR_files = glob.glob(LR_directory + 'sample*.npy')
    LR_files = np.sort(LR_files)
    LR_data = [np.load(f) for f in LR_files]
    LR_data = tf.cast(LR_data, dtype=tf.float32)
    
    HR_files = glob.glob(HR_directory + 'sample*.npy')
    HR_files = np.sort(HR_files)
    HR_data = [np.load(f) for f in HR_files]
    HR_data = tf.cast(HR_data, dtype=tf.float32)
    
    return LR_data, HR_data


critic_input_shape = (-1, 60, 60, 60, 8)

LR_directory = '../../data/LR_data/music32/'
HR_directory = '../../data/HR_data/music64/'
LR_data, HR_data = load_dataset(LR_directory, HR_directory)

batch_size = 2
HR_data = crop_edge(HR_data, size=2)
dataset = tf.data.Dataset.from_tensor_slices((LR_data, HR_data))
dataset = dataset.shuffle(len(dataset))
dataset = dataset.batch(batch_size)


#%%
# 3 LR space dimensions + 3 HR space dimensions.
generator_in_channels     = 3 + 3

# 3 LR dimensions + 3 HR dimensions + 1 LR density field + 1 HR density field.
discriminator_in_channels = 3 + 3 + 1 + 1

# Create the discriminator.
critic = DMSRCritic(discriminator_in_channels)

# Create the generator.
generator = DMSRGenerator(generator_in_channels)

# Create the GAN.
gan = DMSRGAN(critic = critic, generator = generator)


#%%

LR_data = LR_data[:1, ...]
HR_data = HR_data[:1, ...]

#%%
from dmsr.operations.resizing import crop_to_match, scale_up_data
from dmsr.operations.particle_density import ngp_density_field


US_data = scale_up_data(LR_data, scale=2)
US_data = crop_to_match(US_data, HR_data)

HR_density = ngp_density_field(HR_data, 1)
US_density = ngp_density_field(US_data, 1)

#%%
# SR_data = generator(LR_data)

#%%
real_data = tf.concat([HR_density, HR_data, US_density, US_data], axis=-1)
fake_data = tf.random.normal(tf.shape(real_data))

#%%

def critic_loss(real_logits, fake_logits):
    real_loss = tf.reduce_mean(real_logits)
    fake_loss = tf.reduce_mean(fake_logits)
    return fake_loss - real_loss

with tf.GradientTape() as tape:
    
    SR_data = generator(LR_data)
    SR_density = ngp_density_field(SR_data, 1)
    fake_data = tf.concat([SR_density, SR_data, US_density, US_data], axis=-1)
    
    fake_logits = critic(fake_data)
    real_logits = critic(real_data)
    critic_loss = critic_loss(real_logits, fake_logits)
    
    batch_size = tf.shape(real_data)[0]
    eps = tf.random.uniform([batch_size, 1, 1, 1, 1], 0.0, 1.0)
    diff = fake_data - real_data
    interpolated = real_data + eps * diff
    
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        logit = critic(interpolated)
        
    
    gp_grad = gp_tape.gradient(logit, interpolated)
    
    norm = tf.sqrt(tf.reduce_sum(tf.square(gp_grad), axis=[1, 2, 3, 4]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    # loss = tf.norm(logit)
    loss = critic_loss + gp
    
#%%
critic_grad = tape.gradient(loss, critic.trainable_variables)

#%%


#%%
def gradient_penalty(real_data, fake_data):
    """Calculates the gradient penalty.

    This loss is calculated on interpolated data and added to the critic 
    loss.
    """
    # Get the interpolated image
    batch_size = tf.shape(real_data)[0]
    eps = tf.random.uniform([batch_size, 1, 1, 1, 1], 0.0, 1.0)
    diff = fake_data - real_data
    interpolated = real_data + eps * diff

    # 1. Get the critic output for this interpolated data.
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic(interpolated)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, interpolated)
    
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3, 4]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp # tf.cast(0.0, tf.float32)

