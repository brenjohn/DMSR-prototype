#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 15:26:18 2024

@author: john
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dmsr.operations.particle_density import ngp_density_field
from dmsr.operations.particle_density import cic_density_field, cic_paint


#%%
def load_dataset(data_directory):
    """
    """
    HR_data = np.load(data_directory + 'HR_fields.npy')
    HR_data = tf.cast(HR_data, dtype=tf.float32)
    
    meta_file = data_directory + 'metadata.npy'
    meta_data = np.load(meta_file)
    box_size, patch_size, LR_size, HR_size, LR_mass, HR_mass = meta_data
    
    return HR_data, patch_size, LR_size, HR_size


data_directory = '../../data/dmsr_training/'
data = load_dataset(data_directory)
HR_data, box_size, LR_grid_size, HR_grid_size = data
HR_data = HR_data[:1, ...]


#%% Creating a scatter plot of particle positions.
def get_sample_positions(data, box_size):
    
    N = data.shape[-2]
    x = data.numpy()
    cell_size = box_size / N
    r = np.arange(0, box_size, cell_size)
    X, Y, Z = np.meshgrid(r, r, r, indexing='ij')
    xs = X + x[0, 0, :, :, :]
    ys = Y + x[0, 1, :, :, :]
    positions = xs.ravel(), ys.ravel()
    
    return positions


def plot_sample(data, box_size):
    xs, ys = get_sample_positions(data, box_size)
    
    # Create a figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # HR scatter plot
    ax.scatter(xs, ys, alpha=0.2, s=0.5)
    ax.set_title('Particle Plot')
    ax.set_xlim((0, box_size))
    ax.set_ylim((0, box_size))
    
    plt.tight_layout()
    plt.show()
    # fig.savefig('particle_before.png', dpi=300)
    plt.close()

plot_sample(HR_data, box_size)


#%% Create heat map of ngp density field
b = tf.cast(box_size, tf.float32)
ti = time.time()
density = ngp_density_field(HR_data, b)
print(f'ngp time: {time.time() - ti}')
density = tf.squeeze(density, axis=0)
density = tf.squeeze(density, axis=0)
density = tf.reduce_sum(density, axis=2)
density = tf.transpose(density, perm=(1, 0))
density = tf.reverse(density, axis=[0])

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plt.imshow(density)
ax.set_title('Density Plot')
plt.tight_layout()
plt.show()
# fig.savefig('ngp_density.png', dpi=300)
plt.close()

vmin = np.min(density)
vmax = np.max(density)

#%% Create heat map of cic density field
b = tf.cast(box_size, tf.float32)
ti = time.time()
density = cic_density_field(HR_data, b)
print(f'cic time: {time.time() - ti}')
density = tf.squeeze(density, axis=0)
density = tf.squeeze(density, axis=0)
density = tf.reduce_sum(density, axis=2)
density = tf.transpose(density, perm=(1, 0))
density = tf.reverse(density, axis=[0])

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plt.imshow(density)#, vmin=vmin, vmax=vmax)
ax.set_title('Density Plot')
plt.tight_layout()
plt.show()
# fig.savefig('cic_density_before.png', dpi=300)
plt.close()


#%%
fake_data = tf.random.normal((1, 3, 32, 32, 32))
b = tf.cast(box_size, tf.float32)
ti = time.time()
density = cic_density_field(HR_data, b)
print(f'cic time: {time.time() - ti}')


#%%
fake_data = tf.random.normal((1, 3, 32, 32, 32))
b = tf.cast(box_size, tf.float32)

ti = time.time()
density = cic_density_field(HR_data, b)
print(f'cic time: {time.time() - ti}')

ti = time.time()
with tf.GradientTape() as tape:
    tape.watch(fake_data)
    # b = tf.cast(0.25, tf.float32)
    density = cic_density_field(fake_data, b)
    
gp_gradient = tape.gradient(
    density, fake_data
)
print('Gradient tooks:', time.time() - ti)


#%%
ti = time.time()
density = cic_paint(HR_data, box_size)
print(f'cic time: {time.time() - ti}')
 
density = tf.squeeze(density, axis=0)
density = tf.squeeze(density, axis=0)
density = tf.reduce_sum(density, axis=2)
density = tf.transpose(density, perm=(1, 0))
density = tf.reverse(density, axis=[0])
 
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plt.imshow(density)
ax.set_title('Density Plot')
plt.tight_layout()
plt.show()
# fig.savefig('density_before.png', dpi=300)
plt.close()