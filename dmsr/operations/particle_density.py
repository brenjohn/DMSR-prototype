#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:55:53 2024

@author: brennan
"""

import numpy as np
import tensorflow as tf


def cic_density_field_python(positions, box_length):
    """
    Create a density field using the cloud in cells (CIC) method
    """
    N = positions.shape[0] # The grid size
    density = tf.zeros((N, N, N))
    cell_size = box_length / N
                
    for x, y, z in positions.reshape(-1, 3):
        
        i = int(x // cell_size)
        j = int(y // cell_size)
        k = int(z // cell_size)
        
        dx = x - i * cell_size
        dy = y - j * cell_size
        dz = z - k * cell_size
        
        i %= N; j %= N; k %= N
            
        density[        i,         j,         k] += (1-dx) * (1-dy) * (1-dz)
        density[        i,         j, (k+1) % N] += (1-dx) * (1-dy) * dz
        density[        i, (j+1) % N,         k] += (1-dx) * dy     * (1-dz)
        density[        i, (j+1) % N, (k+1) % N] += (1-dx) * dy     * dz
        density[(i+1) % N,         j,         k] += dx     * (1-dy) * (1-dz)
        density[(i+1) % N,         j, (k+1) % N] += dx     * (1-dy) * dz
        density[(i+1) % N, (j+1) % N,         k] += dx     * dy     * (1-dz)
        density[(i+1) % N, (j+1) % N, (k+1) % N] += dx     * dy     * dz
                    
    return density



def ngp_density_field_python(positions, box_length):
    """
    Create a density field using the nearest grid point (ngp) method
    """
    N = positions.shape[0] # The grid size
    density = np.zeros((N, N, N))
    cell_size = box_length / N
                
    for x, y, z in positions.reshape(-1, 3):
        
        i = int(x // cell_size) % N
        j = int(y // cell_size) % N
        k = int(z // cell_size) % N
        
        density[i, j, k] += 1
    
    return density


@tf.function
def ngp_density_field(positions, box_length, periodic=False):
    """
    Compute the density field using the nearest grid point method for a batch 
    of particles.

    Args:
    - particles: A tensor of shape (batch_size, num_particles, num_dimensions) 
                 containing particle positions.
                 
    - grid_shape: A tuple representing the shape of the grid.

    Returns:
    - density_field: A tensor of shape (batch_size, *grid_shape) representing 
                     the density field for each batch.
    """
    data_shape = tf.shape(positions)
    batch_size, grid_size = data_shape[0], data_shape[-1]
    cell_size = tf.cast(
        box_length / tf.cast(grid_size, tf.float32), 
        tf.float32
    )
    
    # Add the grid positions to the given relative positions to get the
    # absolute positions of particles.
    r = tf.range(0, box_length, cell_size)
    r = tf.cast(r, dtype=tf.float32)
    X, Y, Z = tf.meshgrid(r, r, r, indexing='ij')
    grid_positions = tf.stack((X, Y, Z), axis=0)
    grid_positions = grid_positions[None, ...]
    grid_positions = tf.repeat(grid_positions, repeats=batch_size, axis=0)
    positions += grid_positions
    
    # Create batch indices.
    batch_indices = tf.range(batch_size, dtype=tf.int32)
    batch_indices = tf.reshape(batch_indices, (batch_size, 1, 1, 1, 1))
    batch_indices = tf.tile(batch_indices, (1, 1) + 3 * (grid_size,))
    batch_indices = tf.reshape(batch_indices, (-1, 1))
    
    # Compute the grid indices for each particle in the batch.
    grid_indices = tf.floor(positions / cell_size)
    grid_indices = tf.transpose(grid_indices, perm=(0, 2, 3, 4, 1))
    grid_indices = tf.reshape(grid_indices, (-1, 3))
    grid_indices = tf.cast(grid_indices, tf.int32)
    
    if periodic:
        grid_indices = tf.math.mod(grid_indices, tf.cast(grid_size, tf.float32))
    
    # Note: The below block of code ensures indices are within the bounds of
    # the desired denisty tensor. This is only needed for executing this
    # function on a CPU. On GPUs the scatter_nd function ignores indices that
    # are out of bounds.
    
    # else:
    #     # Filter out indices that are outside the boundary.
    #     mask = tf.logical_and(grid_indices >= 0, grid_indices < grid_size)
    #     mask = tf.reduce_all(mask, axis=-1)
        
    #     grid_indices = tf.boolean_mask(grid_indices, mask)
    #     batch_indices = tf.boolean_mask(batch_indices, mask)
    
    indices = tf.concat((batch_indices, grid_indices), axis=-1)
    
    # TODO: This comment makes no sense :P
    # Compute unique grid indices for the entire batch
    updates = tf.ones((tf.shape(indices)[0],), dtype=tf.float32)
    
    # Create a dense grid tensor for the entire batch.
    # Normalise to have M tot = 1
    density_shape = (batch_size, grid_size, grid_size, grid_size)
    density_field = tf.scatter_nd(indices, updates, shape=density_shape)
    density_field /= tf.cast(grid_size**3, dtype=tf.float32)
    
    return density_field[:, None, ...]



@tf.function
def cic_density_field(relative_positions, box_length):
    """
    Compute the density field using the cloud in cells method for a batch 
    of particles.

    Args:
    - relative_positions : Tensor of shape (batch_size, cells, cells, cells, 3) 
                           containing particle positions relative to the
                           initial grid.
                 
    - box_length         : Length of the box to compute the density field for.

    Returns:
    - density_field : A tensor of shape (batch_size, cells, cells, cells, 1) 
                      representing the density field for each batch.
    """
    print('Tracing cic')
    data_shape = tf.shape(relative_positions)
    batch_size, grid_size = data_shape[0], data_shape[-1]
    cell_size = tf.cast(box_length, tf.float32) / tf.cast(grid_size, tf.float32)
    
    # Add the grid positions to the given relative positions to get the
    # absolute positions of particles.
    r = tf.range(0, box_length, cell_size, dtype=tf.float32)
    X, Y, Z = tf.meshgrid(r, r, r, indexing='ij')
    grid_positions = tf.stack((X, Y, Z), axis=0)[None, ...]
    grid_positions = tf.repeat(grid_positions, repeats=batch_size, axis=0)
    
    positions = relative_positions + grid_positions
    positions = positions / cell_size
    
    # Compute the indices for the grid cells associated with each particle.
    # Note, grid indices here can be outside the boundary (0, grid_size).
    # these indices correspond to "ghost zones" that pad the main grid.
    positions = tf.transpose(positions, perm=(0, 2, 3, 4, 1))
    grid_indices = tf.floor(positions)
    
    # Create the batch component of the indices.
    batch_indices = tf.range(batch_size, dtype=tf.int32)
    batch_indices = batch_indices[:, None, None, None, None]
    batch_indices = tf.tile(batch_indices, (1, 1) + 3 * (grid_size,))
    batch_indices = tf.reshape(batch_indices, (-1, 1))
    
    
    #========================================================================#
    
    grid_indices = tf.reshape(grid_indices, (-1, 3))
    positions = tf.reshape(positions, (-1, 3))
    # positions = tf.reshape(relative_positions, (-1, 3))
    
    neighbours_positions = tf.bitwise.right_shift(
        tf.range(8)[:, None], 
        tf.range(2, -1, -1)
    ) & 1
    neighbours_positions = tf.cast(neighbours_positions, tf.float32)

    targets = grid_indices[:, None, :] + neighbours_positions
    # targets = tf.reshape(targets, (-1, 3))
    # weights = tf.reduce_prod(targets, axis=-1, keepdims=True)
    # weights = positions - grid_indices
    # weights = tf.reduce_prod(relative_positions, axis=-1, keepdims=True)
    # weights = tf.repeat(weights, 8, axis=0)
    # weights = positions - targets
    # weights = positions[:, None, :] - targets
    weights = 1 - tf.abs(positions[:, None, :] - targets)
    weights = tf.reduce_prod(weights, axis=-1, keepdims=True)
    weights = tf.reshape(weights, (-1,))
    
    #========================================================================#
    
    grid_indices = tf.cast(grid_indices, tf.int32)
    grid_indices = tf.concat((batch_indices, grid_indices), axis=-1)
    
    neighbours = tf.bitwise.right_shift(
        tf.range(8)[:, None], tf.range(3, -1, -1)
    ) & 1
    
    indices = grid_indices[:, None, :] + neighbours
    indices = tf.reshape(indices, (-1, 4))
    
    # weights = tf.ones((tf.shape(indices)[0],), dtype=tf.float32)
    # weights = tf.random.normal((tf.shape(indices)[0],), dtype=tf.float32)
    
    # Create the (ix, iy, iz)-component of the cic density field.
    density_shape = (batch_size, grid_size, grid_size, grid_size)
    density_field = tf.scatter_nd(indices, weights, shape=density_shape)
    density_field /= tf.cast(grid_size**3, dtype=tf.float32)
    
    return density_field[:, None, ...]


#%%
# batch_indices = tf.Variable([0, 1])
grid = tf.Variable([[0, 1, 1, 1], [0, 2, 2, 2], [1, 3, 3, 3], [1, 4, 4, 4]])
ns = tf.bitwise.right_shift(tf.range(8)[:, None], tf.range(3, -1, -1)) & 1
p = grid[:, None, :] + ns
p = tf.reshape(p, (-1, 4))

#%%
gs = tf.Variable([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
xs = tf.Variable([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3], [3.1, 3.2, 3.3], [4.1, 4.2, 4.3]])

dx = xs - gs
ns = tf.bitwise.right_shift(
    tf.range(8)[:, None], 
    tf.range(2, -1, -1)
) & 1
ns = tf.cast(ns, tf.float32)

ts = gs[:, None, :] + ns
ws = 1 - tf.abs(xs[:, None, :] - ts)
ws = tf.reduce_prod(ws, axis=-1)

#%%
