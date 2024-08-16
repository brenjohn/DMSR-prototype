#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:55:53 2024

@author: brennan
"""

import numpy as np
import tensorflow as tf

def cic_density_field(positions, box_length):
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



def ngp_density_field_first(positions, box_length):
    """
    Create a density field using the nearest grid point (ngp) method
    """
    data_shape = tf.shape(positions)
    batch_size, grid_size = data_shape[0], data_shape[-1]
    density = tf.zeros((batch_size, grid_size, grid_size, grid_size))
    cell_size = tf.cast(box_length / grid_size, tf.float32) # TODO: make sure same dtype as data 
    
    positions /= cell_size
    positions = tf.cast(positions, tf.int32)
    positions = tf.math.mod(positions, grid_size)
    
    batch_indices = tf.range(batch_size, dtype=tf.int32)
    batch_indices = tf.reshape(batch_indices, (batch_size, 1, 1, 1, 1))
    batch_indices = tf.tile(batch_indices, (1, 1, grid_size, grid_size, grid_size))
    
    indices = tf.concat((batch_indices, positions), axis=-1)
    indices = tf.reshape(indices, (-1, 4))
    
    # normalised so that sum density = 1
    vals = tf.ones(tf.shape(indices)[0]) / tf.cast(grid_size**3, tf.float32)
    
    density = tf.tensor_scatter_nd_add(density, indices, vals)
    
    return density[..., None]



def ngp_density_field_old(positions, box_length):
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


def ngp_density_field_tmp(positions, box_length):
    """
    Create a density field using the nearest grid point (ngp) method
    """
    data_shape = tf.shape(positions)
    batch_size, grid_size = data_shape[0], data_shape[1]
    density = tf.zeros((batch_size, grid_size, grid_size, grid_size))
    cell_size = tf.cast(box_length / grid_size, tf.float32) # TODO: make sure same dtype as data 
    
    positions /= cell_size
    positions = tf.cast(positions, tf.int32)
    positions = tf.math.mod(positions, grid_size)
    
    for batch in range(batch_size):
        
        for i, j, k in tf.reshape(positions[batch,...], (-1, 3)):
            density[batch, i, j, k] += 1 / (grid_size**3)
    
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
        box_length / tf.cast(grid_size, tf.float64), 
        tf.float32
    )
    
    # Add the grid positions to the given relative positions to get the
    # absolute positions of particles.
    r = tf.range(0, box_length, cell_size)
    r = tf.cast(r, dtype=tf.float32)
    X, Y, Z = tf.meshgrid(r, r, r)
    grid_positions = tf.stack((X, Y, Z), axis=0)
    positions += grid_positions
    
    # Create batch indices.
    batch_indices = tf.range(batch_size, dtype=tf.int32)
    batch_indices = tf.reshape(batch_indices, (batch_size, 1, 1, 1, 1))
    batch_indices = tf.tile(batch_indices, (1,) + 3 * (grid_size,) + (1,))
    batch_indices = tf.reshape(batch_indices, (-1, 1))
    
    # Compute the grid indices for each particle in the batch.
    grid_indices = positions / cell_size
    grid_indices = tf.transpose(grid_indices, perm=(0, 2, 3, 4, 1))
    grid_indices = tf.reshape(grid_indices, (-1, 3))
    
    if periodic:
        grid_indices = tf.math.mod(grid_indices, tf.cast(grid_size, tf.float32))
    
    else:
        # Filter out indices that are outside the boundary.
        mask = tf.logical_and(
            grid_indices >= tf.cast(0, tf.float32), 
            grid_indices < tf.cast(grid_size, tf.float32)
        )
        mask = tf.reduce_all(mask, axis=-1)
        grid_indices = tf.boolean_mask(grid_indices, mask)
        batch_indices = tf.boolean_mask(batch_indices, mask)
        
    grid_indices = tf.cast(grid_indices, tf.int32)
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


# @tf.function(experimental_compile=True)
# def new_ngp_density_field(displacements, box_length, periodic=False):
#     """
#     Compute the density field using the nearest grid point method for a batch 
#     of particles.

#     Args:
#     - particles: A tensor of shape (batch_size, 3, grid_size, grid_size, grid_size) 
#                  containing particle displacements from an initial grid.
                 
#     - box_length: A float representing the length of the box.
    
#     - periodic: A boolean indicating whether to use periodic boundary conditions.

#     Returns:
#     - density_field: A tensor of shape (batch_size, 1, *grid_shape) representing 
#                      the density field for each batch.
#     """
    
#     data_shape = tf.shape(displacements)
#     batch_size, grid_size = data_shape[0], data_shape[-1]
#     cell_size = tf.cast(box_length, tf.float32) / tf.cast(grid_size, tf.float32)
    
#     # Add the grid positions to the given relative displacements to get the
#     # absolute positions of particles.
#     r = tf.range(0, box_length, cell_size, dtype=tf.float32)
#     X, Y, Z = tf.meshgrid(r, r, r)
#     grid_positions = tf.stack((X, Y, Z), axis=0)
#     positions = displacements + grid_positions
    
#     # Create batch indices.
#     batch_indices = tf.range(batch_size, dtype=tf.int32)
#     batch_indices = tf.reshape(batch_indices, (batch_size, 1, 1, 1, 1))
#     batch_indices = tf.tile(batch_indices, (1,) + 3 * (grid_size,) + (1,))
#     batch_indices = tf.reshape(batch_indices, (-1, 1))
    
#     # Compute the grid indices for each particle in the batch.
#     grid_indices = positions / cell_size
#     grid_indices = tf.transpose(grid_indices, perm=(0, 2, 3, 4, 1))
#     grid_indices = tf.reshape(grid_indices, (-1, 3))
    
#     if periodic:
#         grid_indices = tf.math.mod(grid_indices, tf.cast(grid_size, tf.float32))
    
#     else:
#         # Filter out indices that are outside the boundary.
#         mask = tf.logical_and(
#             grid_indices >= tf.cast(0, tf.float32), 
#             grid_indices < tf.cast(grid_size, tf.float32)
#         )
#         mask = tf.reduce_all(mask, axis=-1)
#         grid_indices = tf.boolean_mask(grid_indices, mask)
#         batch_indices = tf.boolean_mask(batch_indices, mask)
       
#     # Create a tensor of indices for cells to be updated with a particle mass.
#     grid_indices = tf.cast(grid_indices, tf.int32)
#     indices = tf.concat((batch_indices, grid_indices), axis=-1)
    
#     # Assume all particles have a mass of 1.
#     updates = tf.ones((tf.shape(indices)[0],), dtype=tf.float32)
    
#     # Create a tensor containing density fields for the entire batch.
#     density_shape = (batch_size, grid_size, grid_size, grid_size)
#     density_field = tf.scatter_nd(indices, updates, shape=density_shape)
#     return density_field[:, tf.newaxis, ...]
