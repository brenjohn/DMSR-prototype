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
    batch_size, grid_size = data_shape[0], data_shape[1]
    cell_size = tf.cast(
        box_length / tf.cast(grid_size, tf.float64), 
        tf.float32
    )
    
    # Add the grid positions to the given relative positions to get the
    # absolute positions of particles.
    r = tf.range(0, box_length, cell_size)
    r = tf.cast(r, dtype=tf.float32)
    X, Y, Z = tf.meshgrid(r, r, r, indexing='ij')
    grid_positions = tf.stack((X, Y, Z), axis=-1)
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
    # grid_indices = tf.transpose(grid_indices, perm=(0, 2, 3, 4, 1)) # For channels_first case
    grid_indices = tf.reshape(grid_indices, (-1, 3))
    grid_indices = tf.cast(grid_indices, tf.int32)
    
    if periodic:
        grid_indices = tf.math.mod(grid_indices, tf.cast(grid_size, tf.float32))
    
    else:
        # Filter out indices that are outside the boundary.
        mask = tf.logical_and(grid_indices >= 0, grid_indices < grid_size)
        mask = tf.reduce_all(mask, axis=-1)
        
        grid_indices = tf.boolean_mask(grid_indices, mask)
        batch_indices = tf.boolean_mask(batch_indices, mask)
    
    indices = tf.concat((batch_indices, grid_indices), axis=-1)
    
    # TODO: This comment makes no sense :P
    # Compute unique grid indices for the entire batch
    updates = tf.ones((tf.shape(indices)[0],), dtype=tf.float32)
    
    # Create a dense grid tensor for the entire batch.
    # Normalise to have M tot = 1
    density_shape = (batch_size, grid_size, grid_size, grid_size)
    density_field = tf.scatter_nd(indices, updates, shape=density_shape)
    # density_field /= tf.cast(grid_size**3, dtype=tf.float32)
    
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
    data_shape = tf.shape(relative_positions)
    batch_size, grid_size = data_shape[0], data_shape[1]
    cell_size = tf.cast(
        box_length / tf.cast(grid_size, tf.float64), 
        tf.float32
    )
    
    # Add the grid positions to the given relative positions to get the
    # absolute positions of particles.
    r = tf.range(0, box_length, cell_size)
    r = tf.cast(r, dtype=tf.float32)
    X, Y, Z = tf.meshgrid(r, r, r, indexing='ij')
    grid_positions = tf.stack((X, Y, Z), axis=-1)
    grid_positions = grid_positions[None, ...]
    grid_positions = tf.repeat(grid_positions, repeats=batch_size, axis=0)
    positions = relative_positions + grid_positions
    
    # Compute the indices for the grid cells associated with each particle.
    # Note, grid indices where can be outside the boundary (0, grid_size).
    # these indices correspond to "ghost zones" that pad the main grid.
    positions /= cell_size
    grid_indices = tf.floor(positions)
    # grid_indices = tf.transpose(grid_indices, perm=(0, 2, 3, 4, 1))
    
    # Create the batch component of the indices.
    batch_indices = tf.range(batch_size, dtype=tf.int32)
    batch_indices = batch_indices[:, None, None, None, None]
    batch_indices = tf.tile(batch_indices, (1,) + 3 * (grid_size,) + (1,))
    batch_indices = tf.reshape(batch_indices, (-1, 1))
    
    # Compute the displacements of each particle from their associated grid
    # points. These will determine the weight of each particle to add to
    # neighbouring cells.
    displacement = positions - grid_indices
    displacement = tf.reshape(displacement, (-1, 3))
    dx, dy, dz = tf.split(displacement, 3, axis=-1)
    dx = tf.squeeze(dx, axis=-1)
    dy = tf.squeeze(dy, axis=-1)
    dz = tf.squeeze(dz, axis=-1)
    
    # Reshape grid indices into a list of indices and prepare the arguments
    # for computing the different cic density field components.
    grid_indices = tf.reshape(grid_indices, (-1, 3))
    grid_indices = tf.cast(grid_indices, tf.int32)
    num_indices = tf.shape(grid_indices)[0]
    ones = tf.ones(num_indices, dtype=tf.int32)
    density_component_args = (
        grid_indices, 
        grid_size,
        batch_size,
        batch_indices,
        ones,
        dx, dy, dz
    )
    
    # Compute the cic density fields by summing components from the eight
    # neighbouring cells of each particle.
    density_field  = cic_density_component(*density_component_args, 0, 0, 0)
    density_field += cic_density_component(*density_component_args, 0, 0, 1)
    density_field += cic_density_component(*density_component_args, 0, 1, 0)
    density_field += cic_density_component(*density_component_args, 0, 1, 1)
    density_field += cic_density_component(*density_component_args, 1, 0, 0)
    density_field += cic_density_component(*density_component_args, 1, 0, 1)
    density_field += cic_density_component(*density_component_args, 1, 1, 0)
    density_field += cic_density_component(*density_component_args, 1, 1, 1)
    
    return density_field[:, None, ...]



@tf.function
def cic_density_component(
        grid_indices, 
        grid_size,
        batch_size,
        batch_indices,
        ones,
        dxs, dys, dzs,
        ix, iy, iz
    ):
    """
    Returns the (ix, iy, iz) component of the cloud-in-cells density field for
    the given particle displacements (dx, dy, dz).
    """
    # Get the indices for the (ix, iy, iz) neighbour of each cell by shifting
    # the grid indices appropriately. 
    shift = tf.stack([ix * ones, iy * ones, iz * ones], axis=-1)
    shifted_indices = grid_indices + shift
    
    # collect indices that are inside the boundary.
    mask = tf.logical_and(shifted_indices >= 0, shifted_indices < grid_size)
    mask = tf.reduce_all(mask, axis=-1)
    shifted_indices = tf.boolean_mask(shifted_indices, mask)
    filtered_batch_indices = tf.boolean_mask(batch_indices, mask)
    indices = tf.concat((filtered_batch_indices, shifted_indices), axis=-1)
    
    # Compute the weights for each cell to be added to the density component.
    dx = tf.boolean_mask(dxs, mask)
    dy = tf.boolean_mask(dys, mask)
    dz = tf.boolean_mask(dzs, mask)
    weights  = (1 - ix) - (1 - 2 * ix) * dx
    weights *= (1 - iy) - (1 - 2 * iy) * dy
    weights *= (1 - iz) - (1 - 2 * iz) * dz
    
    # Create the (ix, iy, iz)-component of the cic density field.
    density_shape = (batch_size, grid_size, grid_size, grid_size)
    density_field = tf.scatter_nd(indices, weights, shape=density_shape)
    
    return density_field
