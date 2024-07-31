#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:47:05 2024

@author: brennan

This script loops over seed directories and plots the particle distribution
for each simulation snapshot found in the directory.
"""

import glob
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt


seed_paths = glob.glob('run*/')
for seed_path in seed_paths:
    
    # If the seed directory has no snapshots in it then skip.
    snapshot_paths = np.sort(glob.glob(seed_path + '*/snap*.hdf5'))
    if snapshot_paths.size == 0:
        continue
    
    # Create a figure to hold all the snapshot plots.
    num_runs = len(snapshot_paths) // 3
    figure, axes = plt.subplots(num_runs, 3)
    images = []
    
    # For each snapshot, create a 2d histogram of the particle distribution.
    for n, path in enumerate(snapshot_paths):
        snapshot = h5.File(path, 'r')
        i, j = n // 3, n % 3
        coordinates = np.asarray(snapshot['DMParticles']['Coordinates'])
        
        hist, _, _, image = axes[i, j].hist2d(
            coordinates[:, 0], 
            coordinates[:, 1],
            bins = 200,
            norm = 'log',
            cmap=plt.cm.Blues
        )
        
        images.append(image)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        snapshot.close()
     
    # Set the title and axis labels for the figure.
    axes[0, 0].set_title('z = 4')
    axes[0, 1].set_title('z = 2')
    axes[0, 2].set_title('z = 0')
    
    for n, ax in enumerate(axes[:, 0]):
        num_particles = 64 * 2**(n) 
        ax.set_ylabel(f'N = {num_particles}^3')
        
    # Adjust the colour bar limits and whitespace between subplots.
    for image in images:
        image.set_clim(1, 9000)
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # Save and close the figure.
    plot_name = seed_path + 'particle_histograms.png'
    figure.savefig(plot_name, dpi=200)
    plt.show()
    plt.close()