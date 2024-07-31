#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 20:29:49 2024

@author: brennan
"""

import sys
sys.path.append("../..")

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from dmsr.operations.particle_density import ngp_density_field


#%%
def get_sample_positions(outputs_dir, step, n=1):
    LR_sample = np.load(outputs_dir + f'/LR_sample_{n}_{step}.npy')
    SR_sample = np.load(outputs_dir + f'/SR_sample_{n}_{step}.npy')
    HR_sample = np.load(outputs_dir + f'/HR_sample_{n}_{step}.npy')
    
    return LR_sample, SR_sample, HR_sample


#%%

def plot_samples(output_dir, step, save=False):
    
    positions = get_sample_positions(output_dir, step)
    LR_sample, SR_sample, HR_sample = positions
    
    LR_density = ngp_density_field(LR_sample[None, ...], 1)[0, 0, ...]
    SR_density = ngp_density_field(SR_sample[None, ...], 1)[0, 0, ...]
    HR_density = ngp_density_field(HR_sample[None, ...], 1)[0, 0, ...]
    
    # Create a figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 7))
    
    # LR density plot
    d = np.sum(LR_density, axis=2)
    ax1.imshow(d)
    ax1.set_title('LR')
    
    # SR density plot
    d = np.sum(SR_density, axis=2)
    ax2.imshow(d)
    ax2.set_title('SR')
    
    # HR density plot
    d = np.sum(HR_density, axis=2)
    ax3.imshow(d)
    ax3.set_title('HR')
    
    # Add title and adjust layout
    fig.suptitle(f'Epoch {step}')
    plt.tight_layout()
    
    if save:
        plots_dir = 'plots/outputs/'
        os.makedirs(plots_dir, exist_ok=True)
        plot_name = plots_dir + f'density_output_{step:04}.png'
        plt.savefig(plot_name, dpi=100)
    else:
        plt.show()
        
    plt.close()


#%%
# plot_samples(output_dir, step, save=False)

# step = 69
# output_dir = f'data/training_outputs/step_{step}/'

outputs_dir = 'data/training_outputs/'
output_dirs = glob.glob(outputs_dir + 'step_*')
output_dirs = np.sort(output_dirs)

for output in output_dirs:
    step = int(output.split('_')[-1])
    plot_samples(output, step, save=True)