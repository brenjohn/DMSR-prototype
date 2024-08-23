#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:39:33 2024

@author: brennan
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt


#%%
def get_sample_positions(outputs_dir, LR_box_size, HR_box_size, step, n=0):
    LR_sample = np.load(outputs_dir + f'/LR_sample_{n}_{step}.npy')
    SR_sample = np.load(outputs_dir + f'/SR_sample_{n}_{step}.npy')
    HR_sample = np.load(outputs_dir + f'/HR_sample_{n}_{step}.npy')
    
    N = LR_sample.shape[-1]
    x = LR_sample
    cell_size = LR_box_size / N
    r = np.arange(0, LR_box_size, cell_size)
    X, Y, Z = np.meshgrid(r, r, r, indexing='ij')
    LR_sample_xs = X + x[0, :, :, :]
    LR_sample_ys = Y + x[1, :, :, :]
    LR_positions = LR_sample_xs.ravel(), LR_sample_ys.ravel()
    
    N = SR_sample.shape[-1]
    x = SR_sample
    cell_size = HR_box_size / N
    r = np.arange(0, HR_box_size, cell_size)
    X, Y, Z = np.meshgrid(r, r, r, indexing='ij')
    SR_sample_xs = X + x[0, :, :, :]
    SR_sample_ys = Y + x[1, :, :, :]
    SR_positions = SR_sample_xs.ravel(), SR_sample_ys.ravel()
    
    N = HR_sample.shape[-1]
    x = HR_sample
    cell_size = HR_box_size / N
    r = np.arange(0, HR_box_size, cell_size)
    X, Y, Z = np.meshgrid(r, r, r, indexing='ij')
    HR_sample_xs = X + x[0, :, :, :]
    HR_sample_ys = Y + x[1, :, :, :]
    HR_positions = HR_sample_xs.ravel(), HR_sample_ys.ravel()
    
    return LR_positions, SR_positions, HR_positions


#%%

def plot_samples(output_dir, step, save=False):
    
    LR_box_size = 44.45234710539101 # 71.12375536862562 / 2
    HR_box_size = 31.11664297377371 # 62.23328594754742 / 2
    positions = get_sample_positions(output_dir, LR_box_size, HR_box_size, step)
    (LR_xs, LR_ys), (SR_xs, SR_ys), (HR_xs, HR_ys) = positions
    
    # Create a figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 14))
    
    # LR scatter plot
    ax1.scatter(LR_xs, LR_ys, alpha=0.2, s=0.5)
    ax1.set_title('LR')
    
    # SR scatter plot
    ax2.scatter(SR_xs, SR_ys, alpha=0.2, s=0.1)
    ax2.set_title('SR')
    
    # HR scatter plot
    ax3.scatter(HR_xs, HR_ys, alpha=0.2, s=0.1)
    ax3.set_title('HR')
    
    # SR scatter plot
    ax4.scatter(SR_xs, SR_ys, alpha=0.2, s=0.1)
    ax4.set_title('SR')
    ax4.set_xlim(ax3.get_xlim())
    ax4.set_ylim(ax3.get_ylim())
    
    # Add title and adjust layout
    fig.suptitle(f'Epoch {step}')
    plt.tight_layout()
    
    if save:
        plots_dir = 'plots/training_samples/'
        os.makedirs(plots_dir, exist_ok=True)
        plot_name = plots_dir + f'output_{step:04}.png'
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

existing_plots = glob.glob('plots/training_samples/output_*')
existing_steps = [plot.split('.')[0].split('_')[-1] for plot in existing_plots]
new_outputs = [output for output in output_dirs 
               if not output.split('_')[-1] in existing_steps]

for output in new_outputs:
    step = int(output.split('_')[-1])
    plot_samples(output, step, save=True)