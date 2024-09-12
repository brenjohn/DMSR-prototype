#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:30:33 2024

@author: brennan
"""

import sys
sys.path.append("../..")

import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.fftpack import fftn, fftshift
from dmsr.operations.particle_density import ngp_density_field

font = {'size' : 21}
matplotlib.rc('font', **font)


#%%

def compute_power_spectrum(displacements, particle_mass, box_size, grid_size):
    volume = box_size**3
    cell_size = box_size / grid_size
    cell_volume = cell_size**3
    
    # Compute the denisty field from the given displacement field.
    density = ngp_density_field(displacements[None, ...], box_size).numpy()
    density = density[0, 0, ...] * particle_mass / cell_volume
    
    # Compute the denisty contrast.
    mean_density = np.mean(density)
    density -= mean_density
    density /= mean_density
    
    # Get the fourier transform of the density field and
    # shift the zero-frequency component to the center.
    density_ft = fftn(density) / (grid_size**3)
    
    power_spectrum = np.abs(density_ft)**2
    
    # Compute the frequency arrays
    ks = 2 * np.pi * np.fft.fftfreq(grid_size, box_size/grid_size) / box_size
    kx, ky, kz = np.meshgrid(ks, ks, ks, indexing='ij')
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    
    # Radial bins
    k_bins = np.linspace(0, np.max(k), num=grid_size//2)
    k_bin_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    power_spectrum_radial = np.zeros_like(k_bin_centers)
    
    # Average the power spectrum over spherical shells
    for i in range(len(k_bin_centers)):
        shell_mask = (k >= k_bins[i]) & (k < k_bins[i+1])
        power = k[shell_mask]**3 * power_spectrum[shell_mask] * volume
        power_spectrum_radial[i] = np.mean(power) 
    
    return k_bin_centers, power_spectrum_radial


#%%

def plot_spectra(meta_data, output_dir, step, n = 0, save=False):
    # TODO: These hard coded values should be replaced by metadata values read
    # from a metadata file corresponding to outputs about to be processed.
    LR_size = 20
    HR_size = 32
    patch_size = 20 / 64
    HR_patch_size = 32 / 128
    
    LR_filename = output_dir + f'/LR_sample_{n}_{step}.npy'
    LR_sample = np.load(LR_filename)
    LR_ks, LR_spectrum = compute_power_spectrum(
        LR_sample, 1, patch_size, LR_size
    )
    
    SR_filename = output_dir + f'/SR_sample_{n}_{step}.npy'
    SR_sample = np.load(SR_filename)
    SR_ks, SR_spectrum = compute_power_spectrum(
        SR_sample, 1/8, HR_patch_size, HR_size
    )
    
    HR_filename = output_dir + f'/HR_sample_{n}_{step}.npy'
    HR_sample = np.load(HR_filename)
    HR_ks, HR_spectrum = compute_power_spectrum(
        HR_sample, 1/8, HR_patch_size, HR_size
    )
    
    plt.figure(figsize=(8, 8))
    plt.plot(HR_ks, HR_spectrum, label='HR', linewidth=4, color='red')
    plt.plot(LR_ks, LR_spectrum, label='LR', linewidth=4)
    plt.plot(SR_ks, SR_spectrum, label='SR', linewidth=4, color='black')
    plt.xscale('log')
    plt.yscale('log')
    # plt.ylim((1e4, 5e7))
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.title(f'Epoch {step}')
    plt.grid()
    plt.legend()
    
    if save:
        plots_dir = 'plots/training_spectra/'
        os.makedirs(plots_dir, exist_ok=True)
        plot_name = plots_dir + f'output_power_sprectra_{step:04}.png'
        plt.savefig(plot_name, dpi=100)
    else:
        plt.show()
        
    plt.close()


#%%

outputs_dir = 'data/training_outputs/'
output_dirs = glob.glob(outputs_dir + 'step_*')
output_dirs = np.sort(output_dirs)

data_directory = '../../data/dmsr_training/'
meta_file = data_directory + 'metadata.npy'
meta_data = np.load(meta_file)

for output in output_dirs:
    step = int(output.split('_')[-1])
    plot_spectra(meta_data, output, step, save=True)