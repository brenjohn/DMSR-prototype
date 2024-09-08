#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:30:33 2024

@author: brennan
"""

import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.fftpack import fftn, fftshift

font = {'size' : 21}
matplotlib.rc('font', **font)


#%%
def get_LR_SR_HR_positions(outputs_dir, step, n=0):
    LR_filename = outputs_dir + f'/LR_sample_{n}_{step}.npy'
    SR_filename = outputs_dir + f'/SR_sample_{n}_{step}.npy'
    HR_filename = outputs_dir + f'/HR_sample_{n}_{step}.npy'
    
    LR_positions = get_sample_positions(LR_filename)
    SR_positions = get_sample_positions(SR_filename)
    HR_positions = get_sample_positions(HR_filename)
    
    return LR_positions, SR_positions, HR_positions


def get_sample_positions(filename):
    
    sample = np.load(filename)
    
    N = sample.shape[-1]
    x = sample
    r = np.linspace(0, 1, N)
    X, Y, Z = np.meshgrid(r, r, r, indexing='ij')
    sample_xs = X + x[0, :, :, :]
    sample_ys = Y + x[1, :, :, :]
    sample_zs = Z + x[2, :, :, :]
    
    positions = np.stack((sample_xs.ravel(), 
                          sample_ys.ravel(), 
                          sample_zs.ravel()), 
                         axis=-1)
    
    return positions


#%%

def compute_power_spectrum(positions, particle_mass, box_size, grid_size):
    
    grid_size = int(grid_size)
    bins = (grid_size, grid_size, grid_size)
    bounds = [(0, box_size)] * 3
    density, edges = np.histogramdd(positions, bins=bins, range=bounds)
    cell_volume = (box_size / grid_size)**3
    density = particle_mass * density / cell_volume
    density /= np.mean(density)
    density -= 1
    # print('Density shape:', density.shape)
    
    # Get the fourier transform of the density field and
    # shift the zero-frequency component to the center.
    density_ft = fftn(density)
    density_ft = fftshift(density_ft)
    
    power_spectrum = np.abs(density_ft)**2
    # print('Power Spectrum Shape:', power_spectrum.shape)
    
    # Compute the frequency arrays
    kx = np.fft.fftfreq(grid_size, 1/grid_size)
    ky = np.fft.fftfreq(grid_size, 1/grid_size)
    kz = np.fft.fftfreq(grid_size, 1/grid_size)
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    
    # Radial bins
    k_bins = np.linspace(0, np.max(k), num=50)
    k_bin_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    power_spectrum_radial = np.zeros_like(k_bin_centers)
    
    # Average the power spectrum over spherical shells
    for i in range(len(k_bin_centers)):
        shell_mask = (k >= k_bins[i]) & (k < k_bins[i+1])
        power_spectrum_radial[i] = np.mean(power_spectrum[shell_mask])
        
    # print("Radial Power Spectrum shape:", power_spectrum_radial.shape)
    
    return k_bin_centers, power_spectrum_radial


#%%

def plot_spectra(meta_data, output_dir, step, save=False):
    
    box_size, patch_size, LR_size, HR_size, LR_mass, HR_mass = meta_data
    positions = get_LR_SR_HR_positions(output_dir, step)
    LR_positions, SR_positions, HR_positions = positions
    
    LR_cell_size = patch_size / LR_size
    HR_cell_size = LR_cell_size / 2
    HR_patch_size = HR_cell_size * HR_size
    
    LR_ks, LR_spectrum = compute_power_spectrum(
        LR_positions, LR_mass, patch_size, LR_size
    )
    SR_ks, SR_spectrum = compute_power_spectrum(
        SR_positions, HR_mass, HR_patch_size, HR_size
    )
    HR_ks, HR_spectrum = compute_power_spectrum(
        HR_positions, HR_mass, HR_patch_size, HR_size
    )
    
    # # Create a figure
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 8))
    # fontsize = 20
    
    # # LR scatter plot
    # ax1.plot(LR_ks, LR_spectrum)
    # ax1.set_title('LR')
    # ax1.set_xlabel('k')
    # ax1.set_ylabel('P(k)')
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    
    # # SR scatter plot
    # ax2.plot(SR_ks, SR_spectrum)
    # ax2.set_title('SR')
    # ax2.set_xlabel('k')
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    
    # # HR scatter plot
    # ax3.plot(HR_ks, HR_spectrum)
    # ax3.set_title('HR')
    # ax3.set_xlabel('k')
    # ax3.set_xscale('log')
    # ax3.set_yscale('log')
    
    # # Add title and adjust layout
    # fig.suptitle(f'Epoch {step}')
    # plt.tight_layout()
    
    plt.figure(figsize=(8, 8))
    plt.plot(HR_ks, HR_spectrum, label='HR', linewidth=4, color='red')
    plt.plot(LR_ks, LR_spectrum, label='LR', linewidth=4)
    plt.plot(SR_ks, SR_spectrum, label='SR', linewidth=4, color='black')
    plt.xscale('log')
    plt.yscale('log')
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