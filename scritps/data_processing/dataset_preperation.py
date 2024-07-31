#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 19:02:21 2024

@author: brennan
"""

import sys
import glob
sys.path.append("..")
sys.path.append("../..")

import h5py as h5
import numpy as np

from dmsr.swift_processing import get_displacement_field
from dmsr.operations.resizing import cut_fields


def get_displacement_fields(snapshots):
    displacement_fields = []
    
    for snap in snapshots:
        data = h5.File(snap, 'r')
        grid_size   = data['ICs_parameters'].attrs['Grid Resolution']
        box_size    = data['Header'].attrs['BoxSize'][0]
        IDs         = np.asarray(data['DMParticles']['ParticleIDs'])
        coordinates = np.asarray(data['DMParticles']['Coordinates'])
        coordinates = coordinates.transpose()
        displacement_fields.append(
            get_displacement_field(coordinates, IDs, box_size, grid_size)
        )
        data.close()
        
    return np.stack(displacement_fields), box_size, grid_size


#%%
data_directory = '../../data/dmsr_runs/'

LR_snapshots = np.sort(glob.glob(data_directory + '*/064/snap_0002.hdf5'))
HR_snapshots = np.sort(glob.glob(data_directory + '*/128/snap_0002.hdf5'))


#%%
LR_fields, box_size, LR_grid_size = get_displacement_fields(LR_snapshots)
HR_fields, box_size, HR_grid_size = get_displacement_fields(HR_snapshots)


#%%
LR_fields = cut_fields(LR_fields, 32, LR_grid_size)
HR_fields = cut_fields(HR_fields, 64, HR_grid_size)


#%%
LR_file = '../../data/dmsr_training/LR_fields.npy'
np.save(LR_file, LR_fields)

HR_file = '../../data/dmsr_training/HR_fields.npy'
np.save(HR_file, HR_fields)

meta_file = '../../data/dmsr_training/metadata.npy'
np.save(meta_file, [box_size/2, LR_grid_size//2, HR_grid_size//2])