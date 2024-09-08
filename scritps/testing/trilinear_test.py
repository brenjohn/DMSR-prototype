#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:18:42 2024

@author: john
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import tensorflow as tf
from dmsr.operations.resizing import trilinear_scaling


#%% Make some fake data to rescale.
data = tf.random.normal((1, 3, 4, 4, 4))

#%% Rescale fake data.
scale = 2
x = trilinear_scaling(data, scale)

#%% Plot comparison.
import matplotlib.pyplot as plt

d = data.numpy()
D = x.numpy()

plt.imshow(d[0, 2, :, :, 0])
plt.show()
plt.close()

plt.imshow(D[0, 2, :, :, 0])
plt.show()
plt.close()