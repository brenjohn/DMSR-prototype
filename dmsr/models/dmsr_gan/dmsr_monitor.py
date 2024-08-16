#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:07:54 2024

@author: john
"""

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras


class DMSRMonitor(keras.callbacks.Callback):
    
    def __init__(
            self, 
            generator_noise, 
            LR_samples, 
            HR_samples, 
            output_rate=10
        ):
        
        self.noise = generator_noise
        self.LR_samples = LR_samples
        self.HR_samples = HR_samples
        self.num_samples = tf.shape(LR_samples)[0]
        self.data_dir = 'data/training_outputs/'
        
        self.critic_epoch_loss = []
        self.critic_batch_loss = []
        self.critic_batches = []
        self.critic_epochs = []
        self.critic_loss_epoch_total = 0.0
        self.critic_updates = 0
        
        self.generator_epoch_loss = []
        self.generator_batch_loss = []
        self.generator_batches = []
        self.generator_epochs = []
        self.generator_loss_epoch_total = 0.0
        self.generator_updates = 0
        
        self.grad_pnlt_epoch_loss = []
        self.grad_pnlt_batch_loss = []
        self.grad_pnlt_batches = []
        self.grad_pnlt_epochs = []
        self.grad_pnlt_loss_epoch_total = 0.0
        self.grad_pnlt_updates = 0
        
        self.batch = 0
        self.epoch = 0
        self.output_rate = output_rate
    
    
    def on_epoch_end(self, epoch, logs=None):
        
        self.epoch += 1
        epoch = self.epoch
        
        # Compute the critic loss epoch average.
        average = self.critic_loss_epoch_total / self.critic_updates
        self.critic_epoch_loss.append(average)
        self.critic_epochs.append(self.critic_batches[-1])
        self.critic_loss_epoch_total = 0.0
        self.critic_updates = 0
        
        # Compute the generator loss epoch average.
        average = self.generator_loss_epoch_total / self.generator_updates
        self.generator_epoch_loss.append(average)
        self.generator_epochs.append(self.generator_batches[-1])
        self.generator_loss_epoch_total = 0.0
        self.generator_updates = 0
        
        # Compute the gradient penalty epoch average.
        average = self.grad_pnlt_loss_epoch_total / self.grad_pnlt_updates
        self.grad_pnlt_epoch_loss.append(average)
        self.grad_pnlt_epochs.append(self.grad_pnlt_batches[-1])
        self.grad_pnlt_loss_epoch_total = 0.0
        self.grad_pnlt_updates = 0
        
        if not (epoch % self.output_rate == 0):
            return
        
        generator_inputs = (self.LR_samples,) + self.noise
        SR_samples = self.model.generator(generator_inputs)
        
        output_dir = self.data_dir + f'step_{epoch:04}/'
        os.makedirs(output_dir, exist_ok=True)

        for i in range(self.num_samples):
            LR_sample = self.LR_samples[i].numpy()
            SR_sample = SR_samples[i].numpy()
            HR_sample = self.HR_samples[i].numpy()
            np.save(output_dir + f'SR_sample_{i}_{epoch}.npy', SR_sample)
            np.save(output_dir + f'LR_sample_{i}_{epoch}.npy', LR_sample)
            np.save(output_dir + f'HR_sample_{i}_{epoch}.npy', HR_sample)
            
    
    def on_batch_end(self, batch, logs=None):
        
        self.batch += 1
        batch = self.batch
        
        critic_loss = logs.get('critic_loss')
        if not critic_loss == 2.0:
            self.critic_batch_loss.append(critic_loss)
            self.critic_batches.append(batch)
            self.critic_loss_epoch_total += critic_loss
            self.critic_updates += 1
        
        generator_loss = logs.get('generator_loss')
        if not generator_loss == 2.0:
            self.generator_batch_loss.append(generator_loss)
            self.generator_batches.append(batch)
            self.generator_loss_epoch_total += generator_loss
            self.generator_updates += 1
            
        grad_pnlt_loss = logs.get('gradient_penalty')
        if not grad_pnlt_loss == -1.0:
            self.grad_pnlt_batch_loss.append(grad_pnlt_loss)
            self.grad_pnlt_batches.append(batch)
            self.grad_pnlt_loss_epoch_total += grad_pnlt_loss
            self.grad_pnlt_updates += 1
