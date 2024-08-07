#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 13:28:51 2024

@author: brennan
"""

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from ...operations.particle_density import ngp_density_field
from ...operations.resizing import scale_up_data, crop_to_match

from .dmsr_generator import build_generator, build_latent_space_sampler
from .dmsr_critic import build_critic, CriticNoiseSampler


class DMSRGAN(keras.Model):
    
    def __init__(
            self,
            LR_grid_size, 
            scale_factor,
            HR_box_size,
            generator_channels=256,
            critic_channels=16,
            critic_steps=5,
            gp_weight=10.0,
            gp_rate=16,
            noise_std = 2,
            noise_epochs = 70,
        ):
        
        super().__init__()
        
        self.generator = build_generator(
            LR_grid_size, 
            scale_factor, 
            generator_channels
        )
        self.sampler = build_latent_space_sampler(self.generator)
        
        self.critic = build_critic(
            self.generator,
            critic_channels
        )
        self.noise_sampler = CriticNoiseSampler(
            self.critic, 
            noise_std, 
            noise_epochs
        )
        
        self.critic_steps  = critic_steps
        self.gp_weight     = gp_weight
        self.gp_rate       = gp_rate
        self.box_size      = HR_box_size
        
        self.batch_counter = tf.Variable(0, trainable=False, dtype=tf.float32)


    def compile(self, critic_optimizer, generator_optimizer):
        # TODO: Does this not need an optimizer passed to it? I think the
        # default optimizer is rmsprop.
        super().compile()
        self.critic_optimizer = critic_optimizer
        self.generator_optimizer = generator_optimizer
        
    
    def get_config(self):
        config = super(DMSRGAN, self).get_config()
        config.update({
            'generator'    : self.generator,
            'critic'       : self.critic,
            'critic_steps' : self.critic_steps,
            'gp_weight'    : self.gp_weight,
            'gp_rate'      : self.gp_rate,
            'box_size'     : self.box_size,
            'sampler'      : self.sampler
        })
        return config


    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    
    def supervised_dataset(self, dataset, batch_size):
        """
        Returns a new dataset with latent space samples added to the LR data
        from the given dataset. The new dataset can be used for supervised
        training of a generator to learn to predict the HR data from the LR
        data.
        """
        
        def add_latent_sample(LR_fields, HR_fields):
            lantent_variables = self.sampler(batch_size)
            LR_fields = (LR_fields, ) + lantent_variables
            return LR_fields, HR_fields
        
        return dataset.map(add_latent_sample)
    
    
    @tf.function
    def critic_loss(self, real_logits, fake_logits):
        real_loss = tf.reduce_mean(real_logits)
        fake_loss = tf.reduce_mean(fake_logits)
        return fake_loss - real_loss
    
    
    @tf.function
    def generator_loss(self, fake_logits):
        return -tf.reduce_mean(fake_logits)
    
    
    @tf.function
    def interpolate(self, real_data, fake_data):
        """
        """
        batch_size = tf.shape(real_data)[0]
        eps = tf.random.uniform([batch_size, 1, 1, 1, 1])
        diff = fake_data - real_data
        return real_data + eps * diff
    
    
    @tf.function
    def prepare_critic_data(self, data, US_data, batch_size):
        """
        Prepares the given data to be passed to the critic model.
        
        The data is augmented with noise, determined be the noise_sampler, and
        then concatenated with both the US data it's conditioned on and a 
        density field computed from the data.
        """
        data = data + self.noise_sampler(batch_size)
        density = ngp_density_field(data, self.box_size)
        data = tf.concat((density, data, US_data), axis=1)
        return data


    @tf.function
    def train_step(self, LR_HR_data):
        """
        Train step for the DMSR-GAN.
        """
        # Unpack and prepare the data batch for training:
        #   a. Scale up LR data to create US data which will be fed to the
        #      critic model. US data represents the LR data that the SR and HR
        #      data are conditioned on.
        #   b. Create density fields from the US data
        #   c. Concatenate the density fields with their respective US data.
        LR_data, HR_data = LR_HR_data
        US_data = scale_up_data(LR_data, scale=2)
        US_data = crop_to_match(US_data, HR_data)
        US_density = ngp_density_field(US_data, self.box_size)
        US_data = tf.concat((US_density, US_data), axis=1)
        
        # Train the critic model first and retain the critic loss.
        critic_loss, gp_loss = self.critic_train_step(
            LR_data, US_data, HR_data
        )
        losses = {
            "critic_loss"      : critic_loss,
            "generator_loss"   : 1.0,
            "gradient_penalty" : gp_loss
        }
           
        # Train the generator model and get the generator loss
        if tf.equal(self.batch_counter % self.critic_steps, 0.0):
            gen_loss = self.generator_train_step(LR_data, US_data)
            losses["generator_loss"] = gen_loss
        
        self.batch_counter.assign_add(1)
        return losses
    
    
    @tf.function
    def critic_train_step(self, LR_data, US_data, HR_data):
        """
        Train step for the critic.
        """
        batch_size = tf.shape(LR_data)[0]
        
        # Use the generator to generate SR data.
        noise = self.sampler(batch_size)
        generator_inputs = (LR_data, ) + noise
        SR_data = self.generator(generator_inputs)
        
        # Add the density and US fields to data to be passed to the critic.
        HR_data = self.prepare_critic_data(HR_data, US_data, batch_size)
        SR_data = self.prepare_critic_data(SR_data, US_data, batch_size)
        
        # Compute the critic loss
        with tf.GradientTape() as tape:
            fake_logits = self.critic(SR_data)
            real_logits = self.critic(HR_data)
            critic_loss = self.critic_loss(real_logits, fake_logits)
        
        # Get the critic gradients w.r.t the critic loss.
        critic_grad = tape.gradient(
            critic_loss, self.critic.trainable_variables
        )
        
        # Apply updates from the gradient penalty term.
        grad_pnlt = self.gradient_penalty(HR_data, SR_data)
            
        # Update the critic weights using the gradient of the critic loss.
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )
        
        return critic_loss, grad_pnlt
    
    
    @tf.function
    def gradient_penalty(self, HR_data, SR_data):
        """
        Compute the derivative of the gradient penalty term and update the
        weights of the critic accordingly.
        
        Note, as this function both calulates the gradient of the gradient
        penalty and applies the gradients to the critic weights, it needs to be
        used after the gradient of the critic loss is computed and before the
        critic's weights are updated with the gradients of the critic loss.
        """
        # Add gradient penalty term to the loss.
        if tf.equal(self.batch_counter % self.gp_rate, 0.0):
            
            # Create interpolated data for calulating the gradient penalty. 
            GP_data = self.interpolate(HR_data, SR_data)
            
            # Compute the gradient penalty term.
            with tf.GradientTape() as tape:
                gp_logits = self.critic(GP_data)
                gradients = tf.gradients(gp_logits, GP_data)
                grad_norm = tf.square(gradients)
                grad_norm = tf.reduce_sum(grad_norm, axis=[1, 2, 3, 4])
                # grad_norm = tf.sqrt(grad_norm)
                grad_pnlt = tf.reduce_mean((grad_norm - 1.0) ** 2)
                grad_pnlt = grad_pnlt * self.gp_weight
              
            # Update weights using the gradient of the gradient penalty term.
            gp_gradient = tape.gradient(
                grad_pnlt, self.critic.trainable_variables
            )
            self.critic_optimizer.apply_gradients(
                zip(gp_gradient, self.critic.trainable_variables)
            )
            
        else:
            grad_pnlt = 0.0
            
        return grad_pnlt
    
    
    @tf.function
    def generator_train_step(self, LR_data, US_data):
        """
        Train step for the generator.
        """
        # Generate noise for the generator.
        batch_size = tf.shape(LR_data)[0]
        noise = self.sampler(batch_size)
        generator_inputs = (LR_data, ) + noise
        
        # Calculate the generator loss.
        with tf.GradientTape() as tape:
            SR_data     = self.generator(generator_inputs)
            SR_data     = SR_data + self.noise_sampler(batch_size)
            SR_density  = ngp_density_field(SR_data, self.box_size)
            SR_data     = tf.concat((SR_density, SR_data, US_data), axis=1)
            fake_logits = self.critic(SR_data)
            gen_loss    = self.generator_loss(fake_logits)

        # Get the gradients w.r.t the generator loss and update the generator.
        gen_gradient = tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        self.generator_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        
        return gen_loss



class DMSRMonitor(keras.callbacks.Callback):
    
    def __init__(self, generator_noise, LR_samples, HR_samples):
        
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
    
    
    # def on_epoch_begin(self, epoch, logs=None):
    #     # TODO controlling the noise level should be done by a seperate
    #     # controller callback.
    #     self.model.noise_sampler.update()
    #     print(
    #         'Noise level is now', 
    #         self.model.noise_sampler.current_std.read_value()
    #     )
    
    
    def on_epoch_end(self, epoch, logs=None):
        
        epoch_average = self.critic_loss_epoch_total / self.critic_updates
        self.critic_epoch_loss.append(epoch_average)
        self.critic_epochs.append(self.critic_batches[-1])
        self.critic_loss_epoch_total = 0.0
        self.critic_updates = 0
        
        epoch_average = self.generator_loss_epoch_total / self.generator_updates
        self.generator_epoch_loss.append(epoch_average)
        self.generator_epochs.append(self.generator_batches[-1])
        self.generator_loss_epoch_total = 0.0
        self.generator_updates = 0
        
        epoch_average = self.grad_pnlt_loss_epoch_total / self.grad_pnlt_updates
        self.grad_pnlt_epoch_loss.append(epoch_average)
        self.grad_pnlt_epochs.append(self.grad_pnlt_batches[-1])
        self.grad_pnlt_loss_epoch_total = 0.0
        self.grad_pnlt_updates = 0
        
        if not (epoch % 10 == 0):
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
        self.critic_batch_loss.append(critic_loss)
        self.critic_batches.append(batch)
        self.critic_loss_epoch_total += critic_loss
        self.critic_updates += 1
        
        generator_loss = logs.get('generator_loss')
        if not generator_loss == 1.0:
            self.generator_batch_loss.append(generator_loss)
            self.generator_batches.append(batch)
            self.generator_loss_epoch_total += generator_loss
            self.generator_updates += 1
            
        grad_pnlt_loss = logs.get('generator_loss')
        if not grad_pnlt_loss == 0.0:
            self.grad_pnlt_batch_loss.append(grad_pnlt_loss)
            self.grad_pnlt_batches.append(batch)
            self.grad_pnlt_loss_epoch_total += grad_pnlt_loss
            self.grad_pnlt_updates += 1
