#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 13:28:51 2024

@author: brennan

This file defines the DMSR-GAN (Dark Matter Super Resolution - GAN).
"""

import tensorflow as tf
from tensorflow import keras

from ...operations.particle_density import ngp_density_field
from ...operations.resizing import scale_up_data, crop_to_match

from .dmsr_generator import build_generator, build_latent_space_sampler
from .dmsr_critic import build_critic, CriticNoiseSampler

# TODO: Remove the noise sampler for critic inputs. I don't think this is used
# for balancing WGAN training. It is used for regular GAN training.

# TODO: Use a factory pattern to build the components for the DMSRGAN and then
# instantiate one. The constructor for the DMSRGAN should only take is 
# attributes as input. Also, training related parameters should be set in the 
# DMSRGAN's compile method.

def build_dmsrgan(
        LR_grid_size = 16, 
        scale_factor = 2,
        HR_box_size = 26.6,
        generator_channels=256,
        critic_channels=16,
        critic_steps=5,
        generator_steps=1,
        gp_weight=10.0,
        gp_rate=16,
        noise_std = 0,
        noise_epochs = 70,
        **kwargs
    ):
    """
    A factory function for building a DMSRGAN and its components.
    """
    # Build the generator model.
    generator = build_generator(LR_grid_size, scale_factor, generator_channels)
    sampler = build_latent_space_sampler(generator)
    
    # Build the critic model
    critic = build_critic(generator, critic_channels)
    noise_sampler = CriticNoiseSampler(critic, noise_std, noise_epochs)
    
    return DMSRGAN(generator, critic, sampler, noise_sampler)



@keras.utils.register_keras_serializable()
class DMSRGAN(keras.Model):
    """
    This class defines a WGAN-GP style model for enhancing the resolution of
    a dark matter displacement field.
    
    Attributes:
        - generator     : The WGAN generator model.
        - critic        : The WGAN critic model.
        - sampler       : A callable for sampling the generator latent space.
        - noise_sampler : A callable for sampling noise for the critic input.
        - critic_steps  : Number of critic updates before a generator update.
        - gp_weight     : The weight for the gradient penalty term.
        - gp_rate       : Gradient penalty is computed every 'gp_rate' steps. 
        - box_size      : The size of the upscaled box in cm Mpc.
    """
        
    def __init__(
            self,
            LR_grid_size = 16, 
            scale_factor = 2,
            HR_box_size = 26.6,
            generator_channels=256,
            critic_channels=16,
            critic_steps=5,
            generator_steps=1,
            gp_weight=10.0,
            gp_rate=16,
            noise_std = 0,
            noise_epochs = 70,
            **kwargs
        ):
        
        super().__init__()
        
        components = self.build_components(
            LR_grid_size, 
            scale_factor, 
            generator_channels, 
            critic_channels, 
            noise_std,
            noise_epochs
        )
        generator, sampler, critic, noise_sampler = components
        
        self.generator       = generator
        self.sampler         = sampler
        self.critic          = critic
        self.noise_sampler   = noise_sampler
        self.critic_steps    = critic_steps
        self.generator_steps = generator_steps
        self.gp_weight       = gp_weight
        self.gp_rate         = gp_rate
        self.box_size        = HR_box_size
        
        self.batch_counter = tf.Variable(0, trainable=False, dtype=tf.float32)
        
        
    @classmethod
    def build_components(
            cls,
            LR_grid_size, 
            scale_factor,
            generator_channels=256,
            critic_channels=16,
            noise_std = 2,
            noise_epochs = 70
        ):
        
        generator = build_generator(
            LR_grid_size, scale_factor, generator_channels
        )
        critic = build_critic(generator, critic_channels)
        noise_sampler = CriticNoiseSampler(critic, noise_std, noise_epochs)
        sampler = build_latent_space_sampler(generator)
        
        return generator, sampler, critic, noise_sampler


    # =========================================================================
    #                         Training Methods
    # =========================================================================

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
        
        losses = {
            "critic_loss"      : 2.0,
            "generator_loss"   : 2.0,
            "gradient_penalty" : -1.0
        }
        
        # Train the critic model first and retain the critic loss.
        if tf.equal(self.batch_counter % self.generator_steps, 0.0):
            critic_loss, gp_loss = self.critic_train_step(
                LR_data, US_data, HR_data
            )
            losses["critic_loss"] = critic_loss
            losses["gradient_penalty"] = gp_loss
           
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
    
    
    # =========================================================================
    #                      Training Utility Methods
    # =========================================================================
    
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
        Returns a tensor obtained by linearly interpolating between the given
        tensors a random amount. This is used for computing the gradient 
        penalty term during training.
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
    
    
    # =========================================================================
    #                          Utility Methods
    # =========================================================================

    def compile(self, critic_optimizer, generator_optimizer):
        super().compile()
        self.critic_optimizer = critic_optimizer
        self.generator_optimizer = generator_optimizer
        
    
    def get_config(self):
        config = super(DMSRGAN, self).get_config()
        config.update({
            'generator'     : self.generator.get_config(),
            'critic'        : self.critic.get_config(),
            'critic_steps'  : self.critic_steps,
            'gp_weight'     : self.gp_weight,
            'gp_rate'       : self.gp_rate,
            'box_size'      : self.box_size,
            'noise_sampler' : self.noise_sampler.get_config(),
            'batch_counter' : self.batch_counter.numpy()
        })
        return config


    @classmethod
    def from_config(cls, config):
        dmsr_gan = cls(**config)
        dmsr_gan.set_config(config)
        return dmsr_gan
    
    
    def set_config(self, config):
        config['noise_sampler'] = CriticNoiseSampler.from_config(
            config['noise_sampler']
        )
        
        self.noise_sampler = config['noise_sampler']
        self.critic_steps  = config['critic_steps']
        self.gp_weight     = config['gp_weight']
        self.gp_rate       = config['gp_rate']
        self.box_size      = config['box_size']
        self.batch_counter = tf.Variable(
            config['batch_counter'], trainable=False, dtype=tf.float32
        )
        
        sampler = build_latent_space_sampler(self.generator)
        self.sampler = sampler
        
    
    def build(self):
        self.generator.build(self.generator.input)
        self.critic.build(self.critic.input)
        
        
    def create_checkpoint(self, checkpoint_prefix):
        checkpoint = tf.train.Checkpoint(
            generator           = self.generator,
            critic              = self.critic,
            generator_optimizer = self.generator_optimizer,
            critic_optimizer    = self.critic_optimizer,
            batch_counter       = self.batch_counter
        )
        return checkpoint
    

    
class DMSRGANCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint, checkpoint_prefix):
        super(DMSRGANCheckpoint, self).__init__()
        self.checkpoint = checkpoint
        self.checkpoint_prefix = checkpoint_prefix

    def on_epoch_end(self, epoch, logs=None):
        file_prefix=self.checkpoint_prefix.format(epoch=epoch)
        self.checkpoint.save(file_prefix=file_prefix)