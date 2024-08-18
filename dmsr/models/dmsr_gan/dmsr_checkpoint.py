#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 11:21:36 2024

@author: john

This file defines the DMSRGAN-Checkpoint class for saving DMSRGAN checkpoints.
"""

import tensorflow as tf


class DMSRGANCheckpoint(tf.keras.callbacks.Callback):
    """
    A keras callback class for saving DMSRGAN checkpoints. Checkpoints store
    the states of the generator, critic, the optimizers for both of these and
    the batch_counter attributes of the given DMSRGAN.
    """
    
    def __init__(self, dmsr_gan, checkpoint_prefix):
        super(DMSRGANCheckpoint, self).__init__()
        
        checkpoint = tf.train.Checkpoint(
            generator           = dmsr_gan.generator,
            critic              = dmsr_gan.critic,
            generator_optimizer = dmsr_gan.generator_optimizer,
            critic_optimizer    = dmsr_gan.critic_optimizer,
            batch_counter       = dmsr_gan.batch_counter
        )
        self.checkpoint = checkpoint
        self.checkpoint_prefix = checkpoint_prefix


    def on_epoch_end(self, epoch, logs=None):
        """
        Save a checkpoint.
        """
        file_prefix=self.checkpoint_prefix.format(epoch=epoch)
        self.checkpoint.save(file_prefix=file_prefix)
        
        
    def restore(self, checkpoint_name):
        """
        Restore the state of the tracked DMSRGAN attributes from the given
        checkpoint.
        """
        self.checkpoint.restore(checkpoint_name)