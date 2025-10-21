import tensorflow as tf
from tensorflow.keras.layers import Layer
from src.airfoil import airfoil_modifications
import numpy as np


class SamplingLayer(Layer):
    def __init__(self, **kwargs):
        super(SamplingLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """
        Args:
            inputs: A tuple of (z_mean, z_log_var)
                - z_mean: A 2D tensor of shape (batch_size, latent_dim)
                - z_log_var: A 2D tensor of shape (batch_size, latent_dim)
        Returns:
            A 2D tensor of shape (batch_size, latent_dim) representing the sampled latent vector.
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        std_dev = tf.exp(0.5 * z_log_var)
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + std_dev * epsilon
