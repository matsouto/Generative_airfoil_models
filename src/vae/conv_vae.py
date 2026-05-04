import tensorflow as tf

from src.layers.sampling_layer import SamplingLayer
from src.scalers.airfoil_scaler import AirfoilScaler

from .conv_decoder import ConvDecoder
from .conv_encoder import ConvEncoder


class ConvCSTVariationalAutoencoder(tf.keras.Model):
    """CST VAE variant that encodes the airfoil coefficients with 1D convolutions."""

    def __init__(
        self,
        scaler: AirfoilScaler,
        npv=12,
        latent_dim=12,
    ):
        super().__init__()
        self.encoder = ConvEncoder(npv=npv, latent_dim=latent_dim)
        self.decoder = ConvDecoder(npv=npv, latent_dim=latent_dim)
        self.sampling = SamplingLayer()
        self.scaler = scaler

    def call(self, inputs, training=None):
        z_mean, z_log_var = self.encoder(inputs, training=training)
        z = self.sampling([z_mean, z_log_var], training=training)
        reconstructed = self.decoder(z, training=training)

        kl_per_example = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=1,
        )
        kl_loss = tf.reduce_mean(kl_per_example, axis=0)
        self.add_loss(kl_loss)

        return reconstructed
