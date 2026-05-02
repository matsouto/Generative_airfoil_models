import tensorflow as tf

from src.scalers.airfoil_scaler import AirfoilScaler
from src.layers.sampling_layer import SamplingLayer

from .decoder import FiLMDecoder
from .encoder import FiLMEncoder


class FiLMCSTVariationalAutoencoder(tf.keras.Model):
    """Conditional CST VAE that applies FiLM using Cl and alpha."""

    def __init__(
        self,
        scaler: AirfoilScaler,
        npv=8,
        latent_dim=16,
        film_depth=2,
    ):
        super().__init__()
        self.encoder = FiLMEncoder(
            npv=npv,
            latent_dim=latent_dim,
            film_depth=film_depth,
        )
        self.decoder = FiLMDecoder(
            npv=npv,
            latent_dim=latent_dim,
            film_depth=film_depth,
        )
        self.sampling = SamplingLayer()
        self.scaler = scaler

    def _split_inputs(self, inputs):
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError(
                "FiLMCSTVariationalAutoencoder expects inputs=(geometry, condition)."
            )

        return inputs

    def call(self, inputs, training=None):
        geometry, condition = self._split_inputs(inputs)

        z_mean, z_log_var = self.encoder((geometry, condition), training=training)
        z = self.sampling([z_mean, z_log_var], training=training)
        reconstructed = self.decoder((z, condition), training=training)

        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        self.add_loss(kl_loss)

        return reconstructed


if __name__ == "__main__":
    BATCH_SIZE = 4
    NPV = 8
    LATENT_DIM = 16

    scaler = AirfoilScaler()
    vae = FiLMCSTVariationalAutoencoder(
        scaler=scaler,
        npv=NPV,
        latent_dim=LATENT_DIM,
        film_depth=2,
    )

    dummy_geometry = tf.random.normal([BATCH_SIZE, (2 * NPV) + 2])
    dummy_condition = tf.random.uniform([BATCH_SIZE, 2], minval=-1.0, maxval=1.0)

    weights, params = vae((dummy_geometry, dummy_condition))

    print("--- FiLM VAE ---")
    print(f"Geometry shape: {dummy_geometry.shape}")
    print(f"Condition shape: {dummy_condition.shape}")
    print(f"Reconstructed Weights shape: {weights.shape}")
    print(f"Reconstructed Parameters shape: {params.shape}")
    print(f"Total losses (KL): {len(vae.losses)}")
