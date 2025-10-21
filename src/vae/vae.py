import tensorflow as tf
from src.layers.sampling_layer import SamplingLayer
from . import Encoder
from . import Decoder


class CSTVariationalAutoencoder(tf.keras.Model):
    """Combines the new Encoder and Decoder."""

    def __init__(self, npv=12, latent_dim=128, use_modifications=True):
        super().__init__()
        self.encoder = Encoder(npv=npv, latent_dim=latent_dim)
        self.decoder = Decoder(
            npv=npv, latent_dim=latent_dim, use_modifications=use_modifications
        )
        self.sampling = SamplingLayer()

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = self.sampling([z_mean, z_log_var])
        reconstructed = self.decoder(z)

        # Adding additional KL divergence loss to the model
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        self.add_loss(kl_loss)

        return reconstructed


# --- Main Execution Block ---
if __name__ == "__main__":
    BATCH_SIZE = 4
    NPV = 12
    LATENT_DIM = 128

    vae = CSTVariationalAutoencoder(npv=NPV, latent_dim=LATENT_DIM)

    # Create dummy input data
    dummy_weights = tf.random.normal([BATCH_SIZE, 2 * NPV])
    dummy_params = tf.random.uniform([BATCH_SIZE, 2])
    vae_input = tf.concat([dummy_weights, dummy_params], axis=1)

    # Pass data through the VAE
    coords, weights, params = vae(vae_input)

    print("--- VAE ---")
    print(f"Input shape: {vae_input.shape}")
    print(f"Reconstructed Coordinates shape: {coords.shape}")
    print(f"Reconstructed Weights shape: {weights.shape}")
    print(f"Reconstructed Parameters shape: {params.shape}")
    print(f"Total losses (KL): {len(vae.losses)}")
