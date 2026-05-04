import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    LeakyReLU,
    Flatten,
    Concatenate,
)


class Encoder(tf.keras.Model):
    """
    A simple Dense Encoder.
    Ideal for fixed-size parameter vectors like CST coefficients.
    """

    def __init__(self, npv=12, latent_dim=16):  # Reduced latent_dim to 16
        super().__init__()
        self.npv = npv
        self.latent_dim = latent_dim

        # Flatten input immediately
        self.flatten = Flatten()

        # Simple MLP (Multi-Layer Perceptron)
        self.dense1 = Dense(256)
        self.bn1 = BatchNormalization()
        self.act1 = LeakyReLU(0.2)

        self.dense2 = Dense(512)
        self.bn2 = BatchNormalization()
        self.act2 = LeakyReLU(0.2)

        # Latent Space Heads
        self.dense_mean = Dense(self.latent_dim, name="z_mean")
        self.dense_log_var = Dense(self.latent_dim, name="z_log_var")

    def call(self, inputs):
        # Input shape: (Batch, 2, 12) or (Batch, 26) depending on how you pass it
        x = self.flatten(inputs)

        x = self.dense1(x)
        # x = self.bn1(x)
        x = self.act1(x)

        x = self.dense2(x)
        # x = self.bn2(x)
        x = self.act2(x)

        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)

        return z_mean, z_log_var


if __name__ == "__main__":
    BATCH_SIZE = 4
    NPV = 8
    LATENT_DIM = 128

    # 1. Instantiate the Encoder directly
    encoder = Encoder(npv=NPV, latent_dim=LATENT_DIM)

    # 2. Create a batch of dummy input data (same as the VAE input)
    dummy_weights = tf.random.normal([BATCH_SIZE, 2 * NPV])
    dummy_params = tf.random.uniform([BATCH_SIZE, 2])
    encoder_input = tf.concat([dummy_weights, dummy_params], axis=1)

    # 3. Pass the data through the encoder
    z_mean, z_log_var = encoder(encoder_input)

    # 4. Check the output shapes
    print("--- Encoder Test ---")
    print(f"Input shape: {encoder_input.shape}")
    print(f"Output z_mean shape: {z_mean.shape}")
    print(f"Output z_log_var shape: {z_log_var.shape}")

    # Expected output shape for z_mean and z_log_var is (4, 128)
    assert z_mean.shape == (BATCH_SIZE, LATENT_DIM)
    assert z_log_var.shape == (BATCH_SIZE, LATENT_DIM)
    print("\nEncoder shapes are correct!")
