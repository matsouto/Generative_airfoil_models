import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Flatten,
    Concatenate,
    Dense,
    BatchNormalization,
    LeakyReLU,
    Conv2D,
    Reshape,
)


class Encoder(tf.keras.Model):
    """
    It uses Conv2D layers to process the airfoil weights.
    """

    def __init__(self, npv=12, latent_dim=128):
        super().__init__()
        self.npv = npv
        self.latent_dim = latent_dim

        self.reshape_input = Reshape((2, self.npv, 1))

        # Convolutional Stack
        # Layer 1: 32 filters
        self.conv1 = Conv2D(32, kernel_size=(2, 3), strides=(1, 2), padding="same")
        self.bn1 = BatchNormalization()
        self.leaky_relu1 = LeakyReLU(0.2)

        # Layer 2: 64 filters
        self.conv2 = Conv2D(64, kernel_size=(2, 3), strides=(1, 2), padding="same")
        self.bn2 = BatchNormalization()
        self.leaky_relu2 = LeakyReLU(0.2)

        # Layer 3: 128 filters
        self.conv3 = Conv2D(128, kernel_size=(2, 3), strides=(1, 1), padding="same")
        self.bn3 = BatchNormalization()
        self.leaky_relu3 = LeakyReLU(0.2)

        self.flatten = Flatten()
        self.concat = Concatenate()

        # Dense head for latent space
        self.dense1 = Dense(256)
        self.bn4 = BatchNormalization()
        self.leaky_relu4 = LeakyReLU(0.2)

        self.dense_mean = Dense(self.latent_dim, name="z_mean")
        self.dense_log_var = Dense(self.latent_dim, name="z_log_var")

    def call(self, inputs):

        # Split input into weights (for conv layers) and parameters (for later)
        weights, parameters = tf.split(inputs, [2 * self.npv, 2], axis=1)

        # --- Convolutional Path for Weights ---
        x = self.reshape_input(weights)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu3(x)

        x = self.flatten(x)

        # --- Combine and Final Dense Layers ---
        # Concatenate flattened weight features with the global airfoil parameters
        x = self.concat([x, parameters])

        x = self.dense1(x)
        x = self.bn4(x)
        x = self.leaky_relu4(x)

        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)

        return z_mean, z_log_var


if __name__ == "__main__":
    BATCH_SIZE = 4
    NPV = 12
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
    print("--- Encoder Test ✅ ---")
    print(f"Input shape: {encoder_input.shape}")
    print(f"Output z_mean shape: {z_mean.shape}")
    print(f"Output z_log_var shape: {z_log_var.shape}")

    # Expected output shape for z_mean and z_log_var is (4, 128)
    assert z_mean.shape == (BATCH_SIZE, LATENT_DIM)
    assert z_log_var.shape == (BATCH_SIZE, LATENT_DIM)
    print("\nEncoder shapes are correct!")
