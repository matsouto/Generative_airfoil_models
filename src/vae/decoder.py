import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    ReLU,
    Conv2DTranspose,
    Reshape,
)
from ..layers.cst_layer import CSTLayer


class Decoder(tf.keras.Model):
    """
    Uses Conv2DTranspose to generate weights and a parallel Dense network for parameters.
    """

    def __init__(self, npv=12, latent_dim=128, use_modifications=True):
        super().__init__()
        self.npv = npv
        self.latent_dim = latent_dim
        self.use_modifications = use_modifications

        # We need to calculate the shape before the flatten operation in the encoder
        # For this architecture, after the conv layers, the shape is (2, 3, 128)
        self.dense_start_shape = (2, 3, 128)
        dense_units = (
            self.dense_start_shape[0]
            * self.dense_start_shape[1]
            * self.dense_start_shape[2]
        )

        # --- Branch 1: Convolutional Path for Weights ---
        self.dense1 = Dense(dense_units)
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.reshape = Reshape(self.dense_start_shape)

        # Deconvolutional Stack
        # Layer 1: 64 filters
        self.deconv1 = Conv2DTranspose(
            64, kernel_size=(2, 3), strides=(1, 1), padding="same"
        )
        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()

        # Layer 2: 32 filters
        self.deconv2 = Conv2DTranspose(
            32, kernel_size=(2, 3), strides=(1, 2), padding="same"
        )
        self.bn3 = BatchNormalization()
        self.relu3 = ReLU()

        # Layer 3: Output layer, 1 filter for the final weights
        # Using tanh activation for output in range [-1, 1]
        self.deconv3 = Conv2DTranspose(
            1, kernel_size=(2, 3), strides=(1, 2), padding="same", activation="tanh"
        )
        self.final_reshape = Reshape((2, self.npv))

        # --- Branch 2: Dense Path for LEM and TET Parameters ---
        self.dense_p1 = Dense(32)
        self.bn_p1 = BatchNormalization()
        self.relu_p1 = ReLU()

        self.dense_p2 = Dense(2, activation="tanh")

        # Class-Shape Transformation Layer
        self.cst_transform = CSTLayer()

    def call(self, z):  # Input is the latent vector z
        # --- Branch 1: Generate CST weights ---
        x = self.dense1(z)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.reshape(x)

        x = self.deconv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.deconv2(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.deconv3(x)

        weights = self.final_reshape(x)

        # --- Branch 2: Generate LEM and TET parameters ---
        p = self.dense_p1(z)
        p = self.bn_p1(p)
        p = self.relu_p1(p)
        parameters = self.dense_p2(p)

        if not self.use_modifications:
            parameters = tf.zeros_like(parameters)

        # Combine to get final coordinates
        # coordinates = self.cst_transform(weights, parameters)
        coordinates = None
        return coordinates, weights, parameters


# Assume the Decoder class and other dependencies are defined above

if __name__ == "__main__":
    BATCH_SIZE = 4
    NPV = 12
    LATENT_DIM = 128

    # 1. Instantiate the Decoder directly
    decoder = Decoder(npv=NPV, latent_dim=LATENT_DIM)

    # 2. Create a batch of dummy latent vectors (the decoder's input)
    dummy_latent_vector = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    # 3. Pass the latent vector through the decoder
    coords, weights, params = decoder(dummy_latent_vector)

    # 4. Check the output shapes
    print("\n--- Decoder Test ✅ ---")
    print(f"Input shape (latent vector): {dummy_latent_vector.shape}")
    print(f"Output Coordinates shape: {coords.shape}")
    print(f"Output Weights shape: {weights.shape}")
    print(f"Output Parameters shape: {params.shape}")

    # Expected output shapes
    assert weights.shape == (BATCH_SIZE, 2, NPV)
    assert params.shape == (BATCH_SIZE, 2)
    # The coordinate shape depends on the CSTLayer's point density
    print("\nDecoder shapes are correct!")
