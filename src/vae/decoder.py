import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Reshape  # type: ignore
from ..layers.cst_layer import CSTLayer


class Decoder(tf.keras.Model):  # type: ignore
    """
    A simple Dense Decoder.
    Directly maps latent code z -> CST coefficients.
    """

    def __init__(self, npv=12, latent_dim=16):
        super().__init__()
        self.npv = npv
        self.latent_dim = latent_dim

        # Calculate output size: (2 surfaces * NPV)
        self.num_weights = 2 * self.npv

        # MLP
        self.dense1 = Dense(512)
        self.bn1 = BatchNormalization()
        self.act1 = ReLU()

        self.dense2 = Dense(256)
        self.bn2 = BatchNormalization()
        self.act2 = ReLU()

        # --- Output Heads ---

        # Head 1: Weights (Range -1 to 1 via Tanh)
        self.dense_weights = Dense(self.num_weights, activation="tanh")
        self.reshape_weights = Reshape((2, self.npv))

        # Head 2: Parameters (Range -1 to 1 via Tanh)
        self.dense_params = Dense(
            2, activation="tanh"
        )  # 2 parameters: thickness and camber

        # CST Layer for coordinate generation (Same as before)
        self.cst_transform = CSTLayer(num_weights=self.npv)

    def call(self, z, training=None):
        x = self.dense1(z)
        # x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.dense2(x)
        # x = self.bn2(x, training=training)
        x = self.act2(x)

        # Generate raw flat weights
        weights_flat = self.dense_weights(x)
        weights = self.reshape_weights(weights_flat)

        # Generate parameters
        parameters = self.dense_params(x)

        # Returning coords=None during training to save speed
        return weights, parameters


if __name__ == "__main__":
    BATCH_SIZE = 4
    NPV = 12
    LATENT_DIM = 128

    # 1. Instantiate the Decoder directly
    decoder = Decoder(npv=NPV, latent_dim=LATENT_DIM)

    # 2. Create a batch of dummy latent vectors (the decoder's input)
    dummy_latent_vector = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    # 3. Pass the latent vector through the decoder
    weights, params = decoder(dummy_latent_vector)

    # 4. Check the output shapes
    print("\n--- Decoder Test ---")
    print(f"Input shape (latent vector): {dummy_latent_vector.shape}")
    # print(f"Output Coordinates: {coords}")
    print(f"Output Weights shape: {weights.shape}")
    print(f"Output Parameters shape: {params.shape}")

    # Expected output shapes
    assert weights.shape == (BATCH_SIZE, 2, NPV)
    assert params.shape == (BATCH_SIZE, 2)
    # The coordinate shape depends on the CSTLayer's point density
    print("\nDecoder shapes are correct!")
