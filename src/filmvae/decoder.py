import tensorflow as tf
from tensorflow.keras.layers import Add, Dense, LeakyReLU, Multiply, Reshape

from ..layers.cst_layer import CSTLayer


class FiLMDecoder(tf.keras.Model):
    """Decoder that reconstructs CST geometry conditioned by Cl and alpha via FiLM."""

    def __init__(self, npv=8, latent_dim=16):
        super().__init__()
        self.npv = npv
        self.latent_dim = latent_dim
        self.num_weights = 2 * self.npv

        self.latent_dense1 = Dense(512)
        self.gamma1 = Dense(512)
        self.beta1 = Dense(512)
        self.multiply1 = Multiply()
        self.add1 = Add()
        self.act1 = LeakyReLU(negative_slope=0.2)

        self.latent_dense2 = Dense(256)
        self.gamma2 = Dense(256)
        self.beta2 = Dense(256)
        self.multiply2 = Multiply()
        self.add2 = Add()
        self.act2 = LeakyReLU(negative_slope=0.2)

        self.dense_weights = Dense(self.num_weights, activation="tanh")
        self.reshape_weights = Reshape((2, self.npv))
        self.dense_params = Dense(2)

        self.cst_transform = CSTLayer(num_weights=self.npv)

    def build(self, input_shape):
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
            raise ValueError(
                "FiLMDecoder build expects input_shape=(latent_shape, condition_shape)."
            )

        latent_shape, condition_shape = input_shape
        hidden1_shape = tf.TensorShape([None, 512])
        hidden2_shape = tf.TensorShape([None, 256])

        self.latent_dense1.build(latent_shape)
        self.gamma1.build(condition_shape)
        self.beta1.build(condition_shape)

        self.latent_dense2.build(hidden1_shape)
        self.gamma2.build(condition_shape)
        self.beta2.build(condition_shape)

        self.dense_weights.build(hidden2_shape)
        self.reshape_weights.build(tf.TensorShape([None, self.num_weights]))
        self.dense_params.build(hidden2_shape)

        self.cst_transform(
            tf.zeros((1, 2, self.npv), dtype=tf.float32),
            tf.zeros((1, 2), dtype=tf.float32),
        )

        super().build(input_shape)

    def _split_inputs(self, inputs):
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError("FiLMDecoder expects inputs=(latent, condition).")

        return inputs

    def _apply_film(
        self,
        x,
        condition,
        gamma_layer,
        beta_layer,
        multiply_layer,
        add_layer,
        activation,
    ):
        gamma = gamma_layer(condition)
        beta = beta_layer(condition)
        x = multiply_layer([x, gamma])
        x = add_layer([x, beta])
        return activation(x)

    def call(self, inputs, training=None):
        del training
        latent, condition = self._split_inputs(inputs)

        x = self.latent_dense1(latent)
        x = self._apply_film(
            x,
            condition,
            self.gamma1,
            self.beta1,
            self.multiply1,
            self.add1,
            self.act1,
        )

        x = self.latent_dense2(x)
        x = self._apply_film(
            x,
            condition,
            self.gamma2,
            self.beta2,
            self.multiply2,
            self.add2,
            self.act2,
        )

        weights_flat = self.dense_weights(x)
        weights = self.reshape_weights(weights_flat)
        params = self.dense_params(x)

        return weights, params


if __name__ == "__main__":
    BATCH_SIZE = 4
    NPV = 8
    LATENT_DIM = 16

    decoder = FiLMDecoder(npv=NPV, latent_dim=LATENT_DIM)

    dummy_latent = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    dummy_condition = tf.random.uniform([BATCH_SIZE, 2], minval=-1.0, maxval=1.0)

    weights, params = decoder((dummy_latent, dummy_condition))

    print("--- FiLM Decoder Test ---")
    print(f"Latent shape: {dummy_latent.shape}")
    print(f"Condition shape: {dummy_condition.shape}")
    print(f"Output Weights shape: {weights.shape}")
    print(f"Output Parameters shape: {params.shape}")

    assert weights.shape == (BATCH_SIZE, 2, NPV)
    assert params.shape == (BATCH_SIZE, 2)
    print("\nDecoder shapes are correct!")
