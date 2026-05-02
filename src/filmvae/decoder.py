import tensorflow as tf
from tensorflow.keras.layers import Add, Dense, LeakyReLU, Multiply, Reshape

from ..layers.cst_layer import CSTLayer
from .encoder import DEFAULT_FILM_HIDDEN_DIMS


class FiLMDecoder(tf.keras.Model):
    """Decoder that reconstructs CST geometry conditioned by Cl and alpha via FiLM."""

    def __init__(self, npv=8, latent_dim=16, film_depth=2):
        super().__init__()
        self.npv = npv
        self.latent_dim = latent_dim
        self.num_weights = 2 * self.npv
        if film_depth < 1 or film_depth > len(DEFAULT_FILM_HIDDEN_DIMS):
            raise ValueError(
                f"film_depth must be between 1 and {len(DEFAULT_FILM_HIDDEN_DIMS)}, got {film_depth}."
            )
        self.film_depth = film_depth
        self.hidden_dims = tuple(reversed(DEFAULT_FILM_HIDDEN_DIMS[:film_depth]))

        self.latent_dense_layers = [Dense(width) for width in self.hidden_dims]
        self.gamma_layers = [Dense(width) for width in self.hidden_dims]
        self.beta_layers = [Dense(width) for width in self.hidden_dims]
        self.multiply_layers = [Multiply() for _ in self.hidden_dims]
        self.add_layers = [Add() for _ in self.hidden_dims]
        self.activation_layers = [
            LeakyReLU(negative_slope=0.2) for _ in self.hidden_dims
        ]

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
        current_shape = latent_shape
        for dense_layer, gamma_layer, beta_layer, width in zip(
            self.latent_dense_layers,
            self.gamma_layers,
            self.beta_layers,
            self.hidden_dims,
        ):
            dense_layer.build(current_shape)
            gamma_layer.build(condition_shape)
            beta_layer.build(condition_shape)
            current_shape = tf.TensorShape([None, width])

        self.dense_weights.build(current_shape)
        self.reshape_weights.build(tf.TensorShape([None, self.num_weights]))
        self.dense_params.build(current_shape)

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

        x = latent
        for (
            dense_layer,
            gamma_layer,
            beta_layer,
            multiply_layer,
            add_layer,
            activation_layer,
        ) in zip(
            self.latent_dense_layers,
            self.gamma_layers,
            self.beta_layers,
            self.multiply_layers,
            self.add_layers,
            self.activation_layers,
        ):
            x = dense_layer(x)
            x = self._apply_film(
                x,
                condition,
                gamma_layer,
                beta_layer,
                multiply_layer,
                add_layer,
                activation_layer,
            )

        weights_flat = self.dense_weights(x)
        weights = self.reshape_weights(weights_flat)
        params = self.dense_params(x)

        return weights, params


if __name__ == "__main__":
    BATCH_SIZE = 4
    NPV = 8
    LATENT_DIM = 16

    decoder = FiLMDecoder(npv=NPV, latent_dim=LATENT_DIM, film_depth=2)

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
