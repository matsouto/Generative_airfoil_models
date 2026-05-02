import tensorflow as tf
from tensorflow.keras.layers import Add, Dense, Flatten, LeakyReLU, Multiply

DEFAULT_FILM_HIDDEN_DIMS = (256, 512, 512, 512)


class FiLMEncoder(tf.keras.Model):
    """Encoder that modulates geometry features with a 2D condition via FiLM."""

    def __init__(self, npv=8, latent_dim=16, film_depth=2):
        super().__init__()
        self.npv = npv
        self.latent_dim = latent_dim
        if film_depth < 1 or film_depth > len(DEFAULT_FILM_HIDDEN_DIMS):
            raise ValueError(
                f"film_depth must be between 1 and {len(DEFAULT_FILM_HIDDEN_DIMS)}, got {film_depth}."
            )
        self.film_depth = film_depth
        self.hidden_dims = DEFAULT_FILM_HIDDEN_DIMS[:film_depth]

        self.flatten = Flatten()
        self.geom_dense_layers = [Dense(width) for width in self.hidden_dims]
        self.gamma_layers = [Dense(width) for width in self.hidden_dims]
        self.beta_layers = [Dense(width) for width in self.hidden_dims]
        self.multiply_layers = [Multiply() for _ in self.hidden_dims]
        self.add_layers = [Add() for _ in self.hidden_dims]
        self.activation_layers = [
            LeakyReLU(negative_slope=0.2) for _ in self.hidden_dims
        ]

        self.dense_mean = Dense(self.latent_dim, name="z_mean")
        self.dense_log_var = Dense(self.latent_dim, name="z_log_var")

    def _split_inputs(self, inputs):
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError("FiLMEncoder expects inputs=(geometry, condition).")

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
        geometry, condition = self._split_inputs(inputs)

        x = self.flatten(geometry)

        for (
            dense_layer,
            gamma_layer,
            beta_layer,
            multiply_layer,
            add_layer,
            activation_layer,
        ) in zip(
            self.geom_dense_layers,
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

        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)

        return z_mean, z_log_var


if __name__ == "__main__":
    BATCH_SIZE = 4
    NPV = 8
    LATENT_DIM = 16

    encoder = FiLMEncoder(npv=NPV, latent_dim=LATENT_DIM, film_depth=2)

    dummy_geometry = tf.random.normal([BATCH_SIZE, (2 * NPV) + 2])
    dummy_condition = tf.random.uniform([BATCH_SIZE, 2], minval=-1.0, maxval=1.0)

    z_mean, z_log_var = encoder((dummy_geometry, dummy_condition))

    print("--- FiLM Encoder Test ---")
    print(f"Geometry shape: {dummy_geometry.shape}")
    print(f"Condition shape: {dummy_condition.shape}")
    print(f"Output z_mean shape: {z_mean.shape}")
    print(f"Output z_log_var shape: {z_log_var.shape}")

    assert z_mean.shape == (BATCH_SIZE, LATENT_DIM)
    assert z_log_var.shape == (BATCH_SIZE, LATENT_DIM)
    print("\nEncoder shapes are correct!")
