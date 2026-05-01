import tensorflow as tf
from tensorflow.keras.layers import Add, Dense, Flatten, LeakyReLU, Multiply


class FiLMEncoder(tf.keras.Model):
    """Encoder that modulates geometry features with a 2D condition via FiLM."""

    def __init__(self, npv=8, latent_dim=16):
        super().__init__()
        self.npv = npv
        self.latent_dim = latent_dim

        self.flatten = Flatten()

        self.geom_dense1 = Dense(256)
        self.gamma1 = Dense(256)
        self.beta1 = Dense(256)
        self.multiply1 = Multiply()
        self.add1 = Add()
        self.act1 = LeakyReLU(negative_slope=0.2)

        self.geom_dense2 = Dense(512)
        self.gamma2 = Dense(512)
        self.beta2 = Dense(512)
        self.multiply2 = Multiply()
        self.add2 = Add()
        self.act2 = LeakyReLU(negative_slope=0.2)

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

        x = self.geom_dense1(x)
        x = self._apply_film(
            x,
            condition,
            self.gamma1,
            self.beta1,
            self.multiply1,
            self.add1,
            self.act1,
        )

        x = self.geom_dense2(x)
        x = self._apply_film(
            x,
            condition,
            self.gamma2,
            self.beta2,
            self.multiply2,
            self.add2,
            self.act2,
        )

        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)

        return z_mean, z_log_var


if __name__ == "__main__":
    BATCH_SIZE = 4
    NPV = 8
    LATENT_DIM = 16

    encoder = FiLMEncoder(npv=NPV, latent_dim=LATENT_DIM)

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
