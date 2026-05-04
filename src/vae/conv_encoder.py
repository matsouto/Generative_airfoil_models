import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Flatten, LeakyReLU


class ConvEncoder(tf.keras.Model):
    """Convolutional encoder for CST vectors represented as two surfaces."""

    def __init__(self, npv=12, latent_dim=16):
        super().__init__()
        self.npv = npv
        self.latent_dim = latent_dim

        self.conv1 = Conv1D(32, kernel_size=3, padding="same")
        self.act1 = LeakyReLU(negative_slope=0.2)
        self.conv2 = Conv1D(64, kernel_size=3, strides=2, padding="same")
        self.act2 = LeakyReLU(negative_slope=0.2)
        self.conv3 = Conv1D(128, kernel_size=3, strides=2, padding="same")
        self.act3 = LeakyReLU(negative_slope=0.2)

        self.flatten = Flatten()
        self.param_dense = Dense(32)
        self.param_act = LeakyReLU(negative_slope=0.2)
        self.fusion_dense = Dense(256)
        self.fusion_act = LeakyReLU(negative_slope=0.2)

        self.dense_mean = Dense(self.latent_dim, name="z_mean")
        self.dense_log_var = Dense(self.latent_dim, name="z_log_var")

    def _split_inputs(self, inputs):
        weight_dim = 2 * self.npv
        weights = inputs[:, :weight_dim]
        params = inputs[:, weight_dim:]
        return weights, params

    def _reshape_weights(self, weights):
        weights = tf.reshape(weights, [-1, 2, self.npv])
        return tf.transpose(weights, perm=[0, 2, 1])

    def call(self, inputs, training=None):
        del training
        weights, params = self._split_inputs(inputs)
        x = self._reshape_weights(weights)

        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.flatten(x)

        params_features = self.param_dense(params)
        params_features = self.param_act(params_features)

        fused = tf.concat([x, params_features], axis=1)
        fused = self.fusion_dense(fused)
        fused = self.fusion_act(fused)

        z_mean = self.dense_mean(fused)
        z_log_var = self.dense_log_var(fused)
        return z_mean, z_log_var
