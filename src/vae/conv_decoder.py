import math

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D,
    Cropping1D,
    Dense,
    LeakyReLU,
    Reshape,
    UpSampling1D,
)

from ..layers.cst_layer import CSTLayer


class ConvDecoder(tf.keras.Model):
    """Convolutional decoder that reconstructs CST weights and scalar parameters."""

    def __init__(self, npv=12, latent_dim=16):
        super().__init__()
        self.npv = npv
        self.latent_dim = latent_dim
        self.num_weights = 2 * self.npv
        self.initial_length = max(2, math.ceil(self.npv / 4))
        self.final_sequence_length = self.initial_length * 4
        self.crop_size = self.final_sequence_length - self.npv

        self.seed_dense = Dense(self.initial_length * 64)
        self.seed_act = LeakyReLU(negative_slope=0.2)
        self.seed_reshape = Reshape((self.initial_length, 64))

        self.upsample1 = UpSampling1D(size=2)
        self.conv1 = Conv1D(64, kernel_size=3, padding="same")
        self.act1 = LeakyReLU(negative_slope=0.2)

        self.upsample2 = UpSampling1D(size=2)
        self.conv2 = Conv1D(32, kernel_size=3, padding="same")
        self.act2 = LeakyReLU(negative_slope=0.2)

        self.sequence_crop = (
            Cropping1D(cropping=(0, self.crop_size)) if self.crop_size > 0 else None
        )
        self.output_conv = Conv1D(2, kernel_size=3, padding="same", activation="tanh")
        self.param_dense = Dense(32)
        self.param_act = LeakyReLU(negative_slope=0.2)
        self.output_params = Dense(2)

        self.cst_transform = CSTLayer(num_weights=self.npv)

    def build(self, input_shape):
        current_shape = tf.TensorShape(input_shape)

        self.seed_dense.build(current_shape)
        self.seed_reshape.build(tf.TensorShape([None, self.initial_length * 64]))
        current_shape = tf.TensorShape([None, self.initial_length, 64])

        self.conv1.build(tf.TensorShape([None, self.initial_length * 2, 64]))
        current_shape = tf.TensorShape([None, self.initial_length * 2, 64])

        self.conv2.build(tf.TensorShape([None, self.final_sequence_length, 64]))
        current_shape = tf.TensorShape([None, self.final_sequence_length, 32])

        if self.sequence_crop is not None:
            self.sequence_crop.build(current_shape)
            current_shape = tf.TensorShape([None, self.npv, 32])

        self.output_conv.build(current_shape)
        self.param_dense.build(input_shape)
        self.output_params.build(tf.TensorShape([None, 32]))

        self.cst_transform(
            tf.zeros((1, 2, self.npv), dtype=tf.float32),
            tf.zeros((1, 2), dtype=tf.float32),
        )

        super().build(input_shape)

    def call(self, z, training=None):
        del training
        x = self.seed_dense(z)
        x = self.seed_act(x)
        x = self.seed_reshape(x)

        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.act1(x)

        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.act2(x)

        if self.sequence_crop is not None:
            x = self.sequence_crop(x)

        weights_sequence = self.output_conv(x)
        weights = tf.transpose(weights_sequence, perm=[0, 2, 1])

        params = self.param_dense(z)
        params = self.param_act(params)
        params = self.output_params(params)

        return weights, params
