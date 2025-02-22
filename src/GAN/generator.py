import numpy as np
from scipy.special import comb
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    LeakyReLU,
    Conv2DTranspose,
    Conv2D,
    Reshape,
)


class CSTLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CSTLayer, self).__init__(**kwargs)
        self.N1 = 0.5
        self.N2 = 1
        self.TE_thickness = 0.0
        self.leading_edge_weight = 0.0
        self.n_points_per_side = 75

    def call(self, inputs):
        # Inputs is a 3D tensor: (batch_size, 2, num_weights)
        # For example, shape: (batch_size, 2, 12)
        batch_size = tf.shape(inputs)[0]
        num_weights = tf.shape(inputs)[2]

        # Split into lower and upper weights
        lower_weights = inputs[:, 0, :]  # Shape: (batch_size, num_weights)
        upper_weights = inputs[:, 1, :]  # Shape: (batch_size, num_weights)

        # Generate cosinically spaced points
        x = (
            1 - tf.cos(tf.linspace(0.0, np.pi, self.n_points_per_side))
        ) / 2  # Shape: (n_points_per_side,)

        # Class function
        C = (x**self.N1) * ((1 - x) ** self.N2)  # Shape: (n_points_per_side,)

        def shape_function(w):
            # Shape function (Bernstein polynomials)
            N = tf.cast(tf.shape(w)[1] - 1, dtype=tf.float32)  # num_weights - 1

            # Bernstein binomial coefficients
            K = tf.cast(
                comb(N.numpy(), np.arange(N.numpy() + 1)), dtype=tf.float32
            )  # Shape: (num_weights,)

            # Expand dimensions for broadcasting
            K = tf.expand_dims(K, axis=-1)  # Shape: (num_weights, 1)
            x_expanded = tf.expand_dims(x, axis=0)  # Shape: (1, n_points_per_side)
            arange = tf.expand_dims(
                tf.cast(tf.range(N + 1), dtype=tf.float32), axis=-1
            )  # Shape: (num_weights, 1)

            # Compute Bernstein polynomials
            S_matrix = (
                K * (x_expanded**arange) * ((1 - x_expanded) ** (N - arange))
            )  # Shape: (num_weights, n_points_per_side)

            # Multiply by weights and sum over the Bernstein polynomials
            w_expanded = tf.expand_dims(
                w, axis=-1
            )  # Shape: (batch_size, num_weights, 1)
            S_x = tf.reduce_sum(
                w_expanded * S_matrix, axis=1
            )  # Shape: (batch_size, n_points_per_side)

            # Calculate the output y
            y = C * S_x  # Shape: (batch_size, n_points_per_side)
            return y

        # Apply shape function to lower and upper weights
        y_lower = shape_function(
            lower_weights
        )  # Shape: (batch_size, n_points_per_side)
        y_upper = shape_function(
            upper_weights
        )  # Shape: (batch_size, n_points_per_side)

        # Trailing edge thickness (TE thickness)
        y_lower -= x * self.TE_thickness / 2  # Shape: (batch_size, n_points_per_side)
        y_upper += x * self.TE_thickness / 2  # Shape: (batch_size, n_points_per_side)

        # Leading edge modification (LEM)
        y_lower += (
            self.leading_edge_weight
            * x
            * (1 - x) ** (tf.cast(num_weights, dtype=tf.float32) + 0.5)
        )
        y_upper += (
            self.leading_edge_weight
            * x
            * (1 - x) ** (tf.cast(num_weights, dtype=tf.float32) + 0.5)
        )

        # Create airfoil coordinates
        x = tf.tile(
            tf.expand_dims(x, axis=0), [batch_size, 1]
        )  # Shape: (batch_size, n_points_per_side)
        x = tf.concat(
            [x[:, ::-1], x[:, 1:]], axis=1
        )  # Shape: (batch_size, 2 * n_points_per_side - 1)
        y = tf.concat(
            [y_upper[:, ::-1], y_lower[:, 1:]], axis=1
        )  # Shape: (batch_size, 2 * n_points_per_side - 1)

        # Stack x and y coordinates
        coordinates = tf.stack(
            [x, y], axis=-1
        )  # Shape: (batch_size, 2 * n_points_per_side - 1, 2)
        return coordinates


class CSTGenerator(tf.keras.Model):
    """
    * Refs:
        - Lin, Jinxing & Zhang, Chenliang & Xie, Xiaoye & Shi, Xingyu & Xu, Xiaoyu & Duan, Yanhui. (2022). CST-GANs: A Generative Adversarial Network Based on CST Parameterization for the Generation of Smooth Airfoils. 600-605. 10.1109/ICUS55513.2022.9987080.
    """

    def __init__(
        self,
        npv: int = 12,
        latent_dim: int = 128,
        kernel_size: tuple = (2, 4),
        depth: int = 256,
    ):
        super().__init__()

        # --- Parameters ---

        """
        * npv: Number of parameterized variables
        * latent_dim: Dimension of input vector (latent vector)
        * depth: Number of channels after first dense layer
        """

        self.npv = npv
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.depth = depth

        # --- Layers ---

        # Fully connected layers
        self.dense1 = Dense(self.depth * 2 * self.npv, input_shape=(self.latent_dim,))
        self.batch1 = BatchNormalization(momentum=0.9)
        self.leaky_relu1 = LeakyReLU(0.2)

        self.reshape = Reshape((2, self.npv, self.depth))

        # Transposed convolutions
        self.deconv1 = Conv2DTranspose(
            int(self.depth / 2), self.kernel_size, strides=(1, 2), padding="same"
        )
        self.batch2 = BatchNormalization(momentum=0.9)
        self.leaky_relu2 = LeakyReLU(0.2)

        self.deconv2 = Conv2DTranspose(
            int(self.depth / 4), self.kernel_size, strides=(1, 2), padding="same"
        )
        self.batch3 = BatchNormalization(momentum=0.9)
        self.leaky_relu3 = LeakyReLU(0.2)

        self.deconv3 = Conv2DTranspose(
            int(self.depth / 8), self.kernel_size, strides=(1, 2), padding="same"
        )
        self.batch4 = BatchNormalization(momentum=0.9)
        self.leaky_relu4 = LeakyReLU(0.2)

        # Convolutional layers
        self.conv1 = Conv2D(
            int(self.depth / 16), self.kernel_size, strides=(1, 2), padding="same"
        )
        self.batch5 = BatchNormalization(momentum=0.9)
        self.leaky_relu5 = LeakyReLU(0.2)

        self.conv2 = Conv2D(
            int(self.depth / 32), self.kernel_size, strides=(1, 2), padding="same"
        )
        self.batch6 = BatchNormalization(momentum=0.9)
        self.leaky_relu6 = LeakyReLU(0.2)

        self.conv3 = Conv2D(
            1, self.kernel_size, strides=(1, 2), padding="same", activation="tanh"
        )

        # Output layers
        self.final_reshape = Reshape((2, self.npv))

        # Class-Shape Transformation
        self.cst_transform = CSTLayer()

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.batch1(x)
        x = self.leaky_relu1(x)

        x = self.reshape(x)

        x = self.deconv1(x)
        x = self.batch2(x)
        x = self.leaky_relu2(x)

        x = self.deconv2(x)
        x = self.batch3(x)
        x = self.leaky_relu3(x)

        x = self.deconv3(x)
        x = self.batch4(x)
        x = self.leaky_relu4(x)

        x = self.conv1(x)
        x = self.batch5(x)
        x = self.leaky_relu5(x)

        x = self.conv2(x)
        x = self.batch6(x)
        x = self.leaky_relu6(x)

        x = self.conv3(x)
        weights = self.final_reshape(x)

        coordinates = self.cst_transform(weights)

        return coordinates, weights


if __name__ == "__main__":
    cst_generator = CSTGenerator()
    latent_vector = tf.random.normal([2, 128])
    coords, weights = cst_generator(latent_vector)
    print("Output shape: ", coords.shape)
