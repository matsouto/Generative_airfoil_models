import tensorflow as tf
from tensorflow.keras.layers import Layer
from aerosandbox.geometry.airfoil import Airfoil
from src.airfoil import airfoil_modifications
import numpy as np


class CSTLayer(Layer):
    def __init__(self, **kwargs):
        super(CSTLayer, self).__init__(**kwargs)
        self.N1 = 0.5
        self.N2 = 1
        self.n_points_per_side = 75

    def call(self, inputs, parameters):
        """
        Args:
            inputs: A 3D tensor of shape (batch_size, 2, num_weights).
                   Represents the CST weights for the upper and lower surfaces.
            parameters: A 2D tensor of shape (batch_size, 2).
                       Represents batch-specific parameters:
                       - parameters[:, 0]: TE_thickness for each sample in the batch.
                       - parameters[:, 1]: leading_edge_weight for each sample in the batch.
        """

        # Inputs is a 3D tensor: (batch_size, 2, num_weights)
        # For example, shape: (batch_size, 2, 12)
        batch_size = tf.shape(inputs)[0]
        num_weights = tf.shape(inputs)[2]

        # Split into lower and upper weights
        lower_weights = inputs[:, 0, :]  # Shape: (batch_size, num_weights)
        upper_weights = inputs[:, 1, :]  # Shape: (batch_size, num_weights)

        # Extract batch-specific parameters
        leading_edge_weight = parameters[:, 0]  # Shape: (batch_size,)
        TE_thickness = parameters[:, 1]  # Shape: (batch_size,)

        # Generate cosinically spaced points
        x = (
            1 - tf.cos(tf.linspace(0.0, np.pi, self.n_points_per_side))
        ) / 2  # Shape: (n_points_per_side,)

        # Class function
        C = (x**self.N1) * ((1 - x) ** self.N2)  # Shape: (n_points_per_side,)

        def shape_function(w):
            # Shape function (Bernstein polynomials)
            N = tf.cast(tf.shape(w)[1] - 1, dtype=tf.float32)  # num_weights - 1

            # Compute binomial coefficients using TensorFlow
            k = tf.range(N + 1, dtype=tf.float32)  # Shape: (num_weights,)
            log_comb = (
                tf.math.lgamma(N + 1)
                - tf.math.lgamma(k + 1)
                - tf.math.lgamma(N - k + 1)
            )
            K = tf.exp(log_comb)  # Shape: (num_weights,)

            # Expand dimensions for broadcasting
            K = tf.expand_dims(K, axis=-1)  # Shape: (num_weights, 1)
            x_expanded = tf.expand_dims(x, axis=0)  # Shape: (1, n_points_per_side)
            arange = tf.expand_dims(k, axis=-1)  # Shape: (num_weights, 1)

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
        # Reshape TE_thickness for broadcasting: (batch_size, 1)
        TE_thickness = tf.expand_dims(TE_thickness, axis=-1)  # Shape: (batch_size, 1)
        y_lower -= x * TE_thickness / 2  # Shape: (batch_size, n_points_per_side)
        y_upper += x * TE_thickness / 2  # Shape: (batch_size, n_points_per_side)

        # Leading edge modification (LEM)
        # Reshape leading_edge_weight for broadcasting: (batch_size, 1)
        leading_edge_weight = tf.expand_dims(
            leading_edge_weight, axis=-1
        )  # Shape: (batch_size, 1)
        y_lower += (
            leading_edge_weight
            * x
            * (1 - x) ** (tf.cast(num_weights, dtype=tf.float32) + 0.5)
        )
        y_upper += (
            leading_edge_weight
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


if __name__ == "__main__":
    # Create an instance of the CSTLayer
    cst_layer = CSTLayer()

    # Create dummy input data
    batch_size = 3
    num_weights = 12  # Number of CST weights for each surface (upper and lower)
    dummy_input = tf.random.normal(
        [batch_size, 2, num_weights]
    )  # Shape: (batch_size, 2, num_weights)
    dummy_parameters = tf.random.normal([batch_size, 2])

    # Pass the input through the CSTLayer
    coordinates = cst_layer(
        dummy_input, dummy_parameters
    )  # Shape: (batch_size, 2 * n_points_per_side - 1, 2)

    test = Airfoil(coordinates=coordinates[0])
    test.draw()
