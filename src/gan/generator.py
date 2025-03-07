import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    LeakyReLU,
    Conv2DTranspose,
    Conv2D,
    Reshape,
    ReLU,
)
from gan.cst_layer import CSTLayer
from aerosandbox.geometry.airfoil import Airfoil, KulfanAirfoil
from src.airfoil import airfoil_modifications


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
        use_modifications=True,
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
        self.use_modifications = use_modifications

        # --- Layers ---

        # Fully connected layers
        # CST Weights
        self.dense1 = Dense(self.depth * 2 * self.npv)
        self.batch1 = BatchNormalization(momentum=0.9)
        self.leaky_relu1 = LeakyReLU(0.2)

        self.reshape = Reshape((2, self.npv, self.depth))

        # LEM and TET
        self.dense2 = Dense(self.depth, input_shape=(self.latent_dim,))
        self.batch7 = BatchNormalization(momentum=0.9)
        self.leaky_relu7 = LeakyReLU(0.2)

        self.dense3 = Dense(int(self.depth / 4))
        self.batch8 = BatchNormalization(momentum=0.9)
        self.leaky_relu8 = LeakyReLU(0.2)

        self.dense4 = Dense(2)
        self.relu1 = ReLU()

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
        x = self.dense1(inputs)  # Input shape: (batch_size, latent_dim)
        x = self.batch1(x)
        x = self.leaky_relu1(x)

        x2 = self.dense2(inputs)
        x2 = self.batch7(x2)
        x2 = self.leaky_relu7(x2)

        x2 = self.dense3(x2)
        x2 = self.batch8(x2)
        x2 = self.leaky_relu8(x2)

        x2 = self.dense4(x2)
        parameters = self.relu1(x2)

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

        if not self.use_modifications:
            parameters = np.zeros_like(parameters)

        coordinates = self.cst_transform(weights, parameters)

        return coordinates, weights, parameters


if __name__ == "__main__":
    cst_generator = CSTGenerator()

    latent_vector = tf.random.normal([2, 128])
    coords, weights, parameters = cst_generator(latent_vector)
    print("Output shape: ", weights.shape)
    print("Coordinates shape: ", coords.shape)
    print("Parameters: ", parameters)

    test = Airfoil(coordinates=coords[0])
    test.draw()
    test2 = KulfanAirfoil(
        lower_weights=weights[0][0],
        upper_weights=weights[0][1],
        leading_edge_weight=parameters[0][0],
        TE_thickness=parameters[0][1],
    )
    test2.draw()
