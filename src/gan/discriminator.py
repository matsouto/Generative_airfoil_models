import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    LeakyReLU,
    Conv2D,
    Dropout,
    Flatten,
)


class CSTDiscriminatorCoords(tf.keras.Model):
    """
    * Refs:
        - Lin, Jinxing & Zhang, Chenliang & Xie, Xiaoye & Shi, Xingyu & Xu, Xiaoyu & Duan, Yanhui. (2022). CST-GANs: A Generative Adversarial Network Based on CST Parameterization for the Generation of Smooth Airfoils. 600-605. 10.1109/ICUS55513.2022.9987080.
    """

    def __init__(
        self,
        kernel_size: tuple = (2, 3),
        dropout: float = 0.4,
        depth: int = 8,
        dense_units: int = 256,
        n_points: int = 149,
    ):
        super().__init__()

        # --- Parameters ---

        self.kernel_size = kernel_size
        self.dropout = dropout
        self.depth = depth
        self.dense_units = dense_units
        self.n_points = n_points

        # --- Layers ---

        # Convolutional layers
        self.conv1 = Conv2D(
            self.depth * 2, self.kernel_size, strides=(1, 1), padding="same"
        )
        self.batch1 = BatchNormalization(momentum=0.9)
        self.leaky_relu1 = LeakyReLU(0.2)
        self.dropout1 = Dropout(dropout)

        self.conv2 = Conv2D(
            self.depth * 4, self.kernel_size, strides=(1, 2), padding="same"
        )
        self.batch2 = BatchNormalization(momentum=0.9)
        self.leaky_relu2 = LeakyReLU(0.2)
        self.dropout2 = Dropout(dropout)

        self.conv3 = Conv2D(
            self.depth * 8, self.kernel_size, strides=(1, 1), padding="same"
        )
        self.batch3 = BatchNormalization(momentum=0.9)
        self.leaky_relu3 = LeakyReLU(0.2)
        self.dropout3 = Dropout(dropout)

        self.conv4 = Conv2D(
            self.depth * 16, self.kernel_size, strides=(1, 2), padding="same"
        )
        self.batch4 = BatchNormalization(momentum=0.9)
        self.leaky_relu4 = LeakyReLU(0.2)
        self.dropout4 = Dropout(dropout)

        self.conv5 = Conv2D(
            self.depth * 32, self.kernel_size, strides=(1, 2), padding="same"
        )
        self.batch5 = BatchNormalization(momentum=0.9)
        self.leaky_relu5 = LeakyReLU(0.2)
        self.dropout5 = Dropout(dropout)

        self.conv6 = Conv2D(
            self.depth * 64, self.kernel_size, strides=(1, 2), padding="same"
        )
        self.batch6 = BatchNormalization(momentum=0.9)
        self.leaky_relu6 = LeakyReLU(0.2)
        self.dropout6 = Dropout(dropout)

        # Fully-connected layers
        self.flatten1 = Flatten()
        self.dense1 = Dense(self.dense_units)
        self.batch7 = BatchNormalization(momentum=0.9)
        self.leaky_relu7 = LeakyReLU(0.2)

        self.dense2 = Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = tf.reshape(inputs, [-1, 2, self.n_points, 1])

        x = self.conv1(x)
        x = self.batch1(x)
        x = self.leaky_relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.leaky_relu3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.batch4(x)
        x = self.leaky_relu4(x)
        x = self.dropout4(x)

        x = self.conv5(x)
        x = self.batch5(x)
        x = self.leaky_relu5(x)
        x = self.dropout5(x)

        x = self.conv6(x)
        x = self.batch6(x)
        x = self.leaky_relu6(x)
        x = self.dropout6(x)

        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.batch7(x)
        x = self.leaky_relu7(x)

        prediction = self.dense2(x)

        return prediction


if __name__ == "__main__":
    cst_discriminator = CSTDiscriminatorCoords()
    latent_vector = tf.random.normal([1, 2, 149, 1])
    output = cst_discriminator(latent_vector)
    print("Output shape: ", output.shape)
    float(output)
