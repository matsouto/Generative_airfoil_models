import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from aerosandbox import Airfoil
from .airfoil import airfoil_modifications


def generate_and_plot_airfoils(
    generator: tf.keras.Model,
    epoch: int = None,
    num_airfoils: int = 9,
    time: float = None,
    figsize: tuple = (2.5, 2.5),
    save_path: Path = None,
    dpi: int = 100,
    show: bool = True,
):

    def plot_samples(
        airfoils,
        scale=0.8,
        scatter=False,
        annotate=False,
        dot_size=20,
        figsize=(5, 5),
        linewidth=1.0,
        **kwargs,
    ):
        """
        Plot airfoil coordinates inline.

        Parameters:
        -----------
        airfoils : list of Airfoil objects
            List of Airfoil objects containing the coordinates to be plotted.
        scale : float, optional
            Scaling factor for the plot.
        scatter : bool, optional
            Whether to plot the shapes as a scatter plot.
        annotate : bool, optional
            Whether to annotate the points with their indices.
        dot_size : int, optional
            Size of the dots in the scatter plot.
        figsize : tuple, optional
            Size of the figure (width, height) in inches. Default is (5, 5).
        linewidth : float, optional
            Thickness of the lines when plotting shapes. Default is 1.0.
        **kwargs : dict
            Additional keyword arguments for plotting.
        """

        # Create a 2D plot with a customizable figure size
        fig, ax = plt.subplots(figsize=figsize)

        # Generate a grid for positioning airfoils
        N = len(airfoils)
        points_per_axis = int(np.sqrt(N))
        bounds = (0.0, 1.0)
        Z = gen_grid(2, points_per_axis, bounds[0], bounds[1])  # Generate a grid

        scale /= points_per_axis

        for i, z in enumerate(Z):
            # Extract coordinates from the Airfoil object
            coordinates = airfoils[i].coordinates
            plot_shape(
                coordinates,
                z[0],
                z[1],
                ax,
                scale,
                scatter,
                dot_size,
                linewidth,
                **kwargs,
            )
            if annotate:
                label = "{0}".format(i + 1)
                ax.annotate(label, xy=(z[0], z[1]), size=10)

        plt.xticks([])
        plt.yticks([])
        plt.axis("off")
        plt.axis("equal")
        plt.tight_layout()

        # Add a text box at the bottom to display losses
        if time is not None and epoch is not None:
            loss_text = f"Epoch: {epoch} / Elapsed Time: {time:.2f}s"
            plt.figtext(0.5, 0.01, loss_text, ha="center", fontsize=4)

        if save_path is not None:
            plt.savefig(f"{save_path}/epoch_{epoch if epoch else "none"}.png", dpi=dpi)

        if show:
            plt.show()  # Display the plot inline

    def plot_shape(
        coordinates, x, y, ax, scale, scatter, dot_size, linewidth, **kwargs
    ):
        """
        Helper function to plot a single shape.

        Parameters:
        -----------
        coordinates : numpy.ndarray
            The coordinates of the shape to be plotted.
        x : float
            The x-coordinate for the shape's position.
        y : float
            The y-coordinate for the shape's position.
        ax : matplotlib.axes.Axes
            The axis to plot on.
        scale : float
            Scaling factor for the plot.
        scatter : bool
            Whether to plot the shapes as a scatter plot.
        dot_size : int
            Size of the dots in the scatter plot.
        linewidth : float
            Thickness of the lines when plotting shapes.
        **kwargs : dict
            Additional keyword arguments for plotting.
        """
        # Adjust coordinates based on the position (x, y) and scale
        adjusted_coords = coordinates * scale + np.array([x, y])

        if scatter:
            ax.scatter(
                adjusted_coords[:, 0],
                adjusted_coords[:, 1],
                s=dot_size,
                color="black",
                **kwargs,
            )
        else:
            ax.plot(
                adjusted_coords[:, 0],
                adjusted_coords[:, 1],
                color="black",
                linewidth=linewidth,
                **kwargs,
            )

    def gen_grid(dim, points_per_axis, lower_bound, upper_bound):
        """
        Generate a grid of points in the specified dimension.

        Parameters:
        -----------
        dim : int
            The dimension of the grid.
        points_per_axis : int
            Number of points per axis.
        lower_bound : float
            Lower bound for the grid.
        upper_bound : float
            Upper bound for the grid.

        Returns:
        --------
        numpy.ndarray
            A grid of points.
        """
        # Generate a grid of points
        grid = np.mgrid[
            [slice(lower_bound, upper_bound, points_per_axis * 1j) for _ in range(dim)]
        ]
        grid = grid.reshape(dim, -1).T
        return grid

    # Example usage:
    latent_dim = generator.latent_dim
    noise = tf.random.normal([num_airfoils, latent_dim])
    generated_coords, _, _ = generator(noise, training=False)

    # Create a list of Airfoil objects with random coordinates
    airfoils = [Airfoil(coordinates=coords) for coords in generated_coords]

    plot_samples(
        airfoils,
        scale=1,
        scatter=False,
        annotate=False,
        figsize=figsize,
        linewidth=0.5,
    )
