import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Union, List
from aerosandbox import Airfoil

# --- Helper Functions (Internal) ---


def _gen_grid(num_items: int, bounds: tuple = (0.0, 1.0)):
    """Generates a standardized grid for plotting multiple items."""
    points_per_axis = int(np.ceil(np.sqrt(num_items)))
    grid = np.mgrid[
        [slice(bounds[0], bounds[1], points_per_axis * 1j) for _ in range(2)]
    ]
    grid = grid.reshape(2, -1).T
    scale_factor = 1.0 / points_per_axis
    return grid[:num_items], scale_factor


def _plot_single_shape(
    ax,
    coordinates: np.ndarray,
    offset: np.ndarray,
    scale: float,
    color: str,
    linewidth: float,
    linestyle: str,
    scatter: bool,
    dot_size: int,
    **kwargs,
):
    """Plots a single shape with affine transformation (scale + translate)."""
    adjusted_coords = coordinates * scale + offset

    if scatter:
        ax.scatter(
            adjusted_coords[:, 0],
            adjusted_coords[:, 1],
            s=dot_size,
            color=color,
            **kwargs,
        )
    else:
        ax.plot(
            adjusted_coords[:, 0],
            adjusted_coords[:, 1],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            **kwargs,
        )


def _finalize_plot(
    text_label: str, save_path: Union[Path, str], filename: str, dpi: int, show: bool
):
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    plt.axis("equal")

    # Reserve bottom 5% of the figure for the text to prevent cutoff
    # rect = [left, bottom, right, top]
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if text_label:
        plt.figtext(0.5, 0.01, text_label, ha="center", fontsize=8)

    if save_path is not None and filename is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(save_path) / filename, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close()


# --- Main Functions ---


def generate_and_plot_airfoils(
    generator: tf.keras.Model,
    text_label: str = None,  # Replaces epoch/time
    num_airfoils: int = 9,
    figsize: tuple = (5, 5),
    scale: float = 0.8,
    save_path: Union[Path, str] = None,
    filename: str = "generated.png",  # explicit filename required now
    dpi: int = 100,
    show: bool = True,
    scatter: bool = False,
    linewidth: float = 1.0,
    linestyle: str = "-",
    dot_size: int = 20,
    **kwargs,
):
    """
    Generates latent noise, runs the generator, and plots the resulting airfoils.
    """
    latent_dim = generator.latent_dim
    noise = tf.random.normal([num_airfoils, latent_dim])
    generated_coords, _, _ = generator(noise, training=False)

    if isinstance(generated_coords, tf.Tensor):
        generated_coords = generated_coords.numpy()

    airfoils = [Airfoil(coordinates=coords) for coords in generated_coords]

    fig, ax = plt.subplots(figsize=figsize)
    grid_points, grid_scale = _gen_grid(num_airfoils)
    final_scale = scale * grid_scale

    for i, (pos, airfoil) in enumerate(zip(grid_points, airfoils)):
        _plot_single_shape(
            ax=ax,
            coordinates=airfoil.coordinates,
            offset=pos,
            scale=final_scale,
            color="black",
            linewidth=linewidth,
            linestyle=linestyle,
            scatter=scatter,
            dot_size=dot_size,
            **kwargs,
        )

    _finalize_plot(text_label, save_path, filename, dpi, show)


def plot_original_and_reconstruction(
    originals: List[Airfoil],
    reconstructions: List[Airfoil],
    text_label: str = None,  # Replaces epoch/time
    figsize: tuple = (5, 5),
    scale: float = 0.8,
    save_path: Union[Path, str] = None,
    filename: str = "reconstruction.png",
    dpi: int = 100,
    show: bool = True,
    scatter: bool = False,
    linewidth: float = 1.0,
    dot_size: int = 5,
    color_original: str = "black",
    color_reconstruction: str = "blue",
    linestyle_original: str = "--",
    linestyle_reconstruction: str = "-",
    annotate: bool = False,
    **kwargs,
):
    """
    Plots original and reconstructed airfoils overlapping in the same grid.
    """
    num_items = min(len(originals), len(reconstructions))

    fig, ax = plt.subplots(figsize=figsize)
    grid_points, grid_scale = _gen_grid(num_items)
    final_scale = scale * grid_scale

    for i, pos in enumerate(grid_points):
        # Plot Original (Dashed by default)
        _plot_single_shape(
            ax=ax,
            coordinates=originals[i].coordinates,
            offset=pos,
            scale=final_scale,
            color=color_original,
            linewidth=linewidth,
            linestyle=linestyle_original,
            scatter=scatter,
            dot_size=dot_size,
            **kwargs,
        )

        # Plot Reconstruction (Solid by default)
        _plot_single_shape(
            ax=ax,
            coordinates=reconstructions[i].coordinates,
            offset=pos,
            scale=final_scale,
            color=color_reconstruction,
            linewidth=linewidth,
            linestyle=linestyle_reconstruction,
            scatter=scatter,
            dot_size=dot_size,
            alpha=0.7,
            **kwargs,
        )

        if annotate:
            ax.annotate(f"{i+1}", xy=(pos[0], pos[1]), size=8)

    _finalize_plot(text_label, save_path, filename, dpi, show)
