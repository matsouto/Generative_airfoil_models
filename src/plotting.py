import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Union, List
from aerosandbox import Airfoil
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import plotly.graph_objects as go

# --- Helper Functions (Internal) ---


def _gen_grid(num_items: int):
    """
    Gera um layout de grid padronizado para plotar múltiplos itens.
    Aplica a ordem de leitura correta (esquerda para direita, cima para baixo)
    e ajusta o espaçamento vertical para perfis aerodinâmicos.
    """
    cols = int(np.ceil(np.sqrt(num_items)))
    rows = int(np.ceil(num_items / cols))

    # Definimos distâncias fixas: dx (largura) e dy (altura)
    dx = 1.5  # Espaçamento horizontal entre perfis
    dy = 0.8  # Espaçamento vertical reduzido (perfis são finos)

    # Cria as coordenadas X (crescente) e Y (decrescente para começar do topo)
    x = np.arange(cols) * dx
    y = np.arange(rows)[::-1] * dy

    xv, yv = np.meshgrid(x, y)

    # Achata as matrizes combinando em pares (X, Y) na ordem de leitura
    grid = np.column_stack((xv.ravel(), yv.ravel()))

    # Como o espaçamento já está resolvido por dx e dy,
    # o fator de escala extra do grid pode ser fixado em 1.0
    scale_factor = 1.0

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
    """Plot a single shape with affine transformation (scale and translate)."""
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

    # Reserve bottom space for text label (rect format: [left, bottom, right, top])
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if text_label:
        plt.figtext(
            0.5, 0.01, text_label, ha="center", fontsize=8, fontfamily="monospace"
        )

    if save_path is not None and filename is not None:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        # O ajuste principal ocorre na linha abaixo
        plt.savefig(
            Path(save_path) / filename, dpi=dpi, bbox_inches="tight", pad_inches=0.1
        )

    if show:
        plt.show()
    else:
        plt.close()


# --- Main Functions ---


def save_latent_walk_gif(
    vae_model,
    start_point,
    end_point,
    filename: str = "latent_walk.gif",
    save_path: Union[Path, str] = None,
    num_steps: int = 40,
    fps: int = 20,
    figsize: tuple = (8, 5),
    scale: float = 1.0,
    start_name: str = "Start",
    end_name: str = "End",
    **kwargs,
):
    """
    Generate an animated GIF of latent space interpolation between two points.

    Creates a smooth morphing animation showing the interpolation path
    with progress tracking and persistent start/end labels.
    """

    # Extract and validate latent vectors
    z1 = start_point[0] if isinstance(start_point, (list, tuple)) else start_point
    z2 = end_point[0] if isinstance(end_point, (list, tuple)) else end_point

    if len(z1.shape) == 1:
        z1 = tf.expand_dims(z1, 0)
    if len(z2.shape) == 1:
        z2 = tf.expand_dims(z2, 0)

    z1 = tf.cast(z1, tf.float32)
    z2 = tf.cast(z2, tf.float32)

    # Linear interpolation in latent space
    alphas = np.linspace(0, 1, num_steps)
    interpolated_vectors = []
    for alpha in alphas:
        v = (1.0 - alpha) * z1 + alpha * z2
        interpolated_vectors.append(v)

    interpolated_batch = tf.concat(interpolated_vectors, axis=0)

    # Decode and transform to physical airfoil coordinates
    pred_weights_norm, pred_params_norm = vae_model.decoder(interpolated_batch)
    pred_weights_flat = tf.reshape(pred_weights_norm, [-1, 24])

    w_phys, p_phys = vae_model.scaler.inverse_transform(
        pred_weights_flat.numpy(), pred_params_norm.numpy()
    )

    w_phys_t = tf.reshape(tf.convert_to_tensor(w_phys, dtype=tf.float32), [-1, 2, 12])
    p_phys_t = tf.convert_to_tensor(p_phys, dtype=tf.float32)
    morphing_coords = vae_model.decoder.cst_transform(w_phys_t, p_phys_t).numpy()

    # Configure plot layout and limits
    all_x = morphing_coords[:, :, 0] * scale
    all_y = morphing_coords[:, :, 1] * scale
    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)

    pad_x = (max_x - min_x) * 0.15
    pad_y = (max_y - min_y) * 0.15

    fig, ax = plt.subplots(figsize=figsize)

    # Increase bottom margin to accommodate status text
    plt.subplots_adjust(bottom=0.35)

    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")

    cmap = plt.get_cmap("viridis")
    (line,) = ax.plot([], [], linewidth=3.0)

    # Position status text below plot area
    status_text = ax.text(
        0.5,
        -0.30,
        "",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        color="black",
        fontfamily="monospace",
        linespacing=1.5,
    )

    def init():
        line.set_data([], [])
        status_text.set_text("")
        return line, status_text

    def update(frame):
        # Update geometry
        coords = morphing_coords[frame]
        adjusted_coords = coords * scale
        line.set_data(adjusted_coords[:, 0], adjusted_coords[:, 1])

        # Update color based on interpolation progress
        progress = frame / (num_steps - 1)
        line.set_color(cmap(progress))

        # Update status text with frame information
        pct = int(progress * 100)
        l1 = f"Interpolation Step {frame}/{num_steps-1} [{pct:>3}% ]"
        l2 = f"{start_name}  ->  {end_name}"
        label = f"{l1}\n{l2}"

        status_text.set_text(label)
        return line, status_text

    # Create and save animation
    ani = animation.FuncAnimation(
        fig, update, frames=num_steps, init_func=init, blit=True, interval=1000 / fps
    )

    if save_path:
        out_path = Path(save_path) / filename
    else:
        out_path = Path(filename)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating GIF...")
    ani.save(str(out_path), writer="pillow", fps=fps)
    plt.close(fig)

    print(f"GIF saved to: {out_path}")


def plot_latent_walk(
    vae_model,
    start_point,
    end_point,
    num_steps: int = 20,
    figsize: tuple = (10, 6),
    scale: float = 1.0,
    text_label: str = "Latent Walk Interpolation",
    filename: str = "latent_walk.png",
    save_path: Union[Path, str] = None,
    show: bool = True,
    start_name: str = "Start",
    end_name: str = "End",
    colorbar: bool = True,
    **kwargs,
):
    """
    Plot linear interpolation in latent space as overlaid airfoils.

    Displays all interpolated airfoils superimposed on the same axes
    with a continuous color gradient representing interpolation progress.
    """

    # Remove 'title' kwarg if provided to avoid conflicts with Line2D
    if "title" in kwargs:
        # Update text_label above if title override is desired
        kwargs.pop("title")

    # Extract and validate latent vectors
    z1 = start_point[0] if isinstance(start_point, (list, tuple)) else start_point
    z2 = end_point[0] if isinstance(end_point, (list, tuple)) else end_point

    if len(z1.shape) == 1:
        z1 = tf.expand_dims(z1, 0)
    if len(z2.shape) == 1:
        z2 = tf.expand_dims(z2, 0)

    z1 = tf.cast(z1, tf.float32)
    z2 = tf.cast(z2, tf.float32)

    # Linear interpolation in latent space
    alphas = np.linspace(0, 1, num_steps)
    interpolated_vectors = []

    for alpha in alphas:
        v = (1.0 - alpha) * z1 + alpha * z2
        interpolated_vectors.append(v)

    interpolated_batch = tf.concat(interpolated_vectors, axis=0)

    # Decode and transform to physical airfoil coordinates
    pred_weights_norm, pred_params_norm = vae_model.decoder(interpolated_batch)
    pred_weights_flat = tf.reshape(pred_weights_norm, [-1, 24])

    w_phys, p_phys = vae_model.scaler.inverse_transform(
        pred_weights_flat.numpy(), pred_params_norm.numpy()
    )

    w_phys_t = tf.reshape(tf.convert_to_tensor(w_phys, dtype=tf.float32), [-1, 2, 12])
    p_phys_t = tf.convert_to_tensor(p_phys, dtype=tf.float32)
    morphing_coords = vae_model.decoder.cst_transform(w_phys_t, p_phys_t).numpy()

    # Create plot with all interpolated airfoils overlaid
    fig, ax = plt.subplots(figsize=figsize)

    # Use viridis for clear, high-contrast gradient
    cmap = plt.get_cmap("viridis")
    origin = np.array([0.0, 0.0])

    for i, coords in enumerate(morphing_coords):
        progress = i / (num_steps - 1)
        color = cmap(progress)

        # Apply styling: emphasize endpoints, de-emphasize intermediate steps
        if i == 0:
            lw = 2.5
            ls = "-"
            zorder = 10
            label = start_name
            alpha = 1.0
        elif i == num_steps - 1:
            lw = 2.5
            ls = "-"
            zorder = 10
            label = end_name
            alpha = 1.0
        else:
            lw = 1.2
            ls = "--"
            zorder = 1
            label = None
            alpha = 0.6

        _plot_single_shape(
            ax=ax,
            coordinates=coords,
            offset=origin,
            scale=scale,
            color=color,
            linewidth=lw,
            linestyle=ls,
            scatter=False,
            dot_size=0,
            alpha=alpha,
            label=label,
            **kwargs,  # Safe to use (title already removed)
        )

    # Add legend
    ax.legend(loc="upper right", framealpha=0.9)

    # Add optional colorbar for interpolation progress
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        cbar = plt.colorbar(sm, ax=ax, ticks=[0, 1])
        cbar.ax.set_yticklabels([start_name, end_name])
        cbar.set_label("Morphing Progress")

    ax.grid(True, alpha=0.3)

    # Finalize
    _finalize_plot(text_label, save_path, filename, 100, show)


def generate_and_plot_airfoils(
    generator: tf.keras.Model,
    text_label: str = None,  # Display label instead of epoch/time
    num_airfoils: int = 9,
    figsize: tuple = (5, 5),
    scale: float = 0.8,
    save_path: Union[Path, str] = None,
    filename: str = "generated.png",  # Required filename for saving
    dpi: int = 100,
    show: bool = True,
    scatter: bool = False,
    linewidth: float = 1.0,
    linestyle: str = "-",
    dot_size: int = 20,
    **kwargs,
):
    """
    Generate random airfoils and plot them in a grid layout.

    Samples random latent vectors, generates airfoils via the generator model,
    and creates a visualization grid.
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
    text_label: str = None,  # Display label instead of epoch/time
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
    Compare original and reconstructed airfoils side-by-side in a grid.

    Displays original airfoils (dashed) overlaid with reconstructions (solid)
    for easy visual comparison of reconstruction quality.
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


def plot_airfoil_list(
    airfoils: Union[List[np.ndarray], List[Airfoil]],
    labels: Union[str, bool, List[str]] = "index",
    text_label: str = None,
    figsize: tuple = (10, 10),
    scale: float = 0.8,
    save_path: Union[Path, str] = None,
    filename: str = "airfoil_list.png",
    dpi: int = 100,
    show: bool = True,
    color: str = "red",
    linewidth: float = 1.5,
    linestyle: str = "-",
    scatter: bool = False,
    dot_size: int = 20,
    **kwargs,
):
    """
    Plot a collection of airfoils in a grid layout.

    Accepts either numpy coordinate arrays (Nx2) or Airfoil objects.
    Useful for visualizing invalid airfoils or other specific subsets.
    """

    if not airfoils:
        print("No airfoils to plot.")
        return

    coords_list = []
    for item in airfoils:
        if isinstance(item, Airfoil):
            coords_list.append(item.coordinates)
        elif isinstance(item, np.ndarray):
            coords_list.append(item)
        else:
            raise ValueError("Input list must contain Airfoil objects or numpy arrays.")

    num_items = len(coords_list)

    # Validação de segurança: se fornecer uma lista de nomes, ela precisa ter o mesmo tamanho da lista de perfis
    if isinstance(labels, list) and len(labels) != num_items:
        print(
            f"Aviso: A quantidade de labels ({len(labels)}) é diferente do número de perfis ({num_items}). Usando índices como fallback."
        )
        labels = "index"

    fig, ax = plt.subplots(figsize=figsize)
    grid_points, grid_scale = _gen_grid(num_items)
    final_scale = scale * grid_scale

    for i, (pos, coords) in enumerate(zip(grid_points, coords_list)):
        _plot_single_shape(
            ax=ax,
            coordinates=coords,
            offset=pos,
            scale=final_scale,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            scatter=scatter,
            dot_size=dot_size,
            **kwargs,
        )

        # Lógica de decisão para o texto
        label_text = None
        if labels == "index" or labels is True:
            label_text = f"Idx: {i}"
        elif isinstance(labels, list):
            label_text = str(labels[i])

        # Só desenha o texto se a variável não for nula
        if label_text:
            ax.text(
                pos[0] + (final_scale * 0.5),
                pos[1] - (final_scale * 0.25),
                label_text,
                ha="center",
                fontsize=8,
                color="black",
                fontfamily="monospace",
            )

    _finalize_plot(text_label, save_path, filename, dpi, show)


def plot_airfoil_superposition(
    airfoils: List[Airfoil],
    draw_mcl: bool = False,
    draw_markers: bool = False,
    show_legend: bool = False,
    show: bool = True,
    base_width: int = 1000,
    save_path: Union[Path, str] = None,
    filename: str = "dataset.png",
    scale: float = 1.0,
) -> Union[None, go.Figure]:
    """
    Plota um dataset de perfis sobrepostos usando Plotly.
    Otimizado para legibilidade em documentos PDF com fontes maiores,
    mantendo as bordas brancas (sem quadro delimitador).
    """
    fig = go.Figure()

    all_x = []
    all_y = []

    for i, airfoil in enumerate(airfoils):
        x = np.reshape(np.array(airfoil.x()), -1)
        y = np.reshape(np.array(airfoil.y()), -1)

        all_x.extend(x)
        all_y.extend(y)

        name = airfoil.name if airfoil.name else f"Perfil {i}"

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers" if draw_markers else "lines",
                name=name,
                opacity=0.5,
                line=dict(width=2.0),
            )
        )

        if draw_mcl:
            x_mcl = np.linspace(np.min(x), np.max(x), len(x))
            y_mcl = airfoil.local_camber(x_mcl)

            fig.add_trace(
                go.Scatter(
                    x=x_mcl,
                    y=y_mcl,
                    mode="lines",
                    name=f"{name} (MCL)",
                    line=dict(dash="dot", width=1.5),
                    opacity=0.4,
                )
            )

    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        pad = (max_x - min_x) * 0.05

        range_x = [min_x - pad, max_x + pad]
        range_y = [min_y - pad, max_y + pad]

        dx = range_x[1] - range_x[0]
        dy = range_y[1] - range_y[0]

        aspect_ratio = dy / dx if dx > 0 else 1

        final_height = int(base_width * aspect_ratio) + 120
    else:
        range_x = None
        range_y = None
        final_height = 400

    fig.update_layout(
        width=base_width,
        height=final_height,
        showlegend=show_legend,
        font=dict(family="Arial, sans-serif", size=18, color="black"),
        xaxis_title="x/c",
        yaxis_title="y/c",
        xaxis_title_font=dict(size=22, family="Arial, sans-serif", color="black"),
        yaxis_title_font=dict(size=22, family="Arial, sans-serif", color="black"),
        xaxis=dict(
            range=range_x,
            tickfont=dict(size=16),
            showline=False,  # Remove a linha inferior do eixo
        ),
        yaxis=dict(
            range=range_y,
            scaleanchor="x",
            scaleratio=1,
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=1.5,
            tickfont=dict(size=16),
            showline=False,  # Remove a linha lateral do eixo
        ),
        hovermode="closest",
        plot_bgcolor="white",
        margin=dict(l=60, r=40, t=30, b=60),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1.5, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridwidth=1.5, gridcolor="LightGray")

    if save_path is not None and filename is not None:
        out_path = Path(save_path)
        out_path.mkdir(parents=True, exist_ok=True)
        full_file_path = out_path / filename

        fig.write_image(str(full_file_path), scale=scale)
        print(f"Gráfico salvo em: {full_file_path} com escala {scale}x")

    if show:
        fig.show()
    else:
        return fig
