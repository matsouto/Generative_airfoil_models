import numpy as np
from src.airfoil.helpers import eng_string
from aerosandbox.geometry.airfoil import Airfoil
import aerosandbox.numpy as np

# Modify some methods from the Airfoil class from AeroSandbox library.


def draw(
    self,
    fig=None,
    draw_mcl=False,
    draw_markers=False,  # Mude para True ao chamar a função se quiser ver os pontos
    backend="matplotlib",
    main_color="#2F4F4F",  # Slate Gray (Elegante e profissional)
    mcl_color="#B22222",  # Firebrick Red (Para a linha média)
    fill=True,
    show=True,
    save_path=None,
) -> None:
    """
    Draw the airfoil object with academic/publication quality.

    Args:
        fig: Matplotlib/Plotly figure to use (optional)
        draw_mcl: Should we draw the mean camber line (MCL)? [boolean]
        draw_markers: Should we plot the discrete nodes? [boolean]
        backend: Which backend should we use? "plotly" or "matplotlib"
        main_color: Primary hex color for the airfoil outline.
        mcl_color: Hex color for the Mean Camber Line.
        fill: Fill the airfoil shape? [boolean]
        show: Should we show the plot? [boolean]

    Returns: None
    """
    x = np.reshape(np.array(self.x()), -1)
    y = np.reshape(np.array(self.y()), -1)

    if draw_mcl:
        x_mcl = np.linspace(np.min(x), np.max(x), len(x))
        y_mcl = self.local_camber(x_mcl)

    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        # Opcional: força uma fonte serifada para combinar com o LaTeX do texto
        # plt.rcParams['font.family'] = 'serif'

        # Configura o estilo da linha e dos marcadores
        if draw_markers:
            plt.plot(
                x,
                y,
                linestyle="-",
                marker="o",
                markersize=3,
                zorder=11,
                color=main_color,
                linewidth=1.5,
            )
        else:
            plt.plot(x, y, linestyle="-", zorder=11, color=main_color, linewidth=1.5)

        if fill:
            plt.fill(x, y, zorder=10, color="#708090", alpha=0.15)

        if draw_mcl:
            plt.plot(
                x_mcl,
                y_mcl,
                "--",
                zorder=12,
                color=mcl_color,
                linewidth=1.2,
                label="MCL",
            )
            plt.legend(loc="best", frameon=False)

        plt.axis("equal")
        plt.grid(True, linestyle=":", alpha=0.6, zorder=0)

        plt.title(f"Perfil Aerodinâmico: {self.name}", fontsize=12, fontweight="bold")
        plt.xlabel(r"$x/c$", fontsize=11)
        plt.ylabel(r"$y/c$", fontsize=11)

        # Remove a borda superior e direita (estilo paper/artigo científico)
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if show:
            plt.tight_layout()
            plt.show()

    elif backend == "plotly":
        import plotly.graph_objects as go

        if fig is None:
            fig = go.Figure()

        # Configura as propriedades do traço dependendo se queremos marcadores ou não
        trace_mode = "lines+markers" if draw_markers else "lines"

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode=trace_mode,
                name="Contorno",
                fill="toself" if fill else None,
                fillcolor="rgba(112, 128, 144, 0.15)",
                line=dict(color=main_color, width=2),
                marker=dict(size=7, color=main_color) if draw_markers else None,
            ),
        )

        if draw_mcl:
            fig.add_trace(
                go.Scatter(
                    x=x_mcl,
                    y=y_mcl,
                    mode="lines",
                    name="Linha de Arqueamento Médio (MCL)",
                    line=dict(color=mcl_color, width=2, dash="dash"),
                )
            )

        fig.update_layout(
            xaxis_title="x/c",
            yaxis_title="y/c",
            # title=dict(text=f"<b>Perfil: {self.name}</b>", x=0.5),
            plot_bgcolor="white",
            xaxis=dict(
                showgrid=True, gridcolor="lightgray", gridwidth=1, zeroline=False
            ),
            yaxis=dict(
                showgrid=True, gridcolor="lightgray", gridwidth=1, zeroline=False
            ),
        )

        if save_path:
            fig.write_image(save_path, width=2000, height=600, scale=4)
            print(f"Gráfico salvo com sucesso em: {save_path}")

        if show:
            fig.show()
        else:
            return fig


def generate_polars(
    self,
    alpha_i: float = 0.0,
    alpha_f: float = 10.0,
    alpha_step: float = 0.25,
    Res: np.ndarray = np.geomspace(1e4, 1e6, 12),
    n_iter: int = 100,
    min_points_to_converged: int = 20,
    working_directory: str = None,
) -> dict:
    """
    Generates polar data for the airfoil using XFOIL over a specified range of angles of attack.

    This method utilizes the `run_xfoil` function to perform simulations and retrieve polar data,
    which includes lift, drag, and moment coefficients for the airfoil at various angles of attack.

    Args:
        alpha_i (float): Initial angle of attack in degrees. Defaults to 0.0.
        alpha_f (float): Final angle of attack in degrees. Defaults to 10.0.
        alpha_step (float): Step size for the angle of attack in degrees. Defaults to 0.25.
        Re (int): Reynolds number for the simulation. Defaults to 1000000.
        n_iter (int): Maximum number of iterations for convergence. Defaults to 100.
        min_points_to_converged (int): Minimum number of points required for convergence. Defaults to 20.
        working_directory (str, optional): Directory to use for temporary files. Defaults to None.

    Returns:
        dict: Dictionary containing the polar data from the XFOIL simulation.
    """

    from xfoil import run_xfoil
    from tqdm import tqdm

    # Get a list of dicts, where each dict is the result of an XFoil run at a particular Re.
    run_datas = [
        run_xfoil(self, alpha_i, alpha_f, alpha_step, Re, n_iter, working_directory)
        for Re in tqdm(
            Res,
            desc=f"Running XFoil to generate polars for Airfoil '{self.name}':",
        )
    ]

    self.polars = run_datas
    self.Res = Res

    return self.polars

    # """Descarta os perfis que não convergirem"""
    # try:
    #     alpha = polar_data[:, 0]
    #     if len(alpha) < min_points_to_converged:
    #         self.converged = False
    #     else:
    #         Cl = polar_data[:, 1]
    #         Cd = polar_data[:, 2]
    #         self.converged = (
    #             True  # Estado que determina se o perfil convergiu na análise
    #         )


def plot_polars(
    self,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(9, 8))
    for data in self.polars:
        ax[0, 0].plot(data["alpha"], data["CL"])
        ax[0, 0].set(
            xlabel=r"Angle of Attack $\alpha$ [deg]",
            ylabel="Lift Coefficient $C_L$",
        )

        ax[0, 1].plot(data["alpha"], data["CD"])
        ax[0, 1].set(
            xlabel=r"Angle of Attack $\alpha$ [deg]",
            ylabel="Drag Coefficient $C_D$",
        )

        ax[1, 0].plot(data["alpha"], data["CM"])
        ax[1, 0].set(
            xlabel=r"Angle of Attack $\alpha$ [deg]",
            ylabel="Moment Coefficient $C_m$",
        )

        ax[1, 1].plot(data["CL"], data["CD"])
        ax[1, 1].set(
            xlabel=r"Angle of Attack $\alpha$ [deg]",
            ylabel=r"Lift-to-Drag Ratio $C_L/C_D$",
        )

    plt.sca(ax[0, 0])
    plt.legend(
        title="Reynolds Number",
        labels=[eng_string(Re) for Re in self.Res],
        ncol=2,
        # Note: `ncol` is old syntax; preserves backwards-compatibility with matplotlib 3.5.x.
        # New matplotlib versions use `ncols` instead.
        fontsize=8,
        loc="lower right",
    )

    # for i, Re in enumerate(Res):
    #     kwargs = dict(alpha=alphas, Re=Re, mach=mach)

    #     plt.sca(ax[0, 0])
    #     plt.plot(self.polars["alphas"], self.CL_function(**kwargs), color=Re_colors[i], alpha=0.7)

    #     plt.sca(ax[0, 1])
    #     plt.plot(alphas, self.CD_function(**kwargs), color=Re_colors[i], alpha=0.7)

    #     plt.sca(ax[1, 0])
    #     plt.plot(alphas, self.CM_function(**kwargs), color=Re_colors[i], alpha=0.7)

    #     plt.sca(ax[1, 1])
    #     plt.plot(
    #         alphas,
    #         self.CL_function(**kwargs) / self.CD_function(**kwargs),
    #         color=Re_colors[i],
    #         alpha=0.7,
    #     )

    # from aerosandbox.tools.string_formatting import eng_string

    # plt.sca(ax[0, 0])
    # plt.legend(
    #     title="Reynolds Number",
    #     labels=[eng_string(Re) for Re in Res],
    #     ncol=2,
    #     # Note: `ncol` is old syntax; preserves backwards-compatibility with matplotlib 3.5.x.
    #     # New matplotlib versions use `ncols` instead.
    #     fontsize=8,
    #     loc="lower right",
    # )


Airfoil.draw = draw
Airfoil.generate_polars = generate_polars
Airfoil.plot_polars = plot_polars

if __name__ == "__main__":
    airfoil = Airfoil("NACA0012")
    airfoil.generate_polars()
