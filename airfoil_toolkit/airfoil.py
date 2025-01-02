import numpy as np
import pandas as pd
from typing import Union
import aerosandbox as asb
from aerosandbox.geometry.airfoil.airfoil_families import (
    get_NACA_coordinates,
    get_UIUC_coordinates,
    get_file_coordinates,
)


class Airfoil:
    def __init__(
        self,
        name: str = "Untitled",
        coordinates: Union[str, None] = None,
    ):
        """
        Initializes an Airfoil object with a given name and optional coordinates path.

        Attempts to retrieve airfoil coordinates based on the provided name from
        NACA or UIUC databases if no coordinates path is specified. If a path is
        provided, it assumes the path points to a .dat file containing the coordinates.

        Args:
            name (str): The name of the airfoil. Defaults to "Untitled".
            coordinates (Union[str, None]): Optional path to a .dat file containing
                airfoil coordinates. Defaults to None.

        Raises:
            UserWarning: If no coordinates could be assigned to the airfoil.
        """
        self.name = name
        self.coordinates = coordinates

        if coordinates is None:
            try:  # See if it's a NACA airfoil
                self.coordinates = get_NACA_coordinates(name=self.name)
            except (ValueError, NotImplementedError):
                try:  # See if it's in the UIUC airfoil database
                    self.coordinates = get_UIUC_coordinates(name=self.name)
                except FileNotFoundError:
                    pass
        else:
            # If coordinates is a string, assume it's a filepath to a .dat file
            self.coordinates = get_file_coordinates(coordinates)

        if self.coordinates is None:
            raise Exception(
                f"Airfoil {self.name} had no coordinates assigned, and could not parse the `coordinates` input!",
            )
        else:
            self.x = self.coordinates[:, 0]
            self.y = self.coordinates[:, 1]

    def __repr__(self) -> str:
        return f"Airfoil {self.name}"

    def draw(
        self, draw_mcl=False, draw_markers=True, backend="matplotlib", show=True
    ) -> None:
        """
        Draw the airfoil object.

        Args:
            draw_mcl: Should we draw the mean camber line (MCL)? [boolean]

            backend: Which backend should we use? "plotly" or "matplotlib"

            show: Should we show the plot? [boolean]

        Returns: None
        """
        x = np.reshape(np.array(self.x), -1)
        y = np.reshape(np.array(self.y), -1)
        # if draw_mcl:
        #     x_mcl = np.linspace(np.min(x), np.max(x), len(x))
        #     y_mcl = self.local_camber(x_mcl)

        if backend == "matplotlib":
            import matplotlib.pyplot as plt
            import aerosandbox.tools.pretty_plots as p

            color = "#280887"
            plt.plot(x, y, ".-" if draw_markers else "-", zorder=11, color=color)
            plt.fill(x, y, zorder=10, color=color, alpha=0.2)
            # if draw_mcl:
            #     plt.plot(x_mcl, y_mcl, "-", zorder=4, color=color, alpha=0.4)
            plt.axis("equal")
            if show:
                p.show_plot(
                    title=f"{self.name} Airfoil",
                    xlabel=r"$x/c$",
                    ylabel=r"$y/c$",
                )

        elif backend == "plotly":
            import plotly.graph_objects as go

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers" if draw_markers else "lines",
                    name="Airfoil",
                    fill="toself",
                    line=dict(color="blue"),
                ),
            )
            # if draw_mcl:
            #     fig.add_trace(
            #         go.Scatter(
            #             x=x_mcl,
            #             y=y_mcl,
            #             mode="lines",
            #             name="Mean Camber Line (MCL)",
            #             line=dict(color="navy"),
            #         )
            #     )
            fig.update_layout(
                xaxis_title="x/c",
                yaxis_title="y/c",
                yaxis=dict(scaleanchor="x", scaleratio=1),
                title=f"{self.name} Airfoil",
            )
            if show:
                fig.show()
            else:
                return fig

    # def generate_polars_xfoil(

    #     airfoil_path: str,
    #     name: str,
    #     alpha_i=0,
    #     alpha_f=10,
    #     alpha_step=0.25,
    #     Re=1000000,
    #     n_iter=100,
    #     polar_path="src/xfoil_runner/data/genome_polar.txt",
    #     min_points_to_converged=20,
    # ):
    #     """Roda simulação pelo XFOIL"""
    #     run_xfoil(
    #         airfoil_path,
    #         name,
    #         alpha_i,
    #         alpha_f,
    #         alpha_step,
    #         Re,
    #         n_iter,
    #         polar_path=polar_path,
    #     )
    #     self.sim = True

    #     with open(polar_path) as file:
    #         """Ref: https://github.com/ashokolarov/Genetic-airfoil"""
    #         polar_data = np.array(
    #             [
    #                 np.array([float(x) for x in line.split()])
    #                 for line in file.readlines()[12:]
    #             ]
    #         )

    #         """Descarta os perfis que não convergirem"""
    #         try:
    #             alpha = polar_data[:, 0]
    #             if len(alpha) < min_points_to_converged:
    #                 self.converged = False
    #             else:
    #                 Cl = polar_data[:, 1]
    #                 Cd = polar_data[:, 2]
    #                 self.converged = (
    #                     True  # Estado que determina se o perfil convergiu na análise
    #                 )

    #                 Cl_integration = np.trapz(Cl, alpha)
    #                 Cd_integration = np.trapz(Cd, alpha)
    #                 Cl_max = max(Cl)
    #                 ClCd = Cl / Cd
    #                 ClCd_integration = np.trapz(ClCd, alpha)
    #                 Cl3Cd2 = (Cl) ** 3 / (Cd) ** 2
    #                 ClCd_max = max(ClCd)
    #                 Cl3Cd2_max = max(Cl3Cd2)

    #                 stall_angle = alpha[np.argmax(Cl)]

    #                 self.alpha = alpha
    #                 self.Cl = Cl
    #                 self.Cd = Cd
    #                 self.Cl_integration = Cl_integration
    #                 self.Cd_integration = Cd_integration
    #                 self.ClCd_integration = ClCd_integration
    #                 self.Cl_max = Cl_max
    #                 self.ClCd_max = ClCd_max
    #                 self.Cl3Cd2_max = Cl3Cd2_max
    #                 self.stall_angle = stall_angle
    #         except:
    #             self.converged = False


if __name__ == "__main__":
    pass
