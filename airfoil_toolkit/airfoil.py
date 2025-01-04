import numpy as np
import aerosandbox as asb
from typing import Union
from pathlib import Path
from aerosandbox.geometry.airfoil.airfoil_families import (
    get_NACA_coordinates,
    get_UIUC_coordinates,
    get_file_coordinates,
)


class Airfoil:
    def __init__(
        self,
        name: str = "Untitled",
        coordinates: str = None,
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

    def write_dat(
        self,
        filepath: Union[Path, str] = None,
        include_name: bool = True,
    ) -> str:
        """
        Writes a .dat file corresponding to this airfoil to a filepath.

        Args:
            filepath: filepath (including the filename and .dat extension) [string]
                If None, this function returns the .dat file as a string.

            include_name: Should the name be included in the .dat file? (In a standard *.dat file, it usually is.)

        Returns: None

        """
        contents = []

        if include_name:
            contents += [self.name]

        contents += ["%f %f" % tuple(coordinate) for coordinate in self.coordinates]

        string = "\n".join(contents)

        if filepath is not None:
            with open(filepath, "w+") as f:
                f.write(string)

        return string

    def generate_polars_xfoil(
        self,
        alpha_i: float = 0.0,
        alpha_f: float = 10.0,
        alpha_step: float = 0.25,
        Re: int = 1000000,
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

        from airfoil_toolkit.xfoil import run_xfoil

        # Run XFoil to get polars
        self.polars = run_xfoil(
            self, alpha_i, alpha_f, alpha_step, Re, n_iter, working_directory
        )

        return self.polars

        # with open(polar_path) as file:
        #     """Ref: https://github.com/ashokolarov/Genetic-airfoil"""
        #     polar_data = np.array(
        #         [
        #             np.array([float(x) for x in line.split()])
        #             for line in file.readlines()[12:]
        #         ]
        #     )

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

        #         Cl_integration = np.trapz(Cl, alpha)
        #         Cd_integration = np.trapz(Cd, alpha)
        #         Cl_max = max(Cl)
        #         ClCd = Cl / Cd
        #         ClCd_integration = np.trapz(ClCd, alpha)
        #         Cl3Cd2 = (Cl) ** 3 / (Cd) ** 2
        #         ClCd_max = max(ClCd)
        #         Cl3Cd2_max = max(Cl3Cd2)

        #         stall_angle = alpha[np.argmax(Cl)]

        #         self.alpha = alpha
        #         self.Cl = Cl
        #         self.Cd = Cd
        #         self.Cl_integration = Cl_integration
        #         self.Cd_integration = Cd_integration
        #         self.ClCd_integration = ClCd_integration
        #         self.Cl_max = Cl_max
        #         self.ClCd_max = ClCd_max
        #         self.Cl3Cd2_max = Cl3Cd2_max
        #         self.stall_angle = stall_angle
        # except:
        #     self.converged = False


if __name__ == "__main__":
    pass
