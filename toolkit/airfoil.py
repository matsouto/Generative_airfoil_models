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
        coordinates_path: Union[str, None] = None,
    ):
        self.name = name

        if coordinates_path is None:
            try:  # See if it's a NACA airfoil
                self.coordinates = get_NACA_coordinates(name=self.name)
            except (ValueError, NotImplementedError):
                try:  # See if it's in the UIUC airfoil database
                    self.coordinates = get_UIUC_coordinates(name=self.name)
                except FileNotFoundError:
                    pass
        else:
            # If coordinates is a string, assume it's a filepath to a .dat file
            self.coordinates = get_file_coordinates(coordinates_path)

        if self.coordinates is None:
            import warnings

            warnings.warn(
                f"Airfoil {self.name} had no coordinates assigned, and could not parse the `coordinates` input!",
                UserWarning,
                stacklevel=2,
            )

    def __repr__(self) -> str:
        return f"Airfoil {self.name} ({self.n_points()} points)"

    # def generate_polars_xfoil(
    #     self,
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
