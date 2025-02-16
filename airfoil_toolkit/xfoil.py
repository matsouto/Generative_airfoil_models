import re
import os
import sys
import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from airfoil_toolkit.airfoil import Airfoil
from pathlib import Path


def run_xfoil(
    airfoil: Airfoil,
    alpha_i: float = 0.0,
    alpha_f: float = 10.0,
    alpha_step: float = 0.25,
    Re: int = 200000,
    n_iter: int = 100,
    working_directory: str = None,
) -> np.ndarray:
    """
    Runs an XFOIL simulation for a given airfoil over a specified range of angles of attack.

    This function generates an input file for XFOIL, executes the simulation, and retrieves
    the polar data from the output file. It supports different operating systems by selecting
    the appropriate XFOIL binary.

    Args:
        airfoil (Airfoil): The airfoil object containing the airfoil name and coordinates.
        alpha_i (float): Initial angle of attack in degrees. Defaults to 0.0.
        alpha_f (float): Final angle of attack in degrees. Defaults to 10.0.
        alpha_step (float): Step size for the angle of attack in degrees. Defaults to 0.25.
        Re (int): Reynolds number for the simulation. Defaults to 200000.
        n_iter (int): Maximum number of iterations for convergence. Defaults to 100.
        working_directory (str, optional): Directory to use for temporary files. Defaults to None.

    Returns:
        np.ndarray: Array containing the polar data from the XFOIL simulation.

    Raises:
        RuntimeError: If the operating system is unsupported or if XFOIL execution fails.
        FileNotFoundError: If the polar output file is not found or is empty.
        RuntimeError: If there is an error reading the polar data.
    """

    # Selects binaries based on OS
    if sys.platform.startswith("win32"):
        XFOIL_BIN = "./XFOIL_BIN/xfoil.exe"
    elif sys.platform.startswith("darwin"):
        XFOIL_BIN = "xfoil"
    elif sys.platform.startswith("linux"):
        XFOIL_BIN = "xfoil"
    else:
        raise RuntimeError("Unsupported operating system for XFOIL execution.")

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)

        # Alternatively, work in another directory for debugging:
        if working_directory is not None:
            directory = Path(working_directory)

        input_file_path = directory / "input.in"
        output_file_path = directory / "output.txt"
        airfoil_dat_path = directory / f"airfoil.dat"

        # Generate airfoil .dat file from object
        airfoil.write_dat(airfoil_dat_path)

        # Generate input file for xfoil
        with open(input_file_path, "w") as file:
            file.write(f"LOAD {airfoil_dat_path}\n")
            file.write(airfoil.name + "\n")
            file.write("PANE\n")
            file.write("OPER\n")
            file.write(f"Visc {Re}\n")
            file.write("PACC\n")
            file.write(f"{output_file_path}\n\n")
            file.write(f"ITER {n_iter}\n")
            file.write(f"ASeq {alpha_i} {alpha_f} {alpha_step}\n")
            file.write("\n\n")
            file.write("quit\n")

        # Run XFOIL
        try:
            subprocess.run(
                f"{XFOIL_BIN} < {input_file_path}",
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            if e.returncode == 11:
                raise Exception(
                    "XFoil segmentation-faulted. This is likely because your input airfoil has too many points.\n"
                )
            elif e.returncode == 8 or e.returncode == 136:
                raise Exception(
                    "XFoil returned a floating point exception. This is probably because you are trying to start\n"
                    "your analysis at an operating point where the viscous boundary layer can't be initialized based\n"
                    "on the computed inviscid flow. (You're probably hitting a Goldstein singularity.) Try starting\n"
                    "your XFoil run at a less-aggressive (alpha closer to 0, higher Re) operating point."
                )
            else:
                raise e
        if (
            not os.path.exists(output_file_path)
            or os.path.getsize(output_file_path) == 0
        ):
            raise FileNotFoundError("Polar output file not found or is empty.")

        # Parse the polar
        regex = re.compile("(?:\s*([+-]?\d*.\d*))")

        with open(output_file_path) as f:
            lines = f.readlines()

            alpha = []
            cl = []
            cd = []
            cdp = []
            cm = []
            xtr_top = []
            xtr_bottom = []

            for line in lines[12:]:
                linedata = regex.findall(line)
                alpha.append(float(linedata[0]))
                cl.append(float(linedata[1]))
                cd.append(float(linedata[2]))
                cdp.append(float(linedata[3]))
                cm.append(float(linedata[4]))
                xtr_top.append(float(linedata[5]))
                xtr_bottom.append(float(linedata[6]))

            output = {
                "alpha": np.array(alpha),
                "CL": np.array(cl),
                "CD": np.array(cd),
                "CDp": np.array(cdp),
                "CM": np.array(cm),
                "Top_Xtr": np.array(xtr_top),
                "Bot_Xtr": np.array(xtr_bottom),
                "Re": Re * np.ones_like(np.array(alpha)),
            }

            os.remove("./:00.bl")

            return output


def plot_polar(axs, polar_path="src/xfoil_runner/data/genome_polar.txt"):
    """
    Plot Cl/alpha, Cd/alpha and Cl/Cd and Cl^3/Cd^2 from a certain analysis.
    :param polar_txt: .txt file where xfoil polar data is stored.
    """
    with open(polar_path) as file:
        data = np.array(
            [
                np.array([float(x) for x in line.split()])
                for line in file.readlines()[12:]
            ]
        )
        alpha = data[:, 0]
        Cl = data[:, 1]
        Cd = data[:, 2]
        Cl3Cd2 = Cl**3 / Cd**2

    axs[0, 0].plot(alpha, Cl)
    axs[0, 0].set(xlabel=r"$\alpha$ [-]", ylabel="$C_{l}$")

    axs[0, 1].plot(alpha, Cd)
    axs[0, 1].set(xlabel=r"$\alpha$ [-]", ylabel="$C_{d}$")

    axs[1, 0].plot(Cd, Cl)
    axs[1, 0].set(xlabel=r"$C_{d}$", ylabel="$C_{l}$")

    axs[1, 1].plot(alpha, Cl3Cd2)
    axs[1, 1].set(xlabel=r"$\alpha$", ylabel="$C_{l}^{3}/C_{d}^{2}$ [-]")


if __name__ == "__main__":
    airfoil = Airfoil(name="NACA0012")
    output = run_xfoil(airfoil)
    print(output)
