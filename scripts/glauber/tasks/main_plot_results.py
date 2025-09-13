import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys

sys.path.append("src")
from plot import plot_Ms_Es, plot_animation
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Plot Ising Glauber results from npz/json"
    )
    parser.add_argument("folder", type=str, help="Folder containing results npz/json")
    args = parser.parse_args()

    folder = Path(args.folder)
    npz_files = list(folder.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No npz file found in {folder}")
    npz_path = npz_files[0]

    data = np.load(npz_path)
    times = data["times"]
    Ms = data["Ms"]
    Es = data["Es"]
    spins = data["spins"]
    spins_meso = data["spins_meso"]

    # 1) Ms, Energy vs time
    plot_Ms_Es(times, Ms, Es, folder)

    # 2) 动图 (micro vs meso)
    plot_animation(times, spins, spins_meso, folder)


if __name__ == "__main__":
    main()
