#!/usr/bin/env python3
"""
Test script for loading data and solving PDE with the loaded initial conditions.

This script combines the data loading approach from train.py with the PDE solver
to solve the nonlocal Allen-Cahn equation using the initial magnetization m0
from the loaded dataset.
"""

import sys
import os
import json
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.append("src")
from pde_solver import (
    NonlocalAllenCahnSolver,
    LocalAllenCahnSolver,
    create_solver_from_metadata,
    create_local_solver_from_metadata,
    create_local_ac_solver_from_metadata,
)
from nonlocal_pde_solver import (
    create_nonlocal_parameter_provider,
    solve_nonlocal_allen_cahn,
    solve_nonlocal_pde_from_data,
)
from plot import load_spins_data
from utils import compute_susceptibility_series


class IsingDataset:
    """
    Dataset for loading Ising model simulation data
    File format: .npz with metadata in .json file
    (Copied from train.py)
    """

    def __init__(self, file_path: str = ""):
        """
        Args:
            file_path: Path to the .npz file
        """
        self.file_path = file_path
        self.data_list = []
        self._load_all_data()

    def _load_all_data(self):
        """Load all .npz files and extract t, x, m, dm"""
        try:
            data = np.load(self.file_path)
            meta_data = json.load(open(self.file_path.replace(".npz", ".json")))

            print("=" * 60)
            print("Loading Dataset")
            print("=" * 60)
            print(f"Metadata keys: {list(meta_data.keys())}")
            print(f"Data keys: {list(data.keys())}")
            print(f"Spins shape: {data['spins'].shape}")
            print(f"Times shape: {data['times'].shape}")

            # Extract data arrays
            t = data["times"]
            x = data["spins"]

            # m is coarse-grained magnetization based on spins
            m = x.reshape(
                t.shape[0],
                meta_data["M"],
                meta_data["B"],
                meta_data["M"],
                meta_data["B"],
            ).mean(axis=(2, 4))

            # dmdt is dm/dt, using np.gradient for finite difference
            dt = meta_data.get("snapshot_dt", t[1] - t[0])
            dmdt = np.gradient(m, dt, axis=0)

            # Convert to tensors
            t_tensor = torch.from_numpy(t).float()
            x_tensor = torch.from_numpy(x).float()
            m_tensor = torch.from_numpy(m).float()
            dmdt_tensor = torch.from_numpy(dmdt).float()

            # Store data info
            data_info = {
                "file_path": self.file_path,
                "t": t_tensor,
                "x": x_tensor,
                "m": m_tensor,
                "dmdt": dmdt_tensor,
                "shape": m_tensor.shape,
                "metadata": meta_data,
            }

            print(f"\nLoaded {os.path.basename(self.file_path)}")
            print(f"  t shape: {t_tensor.shape}")
            print(f"  x shape: {x_tensor.shape}")
            print(f"  m shape: {m_tensor.shape}")
            print(f"  dmdt shape: {dmdt_tensor.shape}")
            print(f"  Time range: {t[0]:.3f} to {t[-1]:.3f}")
            print(f"  Magnetization range: [{m.min():.6f}, {m.max():.6f}]")

            self.data_list.append(data_info)

        except Exception as e:
            print(f"Error loading {self.file_path}: {e}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx) -> dict:
        """Returns a dictionary containing t, x, m, dmdt for the given index"""
        return self.data_list[idx]

    def get_sample_data(
        self, idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Get sample data for testing"""
        if idx >= len(self.data_list):
            idx = 0
        data = self.data_list[idx]
        return data["t"], data["x"], data["m"], data["dmdt"], data["metadata"]


def solve_pde_with_loaded_data(data_path: str, solver_type: str = "nonlocal"):
    """
    Load data and solve PDE using the initial magnetization m0.

    Args:
        data_path: Path to the .npz data file
        solver_type: "nonlocal" or "local"
    """
    print("=" * 60)
    print(f"PDE Solution with {solver_type.capitalize()} Solver")
    print("=" * 60)

    # Load dataset using train.py approach
    dataset = IsingDataset(file_path=data_path)

    if len(dataset) == 0:
        print("❌ No data loaded!")
        return None

    # Get sample data
    t, x, m, dmdt, metadata = dataset.get_sample_data(0)

    # Extract initial magnetization m0 (first time step)
    m0 = m[0].numpy()  # Convert to numpy for PDE solver

    print(f"\nInitial magnetization m0:")
    print(f"  Shape: {m0.shape}")
    print(f"  Range: [{m0.min():.6f}, {m0.max():.6f}]")
    print(f"  Mean: {m0.mean():.6f}")

    # Use new framework for nonlocal solver
    if solver_type == "nonlocal":
        # Parse epsilon from path if available for override
        eps_match = re.search(r"epsilon([0-9]*\.?[0-9]+)", data_path)
        eps_override = float(eps_match.group(1)) if eps_match else None

        # Create provider using new framework (override interaction_range if parsed)
        provider_kwargs = {"frame_idx": 0}
        if eps_override is not None:
            provider_kwargs["interaction_range"] = eps_override
        provider = create_nonlocal_parameter_provider(data_path, **provider_kwargs)
        params = provider.get_params()
        print(f"\n✅ Created nonlocal solver (new framework)")
        print(f"  Interaction strength: {params.interaction_strength}")
        print(f"  Interaction range: {params.interaction_range}")
        print(f"  Mobility prefactor: {params.mobility_prefactor}")
        print(f"  Grid size: {params.M}x{params.M}")

        # Solve using new framework
        print(f"\nSolving PDE with new framework...")
        result = solve_nonlocal_allen_cahn(provider, method="rk4", show_progress=True)
        final_field = result["phi"][-1]  # Last time step
        trajectory = result["phi"]  # Full trajectory
        pde_times = result["times"]

    elif solver_type == "local":
        # Keep using old framework for local solver (for now)
        solver = create_local_solver_from_metadata(metadata)
        print(f"\n✅ Created local solver (old framework)")
        print(f"  Spatial resolution δ: {solver.delta:.6f}")
        print(f"  Beta β: {solver.beta:.6f}")
        print(f"  External field h: {solver.h:.6f}")

        # Get simulation parameters
        dt = metadata["snapshot_dt"]
        t_end = metadata["t_end"]

        print(f"\nSimulation parameters:")
        print(f"  Time step dt: {dt}")
        print(f"  Total time t_end: {t_end}")
        print(f"  Number of steps: {int(t_end/dt)}")

        # Solve PDE
        print(f"\nSolving PDE...")
        final_field, trajectory = solver.solve(
            initial_field=m0,
            dt=dt,
            t_end=t_end,
            save_trajectory=True,
            show_progress=True,
            progress_interval=25,
        )
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

    print(f"\n✅ PDE solution completed")
    print(f"  Final field shape: {final_field.shape}")
    print(
        f"  Final field range: [{final_field.min().item():.6f}, {final_field.max().item():.6f}]"
    )
    print(f"  Trajectory shape: {trajectory.shape}")

    # Prepare return data
    return_data = {
        "initial_field": m0,
        "final_field": final_field,
        "trajectory": trajectory,
        "metadata": metadata,
        "original_data": {
            "t": t.numpy(),
            "m": m.numpy(),
            "dmdt": dmdt.numpy(),
        },
    }

    # Add solver info based on type
    if solver_type == "nonlocal":
        return_data["solver_type"] = "nonlocal_new_framework"
        return_data["params"] = params if "params" in locals() else None
        return_data["pde_times"] = pde_times if "pde_times" in locals() else None
    else:
        return_data["solver"] = solver
        return_data["solver_type"] = "local_old_framework"

    return return_data


def compare_solutions(data_path: str):
    """Compare nonlocal PDE solution with original data."""
    print("=" * 60)
    print("Comparing Nonlocal PDE vs Original Data")
    print("=" * 60)

    # Solve with nonlocal solver
    nonlocal_result = solve_pde_with_loaded_data(data_path, "nonlocal")

    if nonlocal_result is None:
        print("❌ Failed to solve PDE")
        return

    # Extract data for comparison
    original_m = nonlocal_result["original_data"]["m"]
    nonlocal_traj = nonlocal_result["trajectory"]
    nonlocal_times = nonlocal_result.get("pde_times")
    if nonlocal_times is None:
        dt = nonlocal_result["metadata"]["snapshot_dt"]
        t_end = nonlocal_result["metadata"]["t_end"]
        nonlocal_times = np.linspace(0, t_end, len(nonlocal_traj))

    # (Removed Local AC solve)

    # Create time arrays
    pde_times = nonlocal_times
    original_times = nonlocal_result["original_data"]["t"]

    # Calculate magnetization evolution
    original_mag = [original_m[i].mean() for i in range(len(original_m))]
    nonlocal_mag = [nonlocal_traj[i].mean() for i in range(len(nonlocal_traj))]
    # (Removed Local AC magnetization)

    # Indices/time alignment for time-series comparisons
    T_data = len(original_m)
    T_pde = len(nonlocal_traj)
    min_len = min(T_data, T_pde)
    orig_idx = np.linspace(0, T_data - 1, min_len, dtype=int)
    pde_idx = np.linspace(0, T_pde - 1, min_len, dtype=int)
    times_aligned = original_times[orig_idx]

    # Free energy vs time (mean-field functional):
    # F[m] = ∫ [ (1/beta)(p ln p + q ln q) - (J0/2) m (J*m) - h m ] dx
    # with p=(1+m)/2, q=(1-m)/2, J normalized so sum(J)=1 (cyclic conv)
    params = nonlocal_result.get("params")
    if params is not None:
        J0 = float(params.interaction_strength)
        h_field = float(params.h_field)
        beta = float(params.beta)
        Mgrid = int(original_m.shape[-1])

        def _build_kernel_k(M: int, eps: float) -> np.ndarray:
            dx = 1.0 / M
            grid = np.arange(M, dtype=np.float32)
            rx = np.minimum(grid, M - grid) * dx
            ry = rx
            RX, RY = np.meshgrid(rx, ry, indexing="ij")
            R = np.sqrt(RX * RX + RY * RY)
            Jr = np.exp(-(R * R) / (2.0 * params.interaction_range**2)) / (
                2.0 * np.pi * params.interaction_range**2
            )
            s = Jr.sum()
            if s > 0:
                Jr = Jr / s
            Jr_rolled = np.fft.ifftshift(Jr)
            return np.fft.fft2(Jr_rolled)

        kernel_k = _build_kernel_k(Mgrid, float(params.interaction_range))

        def _conv_J(m: np.ndarray) -> np.ndarray:
            return np.fft.ifft2(np.fft.fft2(m) * kernel_k).real

        def _free_energy_series(
            traj: np.ndarray, times: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            Tn = traj.shape[0]
            F = np.empty(Tn, dtype=np.float64)
            dxdy = (1.0 / Mgrid) * (1.0 / Mgrid)
            for i in range(Tn):
                m = traj[i]
                m_clip = np.clip(m, -1.0 + 1e-6, 1.0 - 1e-6)
                p = 0.5 * (1.0 + m_clip)
                q = 0.5 * (1.0 - m_clip)
                ent = (p * np.log(p) + q * np.log(q)) / beta
                Jm = _conv_J(m)
                E_int = -0.5 * J0 * m * Jm
                E_h = -h_field * m
                density = ent + E_int + E_h
                F[i] = np.sum(density) * dxdy
            return times[:Tn], F

        tF_data, F_data = _free_energy_series(original_m, original_times)
        tF_pde, F_pde = _free_energy_series(nonlocal_traj, pde_times)
        T_align_F = min(len(tF_data), len(tF_pde))
        tF = tF_data[:T_align_F]
        F_data = F_data[:T_align_F]
        F_pde = F_pde[:T_align_F]
    else:
        tF = np.array([])
        F_data = np.array([])
        F_pde = np.array([])

    # Create comparison plots (2x4 grid, larger figure to avoid crowding)
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # Plot 1: Original Data - Initial (move to first column)
    im1 = axes[0, 0].imshow(original_m[0], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 0].set_title(
        "Original Data - Initial (t=0)", fontsize=14, fontweight="bold"
    )
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    plt.colorbar(im1, ax=axes[0, 0])

    # Plot 2: Original Data - Final (move to middle)
    im2 = axes[0, 1].imshow(original_m[-1], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 1].set_title(
        "Original Data - Final (t=5.0)", fontsize=14, fontweight="bold"
    )
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    plt.colorbar(im2, ax=axes[0, 1])

    # Plot 3: Magnetization evolution (move to first-row rightmost)
    axes[0, 2].plot(
        original_times, original_mag, "b-", label="Original Data", linewidth=3
    )
    axes[0, 2].plot(pde_times, nonlocal_mag, "r--", label="Nonlocal PDE", linewidth=3)
    # (Local AC curve removed)
    axes[0, 2].set_xlabel("Time", fontsize=12)
    axes[0, 2].set_ylabel("Average Magnetization", fontsize=12)
    axes[0, 2].set_title(
        "Magnetization Evolution Comparison", fontsize=14, fontweight="bold"
    )
    axes[0, 2].legend(fontsize=12, loc="best")
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Nonlocal PDE - Initial
    im3 = axes[1, 0].imshow(nonlocal_traj[0], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 0].set_title("Nonlocal PDE - Initial (t=0)", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    plt.colorbar(im3, ax=axes[1, 0])

    # Plot 5: Nonlocal PDE - Final
    im4 = axes[1, 1].imshow(nonlocal_traj[-1], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 1].set_title("Nonlocal PDE - Final (t=5.0)", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    plt.colorbar(im4, ax=axes[1, 1])

    # Plot 6: Difference (Nonlocal - Original) at final time
    diff = np.abs(nonlocal_traj[-1] - original_m[-1])
    im5 = axes[1, 2].imshow(diff, cmap="hot")
    axes[1, 2].set_title(
        "Absolute Difference\n(PDE - Data) at t=5.0", fontsize=14, fontweight="bold"
    )
    axes[1, 2].set_xlabel("x")
    axes[1, 2].set_ylabel("y")
    plt.colorbar(im5, ax=axes[1, 2])

    # Plot 7: Free Energy vs Time (top-right)
    if tF.size > 0:
        axes[0, 3].plot(tF, F_data, color="blue", linewidth=2, label="Data F[m]")
        axes[0, 3].plot(
            tF, F_pde, color="red", linestyle="--", linewidth=2, label="PDE F[m]"
        )
        axes[0, 3].set_xlabel("Time", fontsize=12)
        axes[0, 3].set_ylabel("Free Energy", fontsize=12)
        axes[0, 3].set_title("Free Energy vs Time", fontsize=14, fontweight="bold")
        axes[0, 3].legend(fontsize=10)
        axes[0, 3].grid(True, alpha=0.3)
    else:
        axes[0, 3].axis("off")

    # Plot 8: Susceptibility χ(t) vs Time (bottom-right)
    beta_val = float(params.beta) if params is not None else 1.0
    chi_data = compute_susceptibility_series(original_m, beta_val)
    chi_pde = compute_susceptibility_series(nonlocal_traj, beta_val)
    # Align lengths for plotting
    chi_data_al = chi_data[orig_idx]
    chi_pde_al = chi_pde[pde_idx]
    axes[1, 3].plot(
        times_aligned, chi_data_al, color="blue", linewidth=2, label="Data χ(t)"
    )
    axes[1, 3].plot(
        times_aligned,
        chi_pde_al,
        color="red",
        linestyle="--",
        linewidth=2,
        label="PDE χ(t)",
    )
    axes[1, 3].set_xlabel("Time", fontsize=12)
    axes[1, 3].set_ylabel("Susceptibility χ", fontsize=12)
    axes[1, 3].set_title("Susceptibility vs Time", fontsize=14, fontweight="bold")
    axes[1, 3].legend(fontsize=10)
    axes[1, 3].grid(True, alpha=0.3)

    # Add overall title with parameters
    title_suffix = ""
    p = nonlocal_result.get("params")
    if p is not None:
        T_val = (1.0 / float(p.beta)) if float(p.beta) != 0 else float("nan")
        h_val = float(p.h_field)
        eps_val = float(p.interaction_range)
        title_suffix = f" (h={h_val:g}, T={T_val:g}, eps={eps_val:g})"
    fig.suptitle(f"PDE Comparison{title_suffix}", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save plot with h, T, epsilon in name
    os.makedirs("results", exist_ok=True)
    p = nonlocal_result.get("params")
    if p is not None:
        T_val = (1.0 / float(p.beta)) if float(p.beta) != 0 else float("nan")
        h_val = float(p.h_field)
        eps_val = float(p.interaction_range)
        output_path = f"results/pde_comparison_h{h_val:g}_T{T_val:g}_eps{eps_val:g}.png"
    else:
        output_path = "results/pde_comparison_with_loaded_data.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Comparison plot saved to: {output_path}")

    plt.show()

    # Print summary statistics
    print(f"\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print(f"Original data:")
    print(f"  Initial magnetization: {original_mag[0]:.6f}")
    print(f"  Final magnetization: {original_mag[-1]:.6f}")
    print(f"  Magnetization change: {original_mag[-1] - original_mag[0]:.6f}")

    print(f"\nNonlocal PDE:")
    print(f"  Initial magnetization: {nonlocal_mag[0]:.6f}")
    print(f"  Final magnetization: {nonlocal_mag[-1]:.6f}")
    print(f"  Magnetization change: {nonlocal_mag[-1] - nonlocal_mag[0]:.6f}")

    # Calculate differences
    nonlocal_diff = np.abs(nonlocal_traj[-1] - original_m[-1])

    print(f"\nDifferences from original data:")
    print(f"  Nonlocal PDE - Mean absolute difference: {nonlocal_diff.mean():.6f}")
    print(f"  Nonlocal PDE - Max absolute difference: {nonlocal_diff.max():.6f}")
    print(f"  Nonlocal PDE - RMS difference: {np.sqrt(np.mean(nonlocal_diff**2)):.6f}")


def main():
    """Main function to run the test."""
    print("🧪 Testing PDE Solver with Loaded Data")
    print("=" * 60)

    # Data path - you can change this to test different datasets
    # data_path = "data/ct_glauber_L1024_ell32_sigma1.2_tau1_m00.1_Tfrac3_J1_h0_tend5.0_dt0.01_block8_kernelnearest_R0.015625_seed0/round0/ct_glauber_L1024_ell32_sigma1.2_tau1_m00.1_Tfrac3_J1_h0_tend5.0_dt0.01_block8_kernelnearest_R0.015625_seed0_round0.npz"

    # Alternative data path with Gaussian kernel
    # data_path = "data/ct_glauber_L1024_ell32_sigma1.2_tau1_m00.1_Tfrac3_J1_h0_tend5.0_dt0.01_block8_kernelgaussian_epsilon0.015625_seed0/round0/ct_glauber_L1024_ell32_sigma1.2_tau1_m00.1_Tfrac3_J1_h0_tend5.0_dt0.01_block8_kernelgaussian_epsilon0.015625_seed0_round0.npz"

    # with T=1 and h=0
    h = 1
    T = 4
    epsilon = 0.03125
    data_path = f"data/ct_glauber_L1024_ell32_sigma1.2_tau1_m00.1_T{T}_J1_h{h}_tend5.0_dt0.01_block8_kernelgaussian_epsilon{epsilon}_seed0/round0/ct_glauber_L1024_ell32_sigma1.2_tau1_m00.1_T{T}_J1_h{h}_tend5.0_dt0.01_block8_kernelgaussian_epsilon{epsilon}_seed0_round0.npz"

    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please check the data path and try again.")
        return

    # Run comparison
    compare_solutions(data_path)

    # Create interactive Plotly visualization
    create_interactive_plot(data_path)

    # Create clear comparison plot with proper labeling
    create_clear_comparison_plot(data_path)


def create_interactive_plot(data_path: str):
    """Create interactive Plotly visualization showing PDE vs data at each time step."""
    print("\n" + "=" * 60)
    print("Creating Interactive Plotly Visualization")
    print("=" * 60)

    # Use new framework for nonlocal solver
    print("Using new nonlocal_pde_solver framework...")

    # Create theory provider
    provider = create_nonlocal_parameter_provider(data_path, frame_idx=0)

    # Get parameters
    params = provider.get_params()
    print(
        f"Parameters: interaction_strength={params.interaction_strength}, interaction_range={params.interaction_range}"
    )

    # Get initial condition and reference data
    phi0 = provider.get_initial_condition()
    ref_data = provider.get_reference_data()

    # Solve PDE using new framework
    print("Solving PDE with new framework...")
    pde_result = solve_nonlocal_allen_cahn(provider, method="rk4", show_progress=False)

    # Prepare data
    original_m = ref_data["macro_field"]
    original_times = ref_data["times"]
    nonlocal_traj = pde_result["phi"]
    pde_times = pde_result["times"]

    # Calculate magnetization evolution
    original_mag = [original_m[i].mean() for i in range(len(original_m))]
    nonlocal_mag = [nonlocal_traj[i].mean() for i in range(len(nonlocal_traj))]

    # Create subplots (simplified to 2x2)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Magnetization Evolution",
            "Original Data (t=0)",
            "Nonlocal PDE (t=final)",
            "Difference (PDE - Original)",
        ],
        specs=[
            [{"colspan": 2}, None],
            [{"type": "heatmap"}, {"type": "heatmap"}],
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    # Plot 1: Magnetization evolution
    fig.add_trace(
        go.Scatter(
            x=original_times,
            y=original_mag,
            mode="lines",
            name="Original Data",
            line=dict(color="blue", width=3),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=pde_times,
            y=nonlocal_mag,
            mode="lines",
            name="Nonlocal PDE",
            line=dict(color="red", width=3, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Plot 2: Original data initial (bottom-left)
    fig.add_trace(
        go.Heatmap(
            z=original_m[0],
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            showscale=False,
            name="Original (t=0)",
        ),
        row=2,
        col=1,
    )

    # Plot 3: Nonlocal PDE final (bottom-right)
    fig.add_trace(
        go.Heatmap(
            z=nonlocal_traj[-1],
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            showscale=True,
            name="Nonlocal (final)",
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title={
            "text": "Interactive Nonlocal PDE vs Data Comparison",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20},
        },
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Update axes
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Average Magnetization", row=1, col=1)

    # Add axis labels for heatmaps
    for i in range(1, 3):
        fig.update_xaxes(title_text="x", row=2, col=i)
        fig.update_yaxes(title_text="y", row=2, col=i)

    # Save interactive plot with parameters in filename and title suffix
    os.makedirs("results", exist_ok=True)
    T_val = (1.0 / float(params.beta)) if float(params.beta) != 0 else float("nan")
    h_val = float(params.h_field)
    eps_val = float(params.interaction_range)
    title_suffix = f" (h={h_val:g}, T={T_val:g}, ε={eps_val:g})"
    fig.update_layout(
        title={
            "text": f"Interactive Nonlocal PDE vs Data Comparison{title_suffix}",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20},
        }
    )
    output_path = (
        f"results/interactive_pde_comparison_h{h_val:g}_T{T_val:g}_eps{eps_val:g}.html"
    )
    fig.write_html(output_path)
    print(f"✅ Interactive plot saved to: {output_path}")

    # Create time series animation
    create_time_series_animation(
        original_m, nonlocal_traj, original_times, pde_times, params
    )


def create_time_series_animation(
    original_m, nonlocal_traj, original_times, pde_times, metadata
):
    """Create animated time series showing evolution of data vs PDE solutions."""
    print("Creating time series animation...")

    # Create frames for animation
    frames = []

    # Determine the number of frames (use fewer for performance)
    n_frames = min(50, len(original_m), len(nonlocal_traj))
    original_indices = np.linspace(0, len(original_m) - 1, n_frames, dtype=int)
    pde_indices = np.linspace(0, len(nonlocal_traj) - 1, n_frames, dtype=int)

    for i in range(n_frames):
        orig_idx = original_indices[i]
        pde_idx = pde_indices[i]

        # Create subplot for this frame - only 2 columns: Data vs PDE
        fig_frame = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                f"Original Data (t={original_times[orig_idx]:.2f})",
                f"Nonlocal PDE Solution (t={pde_times[pde_idx]:.2f})",
            ],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}]],
            horizontal_spacing=0.1,
        )

        # Add heatmaps
        fig_frame.add_trace(
            go.Heatmap(
                z=original_m[orig_idx],
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                showscale=False,
                name="Original Data",
            ),
            row=1,
            col=1,
        )

        fig_frame.add_trace(
            go.Heatmap(
                z=nonlocal_traj[pde_idx],
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                showscale=True,
                name="PDE Solution",
            ),
            row=1,
            col=2,
        )

        # Update layout for this frame
        fig_frame.update_layout(
            title=f"Data vs PDE Evolution (t={original_times[orig_idx]:.2f})",
            height=500,
            width=1000,
        )

        # Add axis labels
        fig_frame.update_xaxes(title_text="x", row=1, col=1)
        fig_frame.update_yaxes(title_text="y", row=1, col=1)
        fig_frame.update_xaxes(title_text="x", row=1, col=2)
        fig_frame.update_yaxes(title_text="y", row=1, col=2)

        frames.append(go.Frame(data=fig_frame.data, name=str(i)))

    # Create main figure with animation using the first frame's layout
    # Use the first frame's subplot structure
    fig_anim = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Original Data",
            "Nonlocal PDE Solution",
        ],
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]],
        horizontal_spacing=0.1,
    )

    # Add the first frame's data
    for trace in frames[0].data:
        fig_anim.add_trace(trace)

    # Add frames to the figure
    fig_anim.frames = frames

    # Add animation controls
    fig_anim.update_layout(
        title="Data vs PDE Time Evolution",
        height=500,
        width=1000,
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 200, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 100},
                            },
                        ],
                        "label": "▶️ Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "⏸️ Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Time: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [str(i)],
                            {
                                "frame": {"duration": 300, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 300},
                            },
                        ],
                        "label": f"{original_times[original_indices[i]]:.2f}",
                        "method": "animate",
                    }
                    for i in range(n_frames)
                ],
            }
        ],
    )

    # Save animated plot with parameters in filename
    os.makedirs("results", exist_ok=True)
    T_val_anim = (
        (1.0 / float(metadata.beta)) if float(metadata.beta) != 0 else float("nan")
    )
    h_val_anim = float(metadata.h_field)
    eps_val_anim = float(metadata.interaction_range)
    anim_output_path = f"results/data_vs_pde_evolution_h{h_val_anim:g}_T{T_val_anim:g}_eps{eps_val_anim:g}.html"
    fig_anim.write_html(anim_output_path)
    print(f"✅ Data vs PDE evolution animation saved to: {anim_output_path}")


def create_clear_comparison_plot(data_path: str):
    """
    Create a clear comparison plot between PDE solution and original data
    with proper labeling to distinguish the two.
    Uses the new nonlocal_pde_solver framework.
    """
    print("\n" + "=" * 60)
    print("Creating Clear Comparison Plot (New Framework)")
    print("=" * 60)

    # Use the new framework
    print(f"Using new nonlocal_pde_solver framework...")

    # Create theory provider
    provider = create_nonlocal_parameter_provider(data_path, frame_idx=0)

    # Get parameters
    params = provider.get_params()
    print(f"Parameters from theory:")
    print(f"  Interaction strength: {params.interaction_strength}")
    print(f"  Interaction range: {params.interaction_range}")
    print(f"  Mobility prefactor: {params.mobility_prefactor}")
    print(f"  Grid size: {params.M}x{params.M}")
    print(f"  Source: {params.source}")

    # Get initial condition
    phi0 = provider.get_initial_condition()
    print(f"\nInitial condition:")
    print(f"  Shape: {phi0.shape}")
    print(f"  Range: [{phi0.min():.6f}, {phi0.max():.6f}]")
    print(f"  Mean: {phi0.mean():.6f}")

    # Get reference data
    ref_data = provider.get_reference_data()
    print(f"\nReference data:")
    print(
        f"  Original times range: [{ref_data['times'][0]:.3f}, {ref_data['times'][-1]:.3f}]"
    )
    print(f"  Original data shape: {ref_data['macro_field'].shape}")
    print(f"  Original initial mean: {ref_data['macro_field'][0].mean():.6f}")

    # Solve PDE using new framework
    print(f"\nSolving PDE with new framework...")
    pde_result = solve_nonlocal_allen_cahn(provider, method="rk4", show_progress=True)

    pde_times = pde_result["times"]
    pde_phi = pde_result["phi"]  # (T, M, M)

    print(f"\n✅ PDE solution completed")
    print(f"  Solution shape: {pde_phi.shape}")
    print(f"  Time range: [{pde_times[0]:.3f}, {pde_times[-1]:.3f}]")
    print(f"  Final field range: [{pde_phi[-1].min():.6f}, {pde_phi[-1].max():.6f}]")

    # Get original data for comparison
    original_times = ref_data["times"]
    original_phi = ref_data["macro_field"]  # (T, M, M)

    # Create clear comparison plot
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    # Plot initial conditions
    axes[0, 0].imshow(original_phi[0], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 0].set_title(
        "Original Data - Initial (t=0)", fontsize=14, fontweight="bold"
    )
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")

    axes[0, 1].imshow(pde_phi[0], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 1].set_title("PDE Solution - Initial (t=0)", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")

    # Plot final conditions
    axes[0, 2].imshow(original_phi[-1], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 2].set_title(
        "Original Data - Final (t=5.0)", fontsize=14, fontweight="bold"
    )
    axes[0, 2].set_xlabel("x")
    axes[0, 2].set_ylabel("y")

    axes[1, 0].imshow(pde_phi[-1], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 0].set_title("PDE Solution - Final (t=5.0)", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")

    # Plot magnetization evolution
    original_mag = [np.mean(phi) for phi in original_phi]
    pde_mag = [np.mean(phi) for phi in pde_phi]

    axes[1, 1].plot(
        original_times, original_mag, "b-", label="Original Data", linewidth=3
    )
    axes[1, 1].plot(pde_times, pde_mag, "r--", label="PDE Solution", linewidth=3)
    axes[1, 1].set_xlabel("Time", fontsize=12)
    axes[1, 1].set_ylabel("Average Magnetization", fontsize=12)
    axes[1, 1].set_title(
        "Magnetization Evolution Comparison", fontsize=14, fontweight="bold"
    )
    axes[1, 1].legend(fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)

    # Plot difference between final states
    diff = np.abs(original_phi[-1] - pde_phi[-1])
    im = axes[1, 2].imshow(diff, cmap="hot")
    axes[1, 2].set_title(
        "Absolute Difference\n(Original - PDE) at t=5.0", fontsize=14, fontweight="bold"
    )
    axes[1, 2].set_xlabel("x")
    axes[1, 2].set_ylabel("y")
    cbar = plt.colorbar(im, ax=axes[1, 2])
    cbar.set_label("Difference", fontsize=12)

    # Add summary statistics as text
    stats_text = f"""Summary Statistics:
    
Original Data:
  Initial: {original_mag[0]:.6f}
  Final: {original_mag[-1]:.6f}
  Change: {original_mag[-1] - original_mag[0]:.6f}

PDE Solution:
  Initial: {pde_mag[0]:.6f}
  Final: {pde_mag[-1]:.6f}
  Change: {pde_mag[-1] - pde_mag[0]:.6f}

Difference (Final):
  Mean: {diff.mean():.6f}
  Max: {diff.max():.6f}"""

    # Add text box with statistics
    fig.text(
        0.02,
        0.02,
        stats_text,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
        verticalalignment="bottom",
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for text box
    plt.savefig(
        "results/pde_vs_data_clear_comparison.png", dpi=150, bbox_inches="tight"
    )
    print("✅ Clear comparison plot saved to results/pde_vs_data_clear_comparison.png")

    # Print detailed comparison statistics
    print("\n" + "=" * 60)
    print("Detailed Comparison Statistics")
    print("=" * 60)

    print(f"Time Evolution Comparison:")
    print(f"  Original Data: {len(original_times)} time points")
    print(f"  PDE Solution: {len(pde_times)} time points")

    print(f"\nInitial Conditions (t=0):")
    print(f"  Original Data: {original_mag[0]:.6f}")
    print(f"  PDE Solution: {pde_mag[0]:.6f}")
    print(f"  Difference: {abs(original_mag[0] - pde_mag[0]):.6f}")

    print(f"\nFinal Conditions (t=5.0):")
    print(f"  Original Data: {original_mag[-1]:.6f}")
    print(f"  PDE Solution: {pde_mag[-1]:.6f}")
    print(f"  Difference: {abs(original_mag[-1] - pde_mag[-1]):.6f}")

    print(f"\nMagnetization Change:")
    print(f"  Original Data: {original_mag[-1] - original_mag[0]:.6f}")
    print(f"  PDE Solution: {pde_mag[-1] - pde_mag[0]:.6f}")
    print(
        f"  Difference in change: {abs((original_mag[-1] - original_mag[0]) - (pde_mag[-1] - pde_mag[0])):.6f}"
    )

    # Field-level comparison
    final_diff = np.abs(original_phi[-1] - pde_phi[-1])
    print(f"\nField-level Comparison (Final State):")
    print(f"  Mean absolute difference: {final_diff.mean():.6f}")
    print(f"  Max absolute difference: {final_diff.max():.6f}")
    print(f"  RMS difference: {np.sqrt(np.mean(final_diff**2)):.6f}")


if __name__ == "__main__":
    main()
