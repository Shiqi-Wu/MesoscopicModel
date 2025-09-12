#!/usr/bin/env python3
"""
Test script for loading data and solving the Local PDE with the loaded initial
conditions (consistent with notes Eq. (21)).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append("src")

from local_pde_solver import (
    create_local_parameter_provider,
    solve_local_pde,
)
from utils import compute_susceptibility_series


def main():
    print("🧪 Testing Local PDE Solver with Loaded Data")
    print("=" * 60)

    # Example data path (edit if needed)
    h = 1
    T = 1
    epsilon = 0.03125
    data_path = f"data/ct_glauber_L1024_ell32_sigma1.2_tau1_m00.1_T{T}_J1_h{h}_tend5.0_dt0.01_block8_kernelgaussian_epsilon{epsilon}_seed0/round0/ct_glauber_L1024_ell32_sigma1.2_tau1_m00.1_T{T}_J1_h{h}_tend5.0_dt0.01_block8_kernelgaussian_epsilon{epsilon}_seed0_round0.npz"

    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return

    # Create provider and solve
    provider = create_local_parameter_provider(data_path, frame_idx=0)
    params = provider.get_params()
    print("Parameters (local):", params)

    print("Solving local PDE...")
    result = solve_local_pde(provider, method="rk4", show_progress=True)

    times = result["times"]
    phi = result["phi"]

    # Reference data for comparison
    ref = provider.get_reference_data()
    original_m = ref.get("macro_field")
    original_times = ref.get("times")

    # If reference is missing, just do quick plot
    if original_m is None or original_times is None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].imshow(phi[0], cmap="RdBu_r", vmin=-1, vmax=1)
        axes[0].set_title("Local PDE - Initial")
        axes[1].imshow(phi[-1], cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1].set_title("Local PDE - Final")
        mag = [np.mean(f) for f in phi]
        axes[2].plot(times, mag, "g-", label="Local PDE")
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Average Magnetization")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        out = f"results/local_pde_quick_h{params.h_field:g}_T{(1.0/params.beta):g}_eps{params.source_info.get('mapped',{}).get('eps','')}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"✅ Local PDE quick plot saved to: {out}")
        return

    # Comparison like nonlocal: 2x4 grid
    original_times = np.asarray(original_times)
    # Magnetizations
    original_mag = [np.mean(mf) for mf in original_m]
    local_mag = [np.mean(f) for f in phi]

    # Align time indices for time-series comparisons
    T_data = len(original_m)
    T_pde = len(phi)
    min_len = min(T_data, T_pde)
    orig_idx = np.linspace(0, T_data - 1, min_len, dtype=int)
    pde_idx = np.linspace(0, T_pde - 1, min_len, dtype=int)
    times_aligned = original_times[orig_idx]

    # Free energy vs time using same functional as nonlocal test
    mapped = params.source_info.get("mapped", {})
    beta = float(params.beta)
    h_field = float(params.h_field)
    eps = float(mapped.get("eps", 0.0))
    M = original_m.shape[-1]

    def _build_kernel_k(M: int, eps: float) -> np.ndarray:
        dx = 1.0 / M
        grid = np.arange(M, dtype=np.float32)
        rx = np.minimum(grid, M - grid) * dx
        ry = rx
        RX, RY = np.meshgrid(rx, ry, indexing="ij")
        R = np.sqrt(RX * RX + RY * RY)
        if eps > 0:
            Jr = np.exp(-(R * R) / (2.0 * eps * eps)) / (2.0 * np.pi * eps * eps)
        else:
            Jr = np.zeros((M, M), dtype=np.float32)
            Jr[0, 0] = 1.0
        s = Jr.sum()
        if s > 0:
            Jr = Jr / s
        Jr_rolled = np.fft.ifftshift(Jr)
        return np.fft.fft2(Jr_rolled)

    kernel_k = _build_kernel_k(M, eps)

    def _conv_J(m: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(np.fft.fft2(m) * kernel_k).real

    def _free_energy_series(
        traj: np.ndarray, times_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        Tn = traj.shape[0]
        F = np.empty(Tn, dtype=np.float64)
        dxdy = (1.0 / M) * (1.0 / M)
        for i in range(Tn):
            mfield = traj[i]
            m_clip = np.clip(mfield, -1.0 + 1e-6, 1.0 - 1e-6)
            p = 0.5 * (1.0 + m_clip)
            q = 0.5 * (1.0 - m_clip)
            ent = (p * np.log(p) + q * np.log(q)) / beta
            Jm = _conv_J(mfield)
            # In local limit J0=1
            E_int = -0.5 * mfield * Jm
            E_h = -h_field * mfield
            density = ent + E_int + E_h
            F[i] = np.sum(density) * dxdy
        return times_arr[:Tn], F

    tF_data, F_data = _free_energy_series(original_m, original_times)
    tF_pde, F_pde = _free_energy_series(phi, times)
    T_align_F = min(len(tF_data), len(tF_pde))
    tF = tF_data[:T_align_F]
    F_data = F_data[:T_align_F]
    F_pde = F_pde[:T_align_F]

    # Susceptibility χ(t)
    chi_data = compute_susceptibility_series(original_m, beta)
    chi_pde = compute_susceptibility_series(phi, beta)

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    # Row 1: original init, original final, magnetization, free energy
    im1 = axes[0, 0].imshow(original_m[0], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 0].set_title("Original - Initial (t=0)")
    plt.colorbar(im1, ax=axes[0, 0])
    im2 = axes[0, 1].imshow(original_m[-1], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 1].set_title("Original - Final")
    plt.colorbar(im2, ax=axes[0, 1])
    axes[0, 2].plot(original_times, original_mag, "b-", label="Original")
    axes[0, 2].plot(times, local_mag, "g-", label="Local PDE")
    axes[0, 2].set_title("Magnetization vs Time")
    axes[0, 2].set_xlabel("Time")
    axes[0, 2].set_ylabel("Average Magnetization")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 3].plot(tF, F_data, color="blue", label="Data F[m]")
    axes[0, 3].plot(tF, F_pde, color="green", linestyle="--", label="Local F[m]")
    axes[0, 3].set_title("Free Energy vs Time")
    axes[0, 3].set_xlabel("Time")
    axes[0, 3].set_ylabel("Free Energy")
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.3)

    # Row 2: local init, local final, difference, susceptibility
    im3 = axes[1, 0].imshow(phi[0], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 0].set_title("Local PDE - Initial")
    plt.colorbar(im3, ax=axes[1, 0])
    im4 = axes[1, 1].imshow(phi[-1], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 1].set_title("Local PDE - Final")
    plt.colorbar(im4, ax=axes[1, 1])
    diff = np.abs(phi[-1] - original_m[-1])
    im5 = axes[1, 2].imshow(diff, cmap="hot")
    axes[1, 2].set_title("|Local − Data| (final)")
    plt.colorbar(im5, ax=axes[1, 2])
    chi_data_al = chi_data[orig_idx]
    chi_pde_al = chi_pde[pde_idx]
    axes[1, 3].plot(times_aligned, chi_data_al, color="blue", label="Data χ(t)")
    axes[1, 3].plot(
        times_aligned, chi_pde_al, color="green", linestyle="--", label="Local χ(t)"
    )
    axes[1, 3].set_title("Susceptibility vs Time")
    axes[1, 3].set_xlabel("Time")
    axes[1, 3].set_ylabel("Susceptibility χ")
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)

    # Title and save
    T_val = (1.0 / float(params.beta)) if float(params.beta) != 0 else float("nan")
    h_val = float(params.h_field)
    eps_val = float(params.source_info.get("mapped", {}).get("eps", eps))
    fig.suptitle(
        f"Local PDE vs Data (h={h_val:g}, T={T_val:g}, eps={eps_val:g})",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs("results", exist_ok=True)
    out = f"results/local_pde_comparison_h{h_val:g}_T{T_val:g}_eps{eps_val:g}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"✅ Local PDE comparison plot saved to: {out}")


if __name__ == "__main__":
    main()
