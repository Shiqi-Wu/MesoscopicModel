#!/usr/bin/env python3
"""
Compare three PDE solvers vs data: Nonlocal (tanh), Local-Poly (cubic AC), Local-Tanh (local limit).

Creates a 4x4 figure:
  Row 1: Data initial/final, error curves (all 3), magnetization (all 3 + data)
  Row 2: Local-Poly initial/final, |Local-Poly − Data| (final), Free energy (all 3 + data)
  Row 3: Local-Tanh initial/final, |Local-Tanh − Data| (final), (unused)
  Row 4: Nonlocal initial/final, |Nonlocal − Data| (final), Susceptibility (all 3 + data)
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append("src")

from local_pde_solver import (
    create_local_parameter_provider,
    solve_local_pde,
)
from nonlocal_pde_solver import (
    create_nonlocal_parameter_provider,
    solve_nonlocal_allen_cahn,
)
from utils import (
    compute_susceptibility_series,
    load_all_rounds,
    build_kernel_fft,
    free_energy_series,
)


def main():
    print("🧪 Comparing Nonlocal vs Local-Poly vs Local-Tanh (Loaded Data)")
    print("=" * 60)
    base_dir = "data/ct_glauber"
    # Data path
    h = 1
    T = 1
    epsilon = 0.03125
    L_scale = 1
    ell = 32 * L_scale
    L = 1024 * L_scale
    block = 8 * L_scale
    t_end = 20.0
    data_path = os.path.join(
        base_dir,
        f"ct_glauber_L{L}_ell{ell}_sigma1.2_tau1_m00.1_T{T}_J1_h{h}_"
        f"tend{t_end}_dt0.01_block{block}_kernelgaussian_epsilon{epsilon}_seed0",
        f"round0",
        f"ct_glauber_L{L}_ell{ell}_sigma1.2_tau1_m00.1_T{T}_J1_h{h}_"
        f"tend{t_end}_dt0.01_block{block}_kernelgaussian_epsilon{epsilon}_seed0_round0.npz",
    )
    scale_std = 20

    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return

    # Local-Poly provider and solve (default rhs_mode="poly")
    local_poly_provider = create_local_parameter_provider(data_path, frame_idx=0)
    local_poly_params = local_poly_provider.get_params()
    res_local_poly = solve_local_pde(
        local_poly_provider, method="rk4", show_progress=True
    )
    times_poly = res_local_poly["times"]
    phi_poly = res_local_poly["phi"]

    # Local-Tanh provider and solve (rhs_mode="tanh_local")
    local_tanh_provider = create_local_parameter_provider(
        data_path, frame_idx=0, rhs_mode="tanh_local"
    )
    local_tanh_params = local_tanh_provider.get_params()
    res_local_tanh = solve_local_pde(
        local_tanh_provider, method="rk4", show_progress=True
    )
    times_tanh = res_local_tanh["times"]
    phi_tanh = res_local_tanh["phi"]

    # Nonlocal provider (override epsilon if present) and solve
    eps_match = re.search(r"epsilon([0-9]*\.?[0-9]+)", data_path)
    provider_kwargs = {"frame_idx": 0}
    if eps_match:
        provider_kwargs["interaction_range"] = float(eps_match.group(1))
    nonlocal_provider = create_nonlocal_parameter_provider(data_path, **provider_kwargs)
    nonlocal_params = nonlocal_provider.get_params()
    res_nonlocal = solve_nonlocal_allen_cahn(
        nonlocal_provider, method="rk4", show_progress=True
    )
    times_nonlocal = res_nonlocal["times"]
    phi_nonlocal = res_nonlocal["phi"]

    # Reference data (from providers; all share same source)
    ref = local_poly_provider.get_reference_data()
    original_m = ref.get("macro_field")
    original_times = np.asarray(ref.get("times"))
    if original_m is None or original_times is None:
        print("⚠️ Reference data missing; cannot compare against data.")
        return

    # Magnetizations
    times, mags_all, Es_all, Fs_all, Chis_all = load_all_rounds(data_path, n_rounds=20)

    if mags_all is not None:
        original_mag = mags_all.mean(axis=0)
        original_mag_std = mags_all.std(axis=0)
        print(f"Std of original_mag: {original_mag_std}")
    else:
        original_mag = [np.mean(mf) for mf in original_m]
        original_mag_std = None
        print("⚠️ No magnetization data found; only single run available.")

    mag_poly = [np.mean(f) for f in phi_poly]
    mag_tanh = [np.mean(f) for f in phi_tanh]
    mag_non = [np.mean(f) for f in phi_nonlocal]

    # Align for time-series comparisons
    T_data = len(original_m)
    T_poly = len(phi_poly)
    T_tanh = len(phi_tanh)
    T_non = len(phi_nonlocal)
    min_len = min(T_data, T_poly, T_tanh, T_non)
    orig_idx = np.linspace(0, T_data - 1, min_len, dtype=int)
    poly_idx = np.linspace(0, T_poly - 1, min_len, dtype=int)
    tanh_idx = np.linspace(0, T_tanh - 1, min_len, dtype=int)
    non_idx = np.linspace(0, T_non - 1, min_len, dtype=int)
    times_aligned = original_times[orig_idx]

    # Mean absolute difference vs time
    mad_poly_ts = [
        np.mean(np.abs(phi_poly[poly_idx[i]] - original_m[orig_idx[i]]))
        for i in range(min_len)
    ]
    mad_tanh_ts = [
        np.mean(np.abs(phi_tanh[tanh_idx[i]] - original_m[orig_idx[i]]))
        for i in range(min_len)
    ]
    mad_non_ts = [
        np.mean(np.abs(phi_nonlocal[non_idx[i]] - original_m[orig_idx[i]]))
        for i in range(min_len)
    ]

    # Free energy vs time (use same functional)
    beta_val = float(local_poly_params.beta)
    h_field = float(local_poly_params.h_field)
    eps_val = float(local_poly_params.source_info.get("mapped", {}).get("eps", epsilon))
    M = original_m.shape[-1]
    kernel_k = build_kernel_fft(M, eps_val)

    if Fs_all is not None:
        tF_data = times
        F_data = Fs_all.mean(axis=0)
        F_data_std = Fs_all.std(axis=0)
        print(f"Std of F_data: {F_data_std}")
    else:
        tF_data, F_data = free_energy_series(
            original_m,
            original_times,
            beta=beta_val,
            h_field=h_field,
            J0=1.0,
            kernel_k=kernel_k,
        )
        F_data_std = None
        print("⚠️ No free energy data found; only single run available.")

    tF_poly, F_poly = free_energy_series(
        phi_poly,
        times_poly,
        beta=beta_val,
        h_field=h_field,
        J0=1.0,
        kernel_k=kernel_k,
    )
    tF_tanh, F_tanh = free_energy_series(
        phi_tanh,
        times_tanh,
        beta=beta_val,
        h_field=h_field,
        J0=1.0,
        kernel_k=kernel_k,
    )
    J0_non = float(nonlocal_params.interaction_strength)
    tF_non, F_non = free_energy_series(
        phi_nonlocal,
        times_nonlocal,
        beta=beta_val,
        h_field=h_field,
        J0=J0_non,
        kernel_k=kernel_k,
    )
    # Align FE series
    T_align_F = min(len(tF_data), len(tF_poly), len(tF_tanh), len(tF_non))
    tF = tF_data[:T_align_F]
    F_data = F_data[:T_align_F]
    F_poly = F_poly[:T_align_F]
    F_tanh = F_tanh[:T_align_F]
    F_non = F_non[:T_align_F]

    # Susceptibility χ(t)
    if Chis_all is not None:
        chi_data = Chis_all.mean(axis=0)
        chi_data_std = Chis_all.std(axis=0)
        print(f"Std of chi_data: {chi_data_std}")
    else:
        chi_data = compute_susceptibility_series(original_m, beta_val)
        chi_data_std = None
        print("⚠️ No susceptibility data found; only single run available.")
    chi_poly = compute_susceptibility_series(phi_poly, beta_val)
    chi_tanh = compute_susceptibility_series(phi_tanh, beta_val)
    chi_non = compute_susceptibility_series(phi_nonlocal, beta_val)
    chi_data_al = chi_data[orig_idx]
    chi_poly_al = chi_poly[poly_idx]
    chi_tanh_al = chi_tanh[tanh_idx]
    chi_non_al = chi_non[non_idx]

    # Figure 4x4
    fig, axes = plt.subplots(4, 4, figsize=(24, 24))

    # Row 1: original init, original final, error curves, magnetization
    im1 = axes[0, 0].imshow(original_m[0], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 0].set_title("Original - Initial (t=0)")
    plt.colorbar(im1, ax=axes[0, 0])
    im2 = axes[0, 1].imshow(original_m[-1], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 1].set_title("Original - Final")
    plt.colorbar(im2, ax=axes[0, 1])
    axes[0, 2].plot(
        times_aligned, mad_poly_ts, color="green", label="|Local-Poly−Data|"
    )
    axes[0, 2].plot(
        times_aligned,
        mad_tanh_ts,
        color="orange",
        linestyle="-.",
        label="|Local-Tanh−Data|",
    )
    axes[0, 2].plot(
        times_aligned, mad_non_ts, color="red", linestyle="--", label="|Nonlocal−Data|"
    )
    axes[0, 2].set_title("Mean |PDE − Data| vs Time")
    axes[0, 2].set_xlabel("Time")
    axes[0, 2].set_ylabel("Mean absolute difference")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    if original_mag_std is not None:
        axes[0, 3].plot(original_times, original_mag, color="blue", label="Data (mean)")
        axes[0, 3].fill_between(
            original_times,
            original_mag - scale_std * original_mag_std,
            original_mag + scale_std * original_mag_std,
            color="blue",
            alpha=0.2,
            label=f"±{scale_std}x std",
        )
    else:
        axes[0, 3].plot(original_times, original_mag, color="blue", label="Data")
    axes[0, 3].plot(times_poly, mag_poly, color="green", label="Local-Poly")
    axes[0, 3].plot(
        times_tanh, mag_tanh, color="orange", linestyle="-.", label="Local-Tanh"
    )
    axes[0, 3].plot(
        times_nonlocal, mag_non, color="red", linestyle="--", label="Nonlocal"
    )
    axes[0, 3].set_title("Magnetization vs Time")
    axes[0, 3].set_xlabel("Time")
    axes[0, 3].set_ylabel("Average Magnetization")
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.3)

    # Row 2: local-poly init, local-poly final, diff, Free Energy (all)
    im3 = axes[1, 0].imshow(phi_poly[0], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 0].set_title("Local-Poly - Initial")
    plt.colorbar(im3, ax=axes[1, 0])
    im4 = axes[1, 1].imshow(phi_poly[-1], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 1].set_title("Local-Poly - Final")
    plt.colorbar(im4, ax=axes[1, 1])
    diff_poly = np.abs(phi_poly[-1] - original_m[-1])
    im5 = axes[1, 2].imshow(diff_poly, cmap="hot")
    axes[1, 2].set_title("|Local-Poly − Data| (final)")
    plt.colorbar(im5, ax=axes[1, 2])
    # Free Energy
    axes[1, 3].plot(tF, F_data, color="blue", label="Data F[m]")
    if F_data_std is not None:
        axes[1, 3].fill_between(
            tF_data,
            F_data - scale_std * F_data_std,
            F_data + scale_std * F_data_std,
            color="blue",
            alpha=0.2,
            label=f"±{scale_std}x std",
        )
    axes[1, 3].plot(tF, F_poly, color="green", label="Local-Poly F[m]")
    axes[1, 3].plot(tF, F_tanh, color="orange", linestyle="-.", label="Local-Tanh F[m]")
    axes[1, 3].plot(tF, F_non, color="red", linestyle="--", label="Nonlocal F[m]")
    axes[1, 3].set_title("Free Energy vs Time")
    axes[1, 3].set_xlabel("Time")
    axes[1, 3].set_ylabel("Free Energy")
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)

    # Row 3: local-tanh init, local-tanh final, diff, (empty)
    im6 = axes[2, 0].imshow(phi_tanh[0], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2, 0].set_title("Local-Tanh - Initial")
    plt.colorbar(im6, ax=axes[2, 0])
    im7 = axes[2, 1].imshow(phi_tanh[-1], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2, 1].set_title("Local-Tanh - Final")
    plt.colorbar(im7, ax=axes[2, 1])
    diff_tanh = np.abs(phi_tanh[-1] - original_m[-1])
    im8 = axes[2, 2].imshow(diff_tanh, cmap="hot")
    axes[2, 2].set_title("|Local-Tanh − Data| (final)")
    plt.colorbar(im8, ax=axes[2, 2])
    axes[2, 3].axis("off")

    # Row 4: nonlocal init, nonlocal final, diff, Susceptibility (all)
    im9 = axes[3, 0].imshow(phi_nonlocal[0], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[3, 0].set_title("Nonlocal - Initial")
    plt.colorbar(im9, ax=axes[3, 0])
    im10 = axes[3, 1].imshow(phi_nonlocal[-1], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[3, 1].set_title("Nonlocal - Final")
    plt.colorbar(im10, ax=axes[3, 1])
    diff_non = np.abs(phi_nonlocal[-1] - original_m[-1])
    im11 = axes[3, 2].imshow(diff_non, cmap="hot")
    axes[3, 2].set_title("|Nonlocal − Data| (final)")
    plt.colorbar(im11, ax=axes[3, 2])
    # Susceptibility (Data, Local-Poly, Local-Tanh, Nonlocal)
    axes[3, 3].plot(times_aligned, chi_data_al, color="blue", label="Data χ(t)")
    if chi_data_std is not None:
        axes[3, 3].fill_between(
            times_aligned,
            chi_data_al - scale_std * chi_data_std,
            chi_data_al + scale_std * chi_data_std,
            color="blue",
            alpha=0.2,
            label=f"±{scale_std}x std",
        )
    axes[3, 3].plot(times_aligned, chi_poly_al, color="green", label="Local-Poly χ(t)")
    axes[3, 3].plot(
        times_aligned,
        chi_tanh_al,
        color="orange",
        linestyle="-.",
        label="Local-Tanh χ(t)",
    )
    axes[3, 3].plot(
        times_aligned, chi_non_al, color="red", linestyle="--", label="Nonlocal χ(t)"
    )
    axes[3, 3].set_title("Susceptibility vs Time")
    axes[3, 3].set_xlabel("Time")
    axes[3, 3].set_ylabel("Susceptibility χ")
    axes[3, 3].legend()
    axes[3, 3].grid(True, alpha=0.3)

    # Title and save
    T_val = (
        (1.0 / float(local_poly_params.beta))
        if float(local_poly_params.beta) != 0
        else float("nan")
    )
    h_val = float(local_poly_params.h_field)
    eps_title = eps_val
    fig.suptitle(
        f"PDE Comparison (Nonlocal vs Local-Tanh vs Local-Poly)\n(h={h_val:g}, T={T_val:g}, eps={eps_title:g})",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    os.makedirs("results", exist_ok=True)
    out = (
        f"results/compare_pde_solvers_L{L:g}_h{h_val:g}_T{T_val:g}_eps{eps_title:g}.png"
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"✅ Comparison plot saved to: {out}")


if __name__ == "__main__":
    main()
