#!/usr/bin/env python3
"""
Resolution study for PDE solvers vs data.

- Choose one solver: 'local-poly', 'local-tanh', or 'nonlocal'
- Run the same solver at resolutions N in {128, 256, 512}
- Initial condition is loaded from data at t=0 and resampled to each N (periodic bilinear)
- After evolution, resample back to data grid to compute |PDE - data| errors
- Plot 4 rows: Data (t=0, t=T, right columns: error vs time, magnetization vs time),
  then one row per resolution (N=128, 256, 512) with initial, final, and final |PDE−Data|
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append("src")

from typing import Dict, Tuple, List

from local_pde_solver import (
    create_local_parameter_provider,
    LocalParameterProvider,
    TheoryLocalProvider,
    solve_local_pde,
)
from nonlocal_pde_solver import (
    create_nonlocal_parameter_provider,
    NonlocalParameterProvider,
    TheoryNonlocalProvider,
    solve_nonlocal_allen_cahn,
)
from utils import (
    compute_susceptibility_series,
    load_all_rounds,
    build_kernel_fft,
    free_energy_series,
)


def resample_periodic(field: np.ndarray, M_target: int) -> np.ndarray:
    """
    Periodic bilinear resampling from M0xM0 to M_target x M_target.
    Assumes domain [0,1]^2 with uniform grids and periodicity.
    """
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError("field must be square 2D array")
    M0 = field.shape[0]
    if M_target == M0:
        return field.astype(np.float32, copy=True)

    # Target sample coordinates in source index space [0, M0)
    u = np.arange(M_target, dtype=np.float32) * (M0 / float(M_target))
    v = u
    uu, vv = np.meshgrid(u, v, indexing="ij")

    i0 = np.floor(uu).astype(np.int64) % M0
    j0 = np.floor(vv).astype(np.int64) % M0
    i1 = (i0 + 1) % M0
    j1 = (j0 + 1) % M0
    du = uu - np.floor(uu)
    dv = vv - np.floor(vv)

    f00 = field[i0, j0]
    f10 = field[i1, j0]
    f01 = field[i0, j1]
    f11 = field[i1, j1]

    out = (
        (1 - du) * (1 - dv) * f00
        + du * (1 - dv) * f10
        + (1 - du) * dv * f01
        + du * dv * f11
    )
    return out.astype(np.float32)


def _prepare_local_manual_provider(
    data_path: str, N: int, rhs_mode: str, record_every: int = 10
) -> Tuple[LocalParameterProvider, Dict[str, np.ndarray]]:
    """
    Build a ManualLocalProvider at resolution N using parameters mapped from data.
    """
    theory = TheoryLocalProvider(data_path=data_path, frame_idx=0)
    params = theory.get_params()
    ref = theory.get_reference_data()
    m0_data = ref["macro_field"][0].astype(np.float32)
    m0 = resample_periodic(m0_data, N)

    # Preserve timing
    dt = float(params.dt) if params.dt is not None else 0.01
    steps = int(params.steps)

    provider = create_local_parameter_provider(
        "manual",
        kappa=float(params.kappa),
        r=float(params.r),
        u=float(params.u),
        beta=float(params.beta),
        h_field=float(params.h_field),
        M=int(N),
        initial_condition=m0,
        rhs_mode=rhs_mode,
        dt=dt,
        steps=steps,
        record_every=record_every,
        J0=float(getattr(params, "J0", 1.0)),
    )
    return provider, ref


def _prepare_nonlocal_manual_provider(
    data_path: str, N: int, record_every: int = 10
) -> Tuple[NonlocalParameterProvider, Dict[str, np.ndarray]]:
    """
    Build a ManualNonlocalProvider at resolution N using parameters derived from data.
    """
    theory = TheoryNonlocalProvider(data_path=data_path, frame_idx=0)
    params = theory.get_params()
    ref = theory.get_reference_data()
    m0_data = ref["macro_field"][0].astype(np.float32)
    m0 = resample_periodic(m0_data, N)

    dt = float(params.dt) if params.dt is not None else 0.01
    steps = int(params.steps)

    provider = create_nonlocal_parameter_provider(
        "manual",
        interaction_strength=float(params.interaction_strength),
        interaction_range=float(params.interaction_range),
        mobility_prefactor=float(params.mobility_prefactor),
        h_field=float(params.h_field),
        M=int(N),
        initial_condition=m0,
        dt=dt,
        steps=steps,
        record_every=record_every,
    )
    return provider, ref


def run_one_resolution(
    solver_type: str,
    data_path: str,
    N: int,
    record_every: int = 10,
) -> Dict[str, np.ndarray]:
    solver_type = solver_type.lower()
    if solver_type == "local-poly":
        provider, ref = _prepare_local_manual_provider(
            data_path, N, rhs_mode="poly", record_every=record_every
        )
        result = solve_local_pde(provider, method="rk4", show_progress=True)
    elif solver_type == "local-tanh":
        provider, ref = _prepare_local_manual_provider(
            data_path, N, rhs_mode="tanh_local", record_every=record_every
        )
        result = solve_local_pde(provider, method="rk4", show_progress=True)
    elif solver_type == "nonlocal":
        provider, ref = _prepare_nonlocal_manual_provider(
            data_path, N, record_every=record_every
        )
        result = solve_nonlocal_allen_cahn(provider, method="rk4", show_progress=True)
    else:
        raise ValueError("solver_type must be one of: local-poly, local-tanh, nonlocal")

    return {
        "times": result["times"],
        "traj": result["phi"],
        "ref": ref,
        "params": result.get("params"),
    }


def compute_error_series(
    traj: np.ndarray, times: np.ndarray, data_traj: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean absolute error vs time between traj (T1, N, N) and data_traj (T2, M, M).
    Each PDE frame is resampled to data grid for comparison. Returns aligned times and errors.
    """
    T1 = traj.shape[0]
    T2 = data_traj.shape[0]
    T = min(T1, T2)
    errs = np.empty(T, dtype=np.float64)
    M_data = data_traj.shape[-1]
    for t in range(T):
        m_res = resample_periodic(traj[t], M_data)
        errs[t] = np.mean(np.abs(m_res - data_traj[t]))
    return times[:T], errs


def main():
    print("🧪 PDE Resolution Comparison")
    print("=" * 60)

    # --- Config ---
    solver_type = os.environ.get(
        "PDE_SOLVER_TYPE", "local-poly"
    )  # 'local-poly', 'local-tanh', 'nonlocal'
    resolutions: List[int] = [128, 256]
    record_every = (
        10  # store fewer frames to reduce memory; times align via subsampling
    )

    base_dir = "data/ct_glauber"
    h = 0
    T = 1
    epsilon = 0.0625
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

    print(f"Solver: {solver_type}")
    print(f"Data:   {data_path}")
    print(f"Ns:     {resolutions}")

    # Load reference data
    base_provider = TheoryLocalProvider(data_path=data_path, frame_idx=0)
    ref = base_provider.get_reference_data()
    data_traj = ref["macro_field"].astype(np.float32)
    data_times = ref["times"].astype(np.float32)

    # Run PDE at each resolution
    results_by_N: Dict[int, Dict[str, np.ndarray]] = {}
    for N in resolutions:
        print("-" * 60)
        print(f"Running N={N} ...")
        run = run_one_resolution(solver_type, data_path, N, record_every=record_every)
        results_by_N[N] = run

    # Compute error series and magnetizations
    err_series: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    mags_data = data_traj.reshape(data_traj.shape[0], -1).mean(axis=1)
    mags_by_N: Dict[int, np.ndarray] = {}
    for N, run in results_by_N.items():
        times = run["times"]
        traj = run["traj"]
        t_al, errs = compute_error_series(traj, times, data_traj)
        err_series[N] = (t_al, errs)
        mags_by_N[N] = traj.reshape(traj.shape[0], -1).mean(axis=1)

    # Prepare data stats (20 rounds) and kernel for energy
    # Use TheoryLocalProvider mapping for eps, beta, h
    try:
        params0 = base_provider.get_params()
        beta_val = float(params0.beta)
        h_field = float(params0.h_field)
        eps_val = float(params0.source_info.get("mapped", {}).get("eps", 0.0))
    except Exception:
        beta_val = 1.0
        h_field = 0.0
        eps_val = 0.0

    times_rounds, mags_all, Es_all, Fs_all, Chis_all = load_all_rounds(
        data_path, n_rounds=20
    )

    # Data magnetization
    if mags_all is not None:
        mag_data_mean = mags_all.mean(axis=0)
        mag_data_std = mags_all.std(axis=0)
        t_mag = times_rounds
    else:
        mag_data_mean = mags_data
        mag_data_std = None
        t_mag = data_times

    # Kernel for free energy calculations at data grid size
    M_data = data_traj.shape[-1]
    kernel_k = build_kernel_fft(M_data, eps_val)

    # Data energy and free energy
    if Es_all is not None:
        E_data_mean = Es_all.mean(axis=0)
        E_data_std = Es_all.std(axis=0)
        t_energy = times_rounds
    else:
        E_data_mean = None
        E_data_std = None
        t_energy = data_times

    if Fs_all is not None:
        F_data_mean = Fs_all.mean(axis=0)
        F_data_std = Fs_all.std(axis=0)
        tF_data = times_rounds
    else:
        tF_data, F_data_mean = free_energy_series(
            data_traj,
            data_times,
            beta=beta_val,
            h_field=h_field,
            J0=1.0,
            kernel_k=kernel_k,
        )
        F_data_std = None

    # Data susceptibility
    if Chis_all is not None:
        chi_data_mean = Chis_all.mean(axis=0)
        chi_data_std = Chis_all.std(axis=0)
        t_chi = times_rounds
    else:
        chi_single = compute_susceptibility_series(data_traj, beta_val)
        chi_data_mean = chi_single
        chi_data_std = None
        t_chi = data_times

    # Distinguishable styles per resolution
    color_map = {128: "tab:green", 256: "tab:orange", 512: "tab:red"}
    style_map = {
        128: "-",
        256: "--",
        512: "-.",
    }

    # Figure: 4 rows x 6 cols (add Energy and Susceptibility panels)
    fig, axes = plt.subplots(4, 6, figsize=(34, 22))

    # Row 0: Data snapshots, Error vs time (all N), Magnetization vs time (data + all N)
    im0 = axes[0, 0].imshow(data_traj[0], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 0].set_title("Data - Initial (t=0)")
    plt.colorbar(im0, ax=axes[0, 0])
    im1 = axes[0, 1].imshow(data_traj[-1], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 1].set_title("Data - Final")
    plt.colorbar(im1, ax=axes[0, 1])

    # Error vs time (distinct styles) with legend ordered as in `resolutions`
    handles_err = []
    labels_err = []
    for N in resolutions:
        tN, eN = err_series[N]
        (h_line,) = axes[0, 2].plot(
            tN,
            eN,
            label=f"N={N}",
            color=color_map.get(N),
            linestyle=style_map.get(N, "-"),
        )
        handles_err.append(h_line)
        labels_err.append(f"N={N}")
    axes[0, 2].set_title("Mean |PDE − Data| vs Time")
    axes[0, 2].set_xlabel("Time")
    axes[0, 2].set_ylabel("Mean absolute error")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend(handles=handles_err, labels=labels_err)

    # Magnetization vs time (data mean±std and PDE lines)
    handles_mag = []
    labels_mag = []
    if mag_data_std is not None:
        (h_data,) = axes[0, 3].plot(
            t_mag, mag_data_mean, color="black", label="Data (mean)"
        )
        handles_mag.append(h_data)
        labels_mag.append("Data (mean)")
        axes[0, 3].fill_between(
            t_mag,
            mag_data_mean - scale_std * mag_data_std,
            mag_data_mean + scale_std * mag_data_std,
            color="gray",
            alpha=0.2,
            label="Data ±20σ",
        )
    else:
        (h_data,) = axes[0, 3].plot(t_mag, mag_data_mean, color="black", label="Data")
        handles_mag.append(h_data)
        labels_mag.append("Data")
    for N in resolutions:
        tN = results_by_N[N]["times"]
        mN = mags_by_N[N]
        Tplot = min(len(tN), len(mN))
        (h_line,) = axes[0, 3].plot(
            tN[:Tplot],
            mN[:Tplot],
            label=f"N={N}",
            color=color_map.get(N),
            linestyle=style_map.get(N, "-"),
        )
        handles_mag.append(h_line)
        labels_mag.append(f"N={N}")
    axes[0, 3].set_title("Magnetization vs Time")
    axes[0, 3].set_xlabel("Time")
    axes[0, 3].set_ylabel("⟨m⟩")
    axes[0, 3].grid(True, alpha=0.3)
    axes[0, 3].legend(handles=handles_mag, labels=labels_mag)

    # Free Energy vs time (Data F[m] ± std; PDE F[m] for each N)
    handles_fe = []
    labels_fe = []
    if F_data_mean is not None:
        (h_dataF,) = axes[0, 4].plot(
            tF_data, F_data_mean, color="black", linestyle=":", label="Data F[m]"
        )
        handles_fe.append(h_dataF)
        labels_fe.append("Data F[m]")
        if F_data_std is not None:
            axes[0, 4].fill_between(
                tF_data,
                F_data_mean - scale_std * F_data_std,
                F_data_mean + scale_std * F_data_std,
                color="black",
                alpha=0.1,
                label="Data F ±20σ",
            )
    # PDE Free energy curves (computed on data grid)
    for N in resolutions:
        traj = results_by_N[N]["traj"]
        # resample traj to data grid
        Tn = traj.shape[0]
        traj_res = np.empty((Tn, M_data, M_data), dtype=np.float32)
        for i in range(Tn):
            traj_res[i] = resample_periodic(traj[i], M_data)
        # choose J0 for free energy: local models J0=1, nonlocal use interaction_strength if available
        paramsN = results_by_N[N]["params"]
        J0_val = float(getattr(paramsN, "interaction_strength", 1.0))
        tF_N, F_N = free_energy_series(
            traj_res,
            results_by_N[N]["times"],
            beta=beta_val,
            h_field=h_field,
            J0=J0_val,
            kernel_k=kernel_k,
        )
        (h_line,) = axes[0, 4].plot(
            tF_N,
            F_N,
            label=f"F[m], N={N}",
            color=color_map.get(N),
            linestyle=style_map.get(N, "-"),
        )
        handles_fe.append(h_line)
        labels_fe.append(f"F[m], N={N}")
    axes[0, 4].set_title("Free Energy vs Time")
    axes[0, 4].set_xlabel("Time")
    axes[0, 4].set_ylabel("F[m]")
    axes[0, 4].grid(True, alpha=0.3)
    axes[0, 4].legend(handles=handles_fe, labels=labels_fe)

    # Susceptibility vs time (Data ± std and PDE chi for each N)
    handles_chi = []
    labels_chi = []
    (h_data_chi,) = axes[0, 5].plot(
        t_chi, chi_data_mean, color="blue", label="Data χ (mean)"
    )
    handles_chi.append(h_data_chi)
    labels_chi.append("Data χ (mean)")
    if chi_data_std is not None:
        axes[0, 5].fill_between(
            t_chi,
            chi_data_mean - scale_std * chi_data_std,
            chi_data_mean + scale_std * chi_data_std,
            color="blue",
            alpha=0.15,
            label="Data χ ±20σ",
        )
    for N in resolutions:
        # Resample PDE trajectory to data grid before computing χ to avoid resolution scaling
        traj = results_by_N[N]["traj"]
        Tn = traj.shape[0]
        traj_res = np.empty((Tn, M_data, M_data), dtype=np.float32)
        for i in range(Tn):
            traj_res[i] = resample_periodic(traj[i], M_data)
        chi_N = compute_susceptibility_series(traj_res, beta_val)
        tN = results_by_N[N]["times"]
        Tplot = min(len(tN), len(chi_N))
        (h_line,) = axes[0, 5].plot(
            tN[:Tplot],
            chi_N[:Tplot],
            label=f"χ, N={N}",
            color=color_map.get(N),
            linestyle=style_map.get(N, "-"),
        )
        handles_chi.append(h_line)
        labels_chi.append(f"χ, N={N}")
    axes[0, 5].set_title("Susceptibility vs Time")
    axes[0, 5].set_xlabel("Time")
    axes[0, 5].set_ylabel("χ(t)")
    axes[0, 5].grid(True, alpha=0.3)
    axes[0, 5].legend(handles=handles_chi, labels=labels_chi)

    # Rows 1-3: For each N show initial, final, final |PDE − Data|
    for row_idx, N in enumerate(resolutions, start=1):
        traj = results_by_N[N]["traj"]
        im2 = axes[row_idx, 0].imshow(traj[0], cmap="RdBu_r", vmin=-1, vmax=1)
        axes[row_idx, 0].set_title(f"N={N} - Initial")
        plt.colorbar(im2, ax=axes[row_idx, 0])
        im3 = axes[row_idx, 1].imshow(traj[-1], cmap="RdBu_r", vmin=-1, vmax=1)
        axes[row_idx, 1].set_title(f"N={N} - Final")
        plt.colorbar(im3, ax=axes[row_idx, 1])
        # Final absolute difference (resampled)
        final_res = resample_periodic(traj[-1], data_traj.shape[-1])
        diff = np.abs(final_res - data_traj[-1])
        im4 = axes[row_idx, 2].imshow(diff, cmap="hot")
        axes[row_idx, 2].set_title(f"|N={N} − Data| (final)")
        plt.colorbar(im4, ax=axes[row_idx, 2])
        # Leave last columns for per-row annotation
        axes[row_idx, 3].axis("off")
        axes[row_idx, 4].axis("off")
        axes[row_idx, 5].axis("off")

    # Title and save
    title_solver = {
        "local-poly": "Local (poly)",
        "local-tanh": "Local (tanh)",
        "nonlocal": "Nonlocal",
    }.get(solver_type, solver_type)
    fig.suptitle(
        f"PDE Resolution Study ({title_solver})",
        fontsize=18,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs("results", exist_ok=True)
    out = f"results/test_pde_resolution_h{h}_T{T}_eps{epsilon}_L{L}_solver{solver_type}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"✅ Resolution comparison figure saved to: {out}")


def test_pde_resolution():
    # Run with default env or pytest
    main()


if __name__ == "__main__":
    main()
