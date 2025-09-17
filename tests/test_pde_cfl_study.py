#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("src")

from local_pde_solver import (
    create_local_parameter_provider,
    TheoryLocalProvider,
    solve_local_pde,
)
from utils import (
    build_kernel_fft,
    free_energy_series,
    compute_susceptibility_series,
    load_all_rounds,
)


def resample_periodic(field: np.ndarray, M_target: int) -> np.ndarray:
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError("field must be square 2D array")
    M0 = field.shape[0]
    if M_target == M0:
        return field.astype(np.float32, copy=True)
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


def compute_error_series(traj: np.ndarray, times: np.ndarray, data_traj: np.ndarray):
    T1 = traj.shape[0]
    T2 = data_traj.shape[0]
    T = min(T1, T2)
    errs = np.empty(T, dtype=np.float64)
    M_data = data_traj.shape[-1]
    for t in range(T):
        m_res = resample_periodic(traj[t], M_data)
        errs[t] = np.mean(np.abs(m_res - data_traj[t]))
    return times[:T], errs


def run_local_case(data_path: str, N: int, dt: float, rhs_mode: str, enforce_cfl: bool):
    base = TheoryLocalProvider(data_path=data_path, frame_idx=0)
    p = base.get_params()
    ref = base.get_reference_data()
    m0 = resample_periodic(ref["macro_field"][0].astype(np.float32), N)

    provider = create_local_parameter_provider(
        "manual",
        kappa=float(p.kappa),
        r=float(p.r),
        u=float(p.u),
        beta=float(p.beta),
        h_field=float(p.h_field),
        M=int(N),
        initial_condition=m0,
        rhs_mode=rhs_mode,
        dt=float(dt),
        steps=int(p.steps),
        record_every=10,
        J0=float(getattr(p, "J0", 1.0)),
        enforce_cfl=bool(enforce_cfl),
    )
    out = solve_local_pde(provider, method="rk4", show_progress=True)
    return out


def main():
    # Data path (same as resolution test)
    base_dir = "data/ct_glauber"
    h = 0
    T = 1
    epsilon = 0.0625
    L = 1024
    ell = 32
    block = 8
    t_end = 20.0
    data_path = os.path.join(
        base_dir,
        f"ct_glauber_L{L}_ell{ell}_sigma1.2_tau1_m00.1_T{T}_J1_h{h}_"
        f"tend{t_end}_dt0.01_block{block}_kernelgaussian_epsilon{epsilon}_seed0",
        "round0",
        f"ct_glauber_L{L}_ell{ell}_sigma1.2_tau1_m00.1_T{T}_J1_h{h}_"
        f"tend{t_end}_dt0.01_block{block}_kernelgaussian_epsilon{epsilon}_seed0_round0.npz",
    )
    if not os.path.exists(data_path):
        print(f"❌ Data not found: {data_path}")
        return

    # Cases to run: (label, N, dt, rhs_mode, enforce_cfl)
    cases = [
        ("N=128, dt=1e-2, CFL off", 128, 1e-2, "poly", False),
        ("N=128, dt=1e-2, CFL on", 128, 1e-2, "poly", True),
        ("N=256, dt=1e-2, CFL off", 256, 1e-2, "poly", False),
        ("N=256, dt=1e-2, CFL on", 256, 1e-2, "poly", True),
        # ("N=512, dt=1e-2, CFL on", 512, 1e-2, "poly", True),
    ]

    # Load reference data
    ref_provider = TheoryLocalProvider(data_path=data_path, frame_idx=0)
    ref = ref_provider.get_reference_data()
    data_traj = ref["macro_field"].astype(np.float32)

    results = []
    for label, N, dt, rhs_mode, enforce_cfl in cases:
        print("-" * 60)
        print(f"Running {label}")
        out = run_local_case(data_path, N, dt, rhs_mode, enforce_cfl)
        t_err, e = compute_error_series(out["phi"], out["times"], data_traj)
        # Store inputs and effective dt for legend construction
        results.append(
            {
                "label": label,
                "N": N,
                "dt_in": dt,
                "cfl": enforce_cfl,
                "out": out,
                "t_err": t_err,
                "err": e,
            }
        )

    # Build metrics for data
    data_times = ref_provider.get_reference_data()["times"]
    mags_data = data_traj.reshape(data_traj.shape[0], -1).mean(axis=1)

    # Use parameter mapping for beta/h/eps; fall back safely
    try:
        params0 = ref_provider.get_params()
        beta_val = float(params0.beta)
        h_field = float(params0.h_field)
        eps_val = float(params0.source_info.get("mapped", {}).get("eps", epsilon))
    except Exception:
        beta_val = 1.0
        h_field = float(h)
        eps_val = float(epsilon)

    M_data = data_traj.shape[-1]
    kernel_k = build_kernel_fft(M_data, eps_val)
    # Load multi-round trajectories and compute all metrics from scratch
    times_rounds, mags_all, Es_all, Fs_all, Chis_all = load_all_rounds(
        data_path, n_rounds=20
    )
    scale_std = 20

    # Check if we have multi-round data by looking for trajectory files
    import glob

    data_dir = os.path.dirname(data_path)  # This should be the round0 directory
    parent_dir = os.path.dirname(
        data_dir
    )  # This should be the main experiment directory
    pattern = os.path.join(parent_dir, "round*", "*.npz")
    round_files = glob.glob(pattern)

    print(f"🔍 Looking for rounds in: {parent_dir}")
    print(f"📁 Found {len(round_files)} round files")

    if len(round_files) > 1:
        # Load all round trajectories and compute metrics from scratch
        all_trajs = []
        all_times = []

        # Sort by round number to ensure proper ordering
        def extract_round_num(filepath):
            import re

            match = re.search(r"round(\d+)", filepath)
            return int(match.group(1)) if match else 0

        sorted_files = sorted(round_files, key=extract_round_num)
        for round_file in sorted_files[:20]:  # limit to 20 rounds
            try:
                data = np.load(round_file)
                if "macro_field" in data and "times" in data:
                    all_trajs.append(data["macro_field"].astype(np.float32))
                    all_times.append(data["times"])
                    print(
                        f"✅ Loaded round {extract_round_num(round_file)}: shape {data['macro_field'].shape}"
                    )
            except Exception as e:
                print(f"❌ Failed to load {round_file}: {e}")
                continue

        if len(all_trajs) > 1:
            # Ensure all trajectories have same time grid
            min_len = min(len(t) for t in all_times)
            all_trajs = [traj[:min_len] for traj in all_trajs]
            times_common = all_times[0][:min_len]

            print(f"📊 Successfully loaded {len(all_trajs)} trajectories")
            print(f"⏱️  Using common time grid of length {min_len}")
            print(f"🎯 Computing statistics with {len(all_trajs)} rounds")

            # Compute magnetization for all rounds
            mags_all_computed = np.array(
                [traj.reshape(traj.shape[0], -1).mean(axis=1) for traj in all_trajs]
            )
            t_mag = times_common
            mag_mean = mags_all_computed.mean(axis=0)
            mag_std = mags_all_computed.std(axis=0)

            # Compute free energy for all rounds
            Fs_all_computed = np.array(
                [
                    free_energy_series(
                        traj,
                        times_common,
                        beta=beta_val,
                        h_field=h_field,
                        J0=1.0,
                        kernel_k=kernel_k,
                    )[1]
                    for traj in all_trajs
                ]
            )
            tF_data = times_common
            F_mean = Fs_all_computed.mean(axis=0)
            F_std = Fs_all_computed.std(axis=0)

            # Compute susceptibility for all rounds
            Chis_all_computed = np.array(
                [compute_susceptibility_series(traj, beta_val) for traj in all_trajs]
            )
            t_chi = times_common
            chi_mean = Chis_all_computed.mean(axis=0)
            chi_std = Chis_all_computed.std(axis=0)
        else:
            # Fallback to single trajectory
            t_mag = data_times
            mag_mean = mags_data
            mag_std = None
            tF_data, F_mean = free_energy_series(
                data_traj,
                data_times,
                beta=beta_val,
                h_field=h_field,
                J0=1.0,
                kernel_k=kernel_k,
            )
            F_std = None
            t_chi = data_times
            chi_mean = compute_susceptibility_series(data_traj, beta_val)
            chi_std = None
    else:
        # Fallback to single trajectory
        t_mag = data_times
        mag_mean = mags_data
        mag_std = None
        tF_data, F_mean = free_energy_series(
            data_traj,
            data_times,
            beta=beta_val,
            h_field=h_field,
            J0=1.0,
            kernel_k=kernel_k,
        )
        F_std = None
        t_chi = data_times
        chi_mean = compute_susceptibility_series(data_traj, beta_val)
        chi_std = None

    # Figure: like resolution study: rows = 2 (data row + metrics row is combined in row 0) + len(cases)
    nrows = 1 + len(cases)
    ncols = 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(34, 6 + 5 * nrows))

    # Row 0: Data init/final + metrics (errors, mag, free energy, susceptibility)
    im0 = axes[0, 0].imshow(data_traj[0], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 0].set_title("Data - Initial (t=0)")
    plt.colorbar(im0, ax=axes[0, 0])
    im1 = axes[0, 1].imshow(data_traj[-1], cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0, 1].set_title("Data - Final")
    plt.colorbar(im1, ax=axes[0, 1])

    # Errors vs time with detailed legend including dt_eff
    for rec in results:
        dt_eff = float(rec["out"].get("dt", rec["dt_in"]))
        lbl = f"N={rec['N']}, dt_in={rec['dt_in']:.2e}, {'CFL on' if rec['cfl'] else 'CFL off'}, dt_eff={dt_eff:.2e}"
        axes[0, 2].plot(rec["t_err"], rec["err"], label=lbl)
    axes[0, 2].set_title("Mean |PDE − Data| vs Time")
    axes[0, 2].set_xlabel("Time")
    axes[0, 2].set_ylabel("Mean absolute error")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()

    # Leave top-row metric panels empty; distribute metrics per-case rows
    for col in (3, 4, 5):
        axes[0, col].axis("off")

    # Rows 1..: per-case snapshots
    metrics_order = ["mag", "F", "chi"]
    for row_idx, rec in enumerate(results, start=1):
        traj = rec["out"]["phi"]
        im2 = axes[row_idx, 0].imshow(traj[0], cmap="RdBu_r", vmin=-1, vmax=1)
        axes[row_idx, 0].set_title(f"{rec['label']} - Init")
        plt.colorbar(im2, ax=axes[row_idx, 0])
        im3 = axes[row_idx, 1].imshow(traj[-1], cmap="RdBu_r", vmin=-1, vmax=1)
        axes[row_idx, 1].set_title(f"{rec['label']} - Final")
        plt.colorbar(im3, ax=axes[row_idx, 1])
        final_res = traj[-1]
        # resample to data grid for diff
        m = final_res
        u = np.arange(M_data, dtype=np.float32) * (m.shape[0] / float(M_data))
        v = u
        uu, vv = np.meshgrid(u, v, indexing="ij")
        i0 = np.floor(uu).astype(np.int64) % m.shape[0]
        j0 = np.floor(vv).astype(np.int64) % m.shape[1]
        i1 = (i0 + 1) % m.shape[0]
        j1 = (j0 + 1) % m.shape[1]
        du = uu - np.floor(uu)
        dv = vv - np.floor(vv)
        f00 = m[i0, j0]
        f10 = m[i1, j0]
        f01 = m[i0, j1]
        f11 = m[i1, j1]
        final_resampled = (
            (1 - du) * (1 - dv) * f00
            + du * (1 - dv) * f10
            + (1 - du) * dv * f01
            + du * dv * f11
        ).astype(np.float32)
        diff = np.abs(final_resampled - data_traj[-1])
        im4 = axes[row_idx, 2].imshow(diff, cmap="hot")
        axes[row_idx, 2].set_title("|PDE − Data| (final)")
        plt.colorbar(im4, ax=axes[row_idx, 2])
        # choose metric for this row (col=3)
        metric = metrics_order[(row_idx - 1) % len(metrics_order)]
        if metric == "mag":
            # Data
            axes[row_idx, 3].plot(t_mag, mag_mean, color="black", label="Data (mean)")
            if mag_std is not None:
                axes[row_idx, 3].fill_between(
                    t_mag,
                    mag_mean - scale_std * mag_std,
                    mag_mean + scale_std * mag_std,
                    color="black",
                    alpha=0.12,
                    label=f"Data ±{scale_std}σ",
                )
            # All PDE cases with different line styles
            line_styles = ["-", "--", "-.", ":"]
            for idx, rec2 in enumerate(results):
                mN2 = (
                    rec2["out"]["phi"]
                    .reshape(rec2["out"]["phi"].shape[0], -1)
                    .mean(axis=1)
                )
                tN2 = rec2["out"]["times"]
                dt_eff2 = float(rec2["out"].get("dt", rec2["dt_in"]))
                style = line_styles[idx % len(line_styles)]
                axes[row_idx, 3].plot(
                    tN2,
                    mN2,
                    linestyle=style,
                    label=f"N={rec2['N']} (dt_eff={dt_eff2:.2e})",
                )
            axes[row_idx, 3].set_title("Magnetization vs Time")
            axes[row_idx, 3].set_xlabel("Time")
            axes[row_idx, 3].set_ylabel("⟨m⟩")
            axes[row_idx, 3].grid(True, alpha=0.3)
            axes[row_idx, 3].legend()
        elif metric == "F":
            axes[row_idx, 3].plot(
                tF_data, F_mean, color="black", linestyle=":", label="Data F[m]"
            )
            if F_std is not None:
                axes[row_idx, 3].fill_between(
                    tF_data,
                    F_mean - scale_std * F_std,
                    F_mean + scale_std * F_std,
                    color="black",
                    alpha=0.12,
                    label=f"Data ±{scale_std}σ",
                )
            # All PDE cases (resampled to data grid) with different line styles
            line_styles = ["-", "--", "-.", ":"]
            for idx, rec2 in enumerate(results):
                traj2 = rec2["out"]["phi"]
                Tn2 = traj2.shape[0]
                traj_res2 = np.empty((Tn2, M_data, M_data), dtype=np.float32)
                for i in range(Tn2):
                    m = traj2[i]
                    u = np.arange(M_data, dtype=np.float32) * (
                        m.shape[0] / float(M_data)
                    )
                    v = u
                    uu, vv = np.meshgrid(u, v, indexing="ij")
                    i0 = np.floor(uu).astype(np.int64) % m.shape[0]
                    j0 = np.floor(vv).astype(np.int64) % m.shape[1]
                    i1 = (i0 + 1) % m.shape[0]
                    j1 = (j0 + 1) % m.shape[1]
                    du = uu - np.floor(uu)
                    dv = vv - np.floor(vv)
                    f00 = m[i0, j0]
                    f10 = m[i1, j0]
                    f01 = m[i0, j1]
                    f11 = m[i1, j1]
                    traj_res2[i] = (
                        (1 - du) * (1 - dv) * f00
                        + du * (1 - dv) * f10
                        + (1 - du) * dv * f01
                        + du * dv * f11
                    ).astype(np.float32)
                tF2, F2 = free_energy_series(
                    traj_res2,
                    rec2["out"]["times"],
                    beta=beta_val,
                    h_field=h_field,
                    J0=1.0,
                    kernel_k=kernel_k,
                )
                dt_eff2 = float(rec2["out"].get("dt", rec2["dt_in"]))
                style = line_styles[idx % len(line_styles)]
                axes[row_idx, 3].plot(
                    tF2,
                    F2,
                    linestyle=style,
                    label=f"N={rec2['N']} (dt_eff={dt_eff2:.2e})",
                )
            axes[row_idx, 3].set_title("Free Energy vs Time")
            axes[row_idx, 3].set_xlabel("Time")
            axes[row_idx, 3].set_ylabel("F[m]")
            axes[row_idx, 3].grid(True, alpha=0.3)
            axes[row_idx, 3].legend()
        else:  # chi
            axes[row_idx, 3].plot(t_chi, chi_mean, color="black", label="Data χ (mean)")
            if chi_std is not None:
                axes[row_idx, 3].fill_between(
                    t_chi,
                    chi_mean - scale_std * chi_std,
                    chi_mean + scale_std * chi_std,
                    color="black",
                    alpha=0.12,
                    label=f"Data ±{scale_std}σ",
                )
            # All PDE cases (resampled χ) with different line styles
            line_styles = ["-", "--", "-.", ":"]
            for idx, rec2 in enumerate(results):
                traj2 = rec2["out"]["phi"]
                Tn2 = traj2.shape[0]
                traj_res2 = np.empty((Tn2, M_data, M_data), dtype=np.float32)
                for i in range(Tn2):
                    m = traj2[i]
                    u = np.arange(M_data, dtype=np.float32) * (
                        m.shape[0] / float(M_data)
                    )
                    v = u
                    uu, vv = np.meshgrid(u, v, indexing="ij")
                    i0 = np.floor(uu).astype(np.int64) % m.shape[0]
                    j0 = np.floor(vv).astype(np.int64) % m.shape[1]
                    i1 = (i0 + 1) % m.shape[0]
                    j1 = (j0 + 1) % m.shape[1]
                    du = uu - np.floor(uu)
                    dv = vv - np.floor(vv)
                    f00 = m[i0, j0]
                    f10 = m[i1, j0]
                    f01 = m[i0, j1]
                    f11 = m[i1, j1]
                    traj_res2[i] = (
                        (1 - du) * (1 - dv) * f00
                        + du * (1 - dv) * f10
                        + (1 - du) * dv * f01
                        + du * dv * f11
                    ).astype(np.float32)
                chi2 = compute_susceptibility_series(traj_res2, beta_val)
                tN2 = rec2["out"]["times"]
                dt_eff2 = float(rec2["out"].get("dt", rec2["dt_in"]))
                style = line_styles[idx % len(line_styles)]
                axes[row_idx, 3].plot(
                    tN2,
                    chi2,
                    linestyle=style,
                    label=f"N={rec2['N']} (dt_eff={dt_eff2:.2e})",
                )
            axes[row_idx, 3].set_title("Susceptibility vs Time")
            axes[row_idx, 3].set_xlabel("Time")
            axes[row_idx, 3].set_ylabel("χ(t)")
            axes[row_idx, 3].grid(True, alpha=0.3)
            axes[row_idx, 3].legend()
        # hide unused columns
        for col in (4, 5):
            axes[row_idx, col].axis("off")

    # Save
    os.makedirs("results", exist_ok=True)
    out_path = f"results/pde_cfl_study_poly_h={h}_T={T}_eps={epsilon}_L={L}.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✅ Saved: {out_path}")


def test_pde_cfl_study():
    main()


if __name__ == "__main__":
    main()
