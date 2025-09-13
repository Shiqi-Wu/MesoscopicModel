import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import time

sys.path.append("src")
from kac_glauber2d_gpu import Glauber2DIsingKacGPU
from init_grf_ifft import grf_gaussian_spectrum_m_field, sample_spins_from_m_field


def test_compare_methods(
    model,
    spin_init,
    beta=0.4,
    J0=1.0,
    h=0.0,
    epsilon=0.05,
    t_end=5.0,
    snapshot_dt=0.1,
    n_runs=5,
    stem="test_results",
    output_dir=".",
):
    """
    Compare Gillespie vs tau-leaping simulation with multiple runs and averaging.
    model: your Glauber-Kac simulator class instance
    n_runs: number of independent runs with the same initial condition
    """

    # Storage for all runs
    all_Ms_g, all_Es_g = [], []
    all_Ms_t, all_Es_t = [], []
    all_times_g, all_times_t = [], []
    total_gillespie_time = 0
    total_tauleap_time = 0

    for run_idx in range(n_runs):
        print(f"\n=== Run {run_idx + 1}/{n_runs} ===")

        # Run tau-leaping
        print(f"Running tau-leaping (run {run_idx + 1})...")
        start_time = time.time()
        times_t, Ms_t, Es_t, snaps_t = model.simulate_kac_tauleap(
            spin_init.copy(),
            beta,
            J0,
            h,
            epsilon,
            t_end=t_end,
            snapshot_dt=snapshot_dt,
            return_snapshots=True,
            verbose_every=10,
        )
        tauleap_time = time.time() - start_time
        total_tauleap_time += tauleap_time
        print(f"Tau-leaping run {run_idx + 1} completed in {tauleap_time:.2f}s")

        # Run Gillespie
        print(f"Running Gillespie (run {run_idx + 1})...")
        start_time = time.time()
        times_g, Ms_g, Es_g, snaps_g = model.simulate_kac(
            spin_init.copy(),
            beta,
            J0,
            h,
            epsilon,
            t_end=t_end,
            snapshot_dt=snapshot_dt,
            return_snapshots=True,
            verbose_every=1000,
        )
        gillespie_time = time.time() - start_time
        total_gillespie_time += gillespie_time
        print(f"Gillespie run {run_idx + 1} completed in {gillespie_time:.2f}s")

        # Store results
        all_Ms_g.append(Ms_g)
        all_Es_g.append(Es_g)
        all_Ms_t.append(Ms_t)
        all_Es_t.append(Es_t)
        all_times_g.append(times_g)
        all_times_t.append(times_t)

        # Save snapshots from the first run only
        if run_idx == 0:
            first_run_snaps_g = snaps_g
            first_run_snaps_t = snaps_t
            first_run_times_g = times_g
            first_run_times_t = times_t

    # Find common time grid for averaging
    min_len_g = min(len(Ms) for Ms in all_Ms_g)
    min_len_t = min(len(Ms) for Ms in all_Ms_t)
    min_len = min(min_len_g, min_len_t)

    # Truncate all arrays to common length and convert to arrays
    Ms_g_array = np.array([Ms[:min_len] for Ms in all_Ms_g])
    Es_g_array = np.array([Es[:min_len] for Es in all_Es_g])
    Ms_t_array = np.array([Ms[:min_len] for Ms in all_Ms_t])
    Es_t_array = np.array([Es[:min_len] for Es in all_Es_t])

    # Compute averages and standard deviations
    Ms_g_mean = np.mean(Ms_g_array, axis=0)
    Ms_g_std = np.std(Ms_g_array, axis=0)
    Es_g_mean = np.mean(Es_g_array, axis=0)
    Es_g_std = np.std(Es_g_array, axis=0)

    Ms_t_mean = np.mean(Ms_t_array, axis=0)
    Ms_t_std = np.std(Ms_t_array, axis=0)
    Es_t_mean = np.mean(Es_t_array, axis=0)
    Es_t_std = np.std(Es_t_array, axis=0)

    times = all_times_g[0][:min_len]

    # Compute errors
    M_err = np.linalg.norm(Ms_g_mean - Ms_t_mean) / np.linalg.norm(Ms_g_mean)
    E_err = np.linalg.norm(Es_g_mean - Es_t_mean) / np.linalg.norm(Es_g_mean)
    print(f"\n=== Averaged Results (N={n_runs} runs) ===")
    print(f"Relative error in averaged Magnetization: {M_err:.3e}")
    print(f"Relative error in averaged Energy: {E_err:.3e}")

    # Plot averaged results with error bars
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Magnetization with error bars
    axs[0, 0].errorbar(
        times,
        Ms_g_mean,
        yerr=Ms_g_std,
        label=f"Gillespie (N={n_runs})",
        alpha=0.8,
        errorevery=len(times) // 10,
    )
    axs[0, 0].errorbar(
        times,
        Ms_t_mean,
        yerr=Ms_t_std,
        label=f"Tau-leaping (N={n_runs})",
        linestyle="--",
        alpha=0.8,
        errorevery=len(times) // 10,
    )
    axs[0, 0].set_title("Averaged Magnetization vs Time")
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Magnetization")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Energy with error bars
    axs[0, 1].errorbar(
        times,
        Es_g_mean,
        yerr=Es_g_std,
        label=f"Gillespie (N={n_runs})",
        alpha=0.8,
        errorevery=len(times) // 10,
    )
    axs[0, 1].errorbar(
        times,
        Es_t_mean,
        yerr=Es_t_std,
        label=f"Tau-leaping (N={n_runs})",
        linestyle="--",
        alpha=0.8,
        errorevery=len(times) // 10,
    )
    axs[0, 1].set_title("Averaged Energy vs Time")
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Energy")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # Difference in averages
    axs[1, 0].plot(times, Ms_g_mean - Ms_t_mean, "r-", linewidth=2)
    axs[1, 0].fill_between(
        times,
        (Ms_g_mean - Ms_g_std) - (Ms_t_mean + Ms_t_std),
        (Ms_g_mean + Ms_g_std) - (Ms_t_mean - Ms_t_std),
        alpha=0.3,
        color="red",
    )
    axs[1, 0].set_title("Magnetization Difference (G - T)")
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("ΔM")
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(times, Es_g_mean - Es_t_mean, "r-", linewidth=2)
    axs[1, 1].fill_between(
        times,
        (Es_g_mean - Es_g_std) - (Es_t_mean + Es_t_std),
        (Es_g_mean + Es_g_std) - (Es_t_mean - Es_t_std),
        alpha=0.3,
        color="red",
    )
    axs[1, 1].set_title("Energy Difference (G - T)")
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_ylabel("ΔE")
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{stem}_compare_N{n_runs}.png"), dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}/{stem}_compare_N{n_runs}.png")

    # Print a few snapshots from the first run
    n_show = min(3, first_run_snaps_g.shape[0])
    indices = np.random.choice(first_run_snaps_g.shape[0], n_show, replace=False)
    print(f"Showing snapshots from first run for indices: {indices}")
    snaps_g = first_run_snaps_g[indices]
    snaps_t = first_run_snaps_t[indices]
    times_g = first_run_times_g[indices]
    times_t = first_run_times_t[indices]
    for i in range(n_show):
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(snaps_g[i], cmap="gray", vmin=-1, vmax=1)
        axs[0].set_title(f"Gillespie t={times_g[i]:.2f}")
        axs[1].imshow(snaps_t[i], cmap="gray", vmin=-1, vmax=1)
        axs[1].set_title(f"Tau-leap t={times_t[i]:.2f}")
        plt.tight_layout()
        fname = os.path.join(output_dir, f"{stem}_snap_{i}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved snapshot {i} to {fname}")

    avg_gillespie_time = total_gillespie_time / n_runs
    avg_tauleap_time = total_tauleap_time / n_runs
    print(f"\nTiming results (N={n_runs} runs):")
    print(
        f"Average Gillespie time: {avg_gillespie_time:.2f}s (total: {total_gillespie_time:.2f}s)"
    )
    print(
        f"Average Tau-leaping time: {avg_tauleap_time:.2f}s (total: {total_tauleap_time:.2f}s)"
    )


if __name__ == "__main__":

    L = 128
    ell = 8.0
    sigma = 1.0
    tau = 1.0
    m0 = 0.2
    seed = 0
    beta = 1.0
    h = 0.0
    epsilon = 0.02
    t_end = 10.0
    snapshot_dt = 0.1
    n_runs = 20  # Number of independent runs with same initial condition

    model = Glauber2DIsingKacGPU(L=L)

    # Generate initial condition once
    m_field = grf_gaussian_spectrum_m_field(
        L=L, ell=ell, sigma=sigma, m0=m0, tau=tau, seed=seed, hard_cut=False
    )
    spin_init = sample_spins_from_m_field(m_field, seed=seed).astype(np.int8)

    output_dir = "./results/ising_tau_vs_gillespie"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    stem = f"ising_L{L}_ell{ell}_sigma{sigma}_tau{tau}_m0{m0}_beta{beta}_eps{epsilon}"

    print(
        f"Starting comparison with {n_runs} independent runs using the same initial condition"
    )
    test_compare_methods(
        model,
        spin_init,
        beta=beta,
        epsilon=epsilon,
        h=h,
        t_end=t_end,
        snapshot_dt=snapshot_dt,
        n_runs=n_runs,
        stem=stem,
        output_dir=output_dir,
    )
