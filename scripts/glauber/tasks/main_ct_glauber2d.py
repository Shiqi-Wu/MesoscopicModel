# main_ct_glauber_gpu.py
import os
import sys
sys.path.append("src")
from pathlib import Path
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from numba import config
config.NUMBA_NUM_THREADS = 1

# Updated import: use the new unified class
from glauber2d_gpu import Glauber2DIsingCT
from init_grf_ifft import (
    grf_gaussian_spectrum_m_field,
    sample_spins_from_m_field,
    block_average,
    plot_initial_and_coarse,
)

def critical_temp_2d(J: float = 1.0) -> float:
    return 2.0 * J / np.log(1.0 + np.sqrt(2.0))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def export_npz_and_json_bundle(out_root: Path, stem: str,
                               times: np.ndarray,
                               snaps: np.ndarray = None,
                               snaps_block_avg: np.ndarray = None,
                               Ms: np.ndarray = None,
                               Es: np.ndarray = None,
                               meta: dict = None,
                               block: int = 8,
                               remote_dir: str = None):
    out_npz = out_root / f"{stem}.npz"
    out_json = out_root / f"{stem}.json"

    np.savez_compressed(out_npz,
                        times=times,
                        spins=snaps,
                        spins_meso=snaps_block_avg,
                        Ms=Ms,
                        Es=Es)
    print(f"[save] {out_npz}")

    meta2 = dict(meta)
    meta2["B"] = int(block)
    M = int(np.ceil(meta2["L"] / block))
    meta2["M"] = M
    if "L" not in meta2 or meta2["L"] is None:
        meta2["L"] = int(M * block)
    meta2["data_level"] = "micro"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta2, f, ensure_ascii=False, indent=2)
    print(f"[save] {out_json}")

    if remote_dir is not None:
        os.system(f"cp {out_npz} {remote_dir}/")
        os.system(f"cp {out_json} {remote_dir}/")
        print(f"[remote copy] to {remote_dir}")

def main():
    parser = argparse.ArgumentParser(description="Continuous-time Glauber simulation (NN and Kac)")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--ell", type=float, default=32.0)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--m0", type=float, default=0.1)
    parser.add_argument("--k_cut", type=float, default=None)
    parser.add_argument("--butter_n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hard_cut", action="store_true", default=False)

    # kernel: "nearest" or "gaussian" (gaussian uses epsilon)
    parser.add_argument("--kernel", type=str, default="nearest")
    parser.add_argument("--epsilon", type=float, default=0.1)

    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--method", type=str, default="gillespie")  # gillespie / tau-leaping
    parser.add_argument("--eps_tau", type=float, default=0.01)

    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.0)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--t_end", type=float, default=50.0)
    parser.add_argument("--snapshot_dt", type=float, default=0.1)

    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--block", type=int, default=16)

    parser.add_argument("--outdir", type=str, default="result/spectral_ct_glauber_gpu")
    parser.add_argument("--remote_dir", type=str, default=None)

    args = parser.parse_args()

    L = args.size
    J = args.J           # Interpreted as J for nearest or J0 for Kac
    h = args.h
    Tc = critical_temp_2d(J)
    T = args.T
    beta = 1.0 / T

    out_root = Path(args.outdir)
    ensure_dir(out_root)
    stem_base = (f"ct_glauber_L{L}_ell{args.ell:g}_sigma{args.sigma:g}_tau{args.tau:g}_m0{args.m0:g}"
                 f"_T{args.T:g}_J{J:g}_h{h:g}_tend{args.t_end}_dt{args.snapshot_dt}_block{args.block}_"
                 f"kernel{args.kernel}_epsilon{args.epsilon:g}_seed{args.seed}")
    run_dir = out_root / stem_base
    ensure_dir(run_dir)

    if args.remote_dir is not None:
        remote_dir = Path(args.remote_dir)
        remote_run_dir = remote_dir / stem_base
        ensure_dir(remote_run_dir)
        print(f"[info] Remote directory specified: {remote_run_dir}")
    else:
        remote_run_dir = None

    print(f"[info] L={L}, T={T:.4f} (Tc={Tc:.4f}, T/Tc={T/Tc:.3f}), "
          f"ell={args.ell}, sigma={args.sigma}, tau={args.tau}, m0={args.m0}, block={args.block}, "
          f"kernel={args.kernel}, epsilon={args.epsilon}, use_gpu={args.use_gpu}")

    # Spectral GRF initialization (unchanged)
    m_field = grf_gaussian_spectrum_m_field(
        L=L, ell=args.ell, sigma=args.sigma, m0=args.m0,
        tau=args.tau, seed=args.seed,
        k_cut=args.k_cut, butter_n=args.butter_n, hard_cut=False
    )
    spins0 = sample_spins_from_m_field(m_field, seed=args.seed).astype(np.int8)
    coarse0 = block_average(spins0, block=args.block)

    fig_path = run_dir / "initial_and_coarse.png"
    plot_initial_and_coarse(spins0, coarse0, timestr="spectral init", save_path=str(fig_path))

    # Construct the new simulator (note: __init__ expects L, not size=)
    sim = Glauber2DIsingCT(L=L, B=args.block, use_gpu=args.use_gpu)

    for r in range(args.rounds):
        print(f"[round {r}] starting...")

        # Map CLI options into the new unified APIs
        if args.method == "gillespie":
            # Use simulate_kac for both nearest and gaussian; select via kernel_type
            times, Ms, Es, snaps = sim.simulate(
                spin_init=spins0,
                beta=beta,
                J0=J,                 # J for nearest; J0 for Kac
                h=h,
                epsilon=args.epsilon, # used only for gaussian
                kernel_type=args.kernel,  # "nearest" or "gaussian"
                t_end=args.t_end,
                snapshot_dt=args.snapshot_dt,
                return_mode="coarse",
                verbose_every=1000
            )
        elif args.method == "tau-leaping":
            times, Ms, Es, snaps = sim.simulate_tauleap(
                spin_init=spins0,
                beta=beta,
                J0=J,
                h=h,
                epsilon=args.epsilon,
                kernel_type=args.kernel,
                t_end=args.t_end,
                snapshot_dt=args.snapshot_dt,
                eps_tau=args.eps_tau,
                return_mode="coarse",
                verbose_every=1000
            )
        else:
            raise ValueError(f"Unknown method: {args.method}")

        meta = dict(
            L=L, J=J, h=h,
            Tc=float(Tc), T=float(args.T),
            ell=float(args.ell), sigma=float(args.sigma), tau=float(args.tau), m0=float(args.m0),
            seed=int(args.seed), block=int(args.block),
            t_end=float(args.t_end), snapshot_dt=float(args.snapshot_dt),
            init="spectral", dynamics="continuous_glauber",
            kernel=args.kernel, epsilon=float(getattr(args, "epsilon", 0.0)),
            round=int(r),
            use_gpu=args.use_gpu
        )

        round_run_dir = run_dir / f"round{r}"
        ensure_dir(round_run_dir)

        if remote_run_dir is not None:
            remote_round_run_dir = remote_run_dir / f"round{r}"
            ensure_dir(remote_round_run_dir)
        else:
            remote_round_run_dir = None
        stem = f"{stem_base}_round{r}"

        export_npz_and_json_bundle(round_run_dir, stem,
                                   times=times, snaps=None, snaps_block_avg=snaps,
                                   Ms=Ms, Es=Es,
                                   meta=meta, block=args.block,
                                   remote_dir=remote_round_run_dir)

    print(f"[done] All outputs under {run_dir}")


if __name__ == "__main__":
    main()
