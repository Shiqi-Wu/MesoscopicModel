# main_ct_glauber.py
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

from glauber2d import Glauber2DIsingCT
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
                               snaps: np.ndarray,
                               snaps_block_avg: np.ndarray,
                               Ms: np.ndarray,
                               Es: np.ndarray,
                               meta: dict, block: int):
    out_npz = out_root / f"{stem}.npz"
    out_json = out_root / f"{stem}.json"

    # npz 里存 times, spins, Ms, Es
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


def main():
    parser = argparse.ArgumentParser(description="Continuous-time Glauber simulation")
    # Lattice & init
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--ell", type=float, default=32.0)
    parser.add_argument("--sigma", type=float, default=1.2)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--m0", type=float, default=0.1)
    parser.add_argument("--k_cut", type=float, default=None)
    parser.add_argument("--butter_n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hard_cut", action="store_true", default=False)
    parser.add_argument("--kernel", type=str, default="nearest")
    parser.add_argument("--R", type=float, default=0.0)  # for Kac

    # Dynamics
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.0)
    parser.add_argument("--T_frac", type=float, default=0.7)
    parser.add_argument("--t_end", type=float, default=50.0)
    parser.add_argument("--snapshot_dt", type=float, default=0.1)

    # Control
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--block", type=int, default=16)

    # Output
    parser.add_argument("--outdir", type=str, default="result/spectral_ct_glauber")

    args = parser.parse_args()

    L = args.size
    J = args.J
    h = args.h
    Tc = critical_temp_2d(J)
    T = args.T_frac * Tc
    beta = 1.0 / T

    out_root = Path(args.outdir)
    ensure_dir(out_root)
    stem_base = (f"ct_glauber_L{L}_ell{args.ell:g}_sigma{args.sigma:g}_tau{args.tau:g}_m0{args.m0:g}"
                 f"_Tfrac{args.T_frac:g}_J{J:g}_h{h:g}_tend{args.t_end}_dt{args.snapshot_dt}_block{args.block}_kernel{args.kernel}_R{args.R:g}_seed{args.seed}")
    run_dir = out_root / stem_base
    ensure_dir(run_dir)

    print(f"[info] L={L}, T={T:.4f} (Tc={Tc:.4f}, T/Tc={T/Tc:.3f}), "
          f"ell={args.ell}, sigma={args.sigma}, tau={args.tau}, m0={args.m0}, block={args.block}")

    # 1) Spectral init
    m_field = grf_gaussian_spectrum_m_field(
        L=L, ell=args.ell, sigma=args.sigma, m0=args.m0,
        tau=args.tau, seed=args.seed,
        k_cut=args.k_cut, butter_n=args.butter_n, hard_cut=False
    )
    spins0 = sample_spins_from_m_field(m_field, seed=args.seed).astype(np.int8)
    coarse0 = block_average(spins0, block=args.block)

    fig_path = run_dir / "initial_and_coarse.png"
    plot_initial_and_coarse(spins0, coarse0, timestr="spectral init", save_path=str(fig_path))

    sim = Glauber2DIsingCT(size=L)

    for r in range(args.rounds):
        print(f"[round {r}] starting...")
        if args.kernel == "nearest":
            times, Ms, Es, snaps = sim.simulate(
                spin_init=spins0,
                beta=beta, J=J, h=h,
                t_end=args.t_end, snapshot_dt=args.snapshot_dt,
                return_snapshots=True
            )
        elif args.kernel == "gaussian":
            times, Ms, Es, snaps = sim.simulate_kac(
                spin_init=spins0,
                beta=beta, J0=J, h=h,
                R=args.R, kernel="gaussian", sigma=args.sigma,
                t_end=args.t_end, snapshot_dt=args.snapshot_dt,
                return_snapshots=True
            )
        elif args.kernel == "uniform_disk":
            times, Ms, Es, snaps = sim.simulate_kac(
                spin_init=spins0,
                beta=beta, J0=J, h=h,
                R=args.R, kernel="uniform_disk", sigma=args.sigma,
                t_end=args.t_end, snapshot_dt=args.snapshot_dt,
                return_snapshots=True
            )
        else:
            raise ValueError(f"Unknown kernel type: {args.kernel}")

        # 块平均
        tiles = int(np.ceil(L / args.block))
        snaps_block_avg = np.zeros((len(times), tiles, tiles), dtype=np.float64)
        for k in range(len(times)):
            for i in range(tiles):
                for j in range(tiles):
                    x0, x1 = i*args.block, min((i+1)*args.block, L)
                    y0, y1 = j*args.block, min((j+1)*args.block, L)
                    snaps_block_avg[k, i, j] = np.mean(snaps[k, x0:x1, y0:y1])

        meta = dict(
            L=L, J=J, h=h,
            T=float(T), Tc=float(Tc), T_frac=float(args.T_frac),
            ell=float(args.ell), sigma=float(args.sigma), tau=float(args.tau), m0=float(args.m0),
            seed=int(args.seed), block=int(args.block),
            t_end=float(args.t_end), snapshot_dt=float(args.snapshot_dt),
            init="spectral", dynamics="continuous_glauber", 
            kernel=args.kernel, R=float(getattr(args, "R", 0.0)),
            round=int(r),
        )

        round_run_dir = run_dir / f"round{r}"
        ensure_dir(round_run_dir)
        stem = f"{stem_base}_round{r}"
        export_npz_and_json_bundle(round_run_dir, stem,
                                   times, snaps, snaps_block_avg, Ms, Es,
                                   meta, args.block)


    print(f"[done] All outputs under {run_dir}")

if __name__ == "__main__":
    main()
