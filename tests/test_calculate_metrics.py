import json
import re
import numpy as np
from pathlib import Path
import os, sys
from typing import Optional
sys.path.append("src")
from utils import build_kernel_fft, compute_susceptibility_series, free_energy_series, ensure_dir

def augment_npz_with_metrics(npz_path: Path, json_path: Path,
                             beta: float, h_field: float, J0: float, kernel_k: np.ndarray,
                             remote_dir: Optional[str] = None):
    """
    Load simulation data, compute free energy & susceptibility, and save back into npz/json.
    """

    # --- Load data
    data = np.load(npz_path, allow_pickle=True)
    print(f"[load] loaded {npz_path}, keys: {list(data.keys())}")
    times = data["times"]
    traj = data["spins_meso"]
    print(f"[load] loaded {npz_path}, traj shape: {traj.shape}, times shape: {times.shape}")

    # --- Compute metrics
    print(f"[compute] computing metrics for {npz_path} ...")
    chi = compute_susceptibility_series(traj, beta)
    _, F = free_energy_series(traj, times, beta, h_field, J0, kernel_k)
    print(f"[compute] done computing metrics for {npz_path}")

    # --- Re-save npz (include new arrays)
    out_npz = npz_path
    tmp_path = npz_path.parent / ("tmp_" + npz_path.name)
    print(f"[DEBUG] tmp_path: {tmp_path}, out_npz: {out_npz}")

    np.savez_compressed(tmp_path,
                        times=times,
                        spins=None,
                        spins_meso=data.get("spins_meso"),
                        Ms=data.get("Ms"),
                        Es=data.get("Es"),
                        chi=chi,
                        free_energy=F)
    os.replace(tmp_path, out_npz)
    print(f"[update] wrote metrics into {out_npz}")

    # --- Update JSON metadata
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["has_metrics"] = True
    meta["metrics"] = {
        "chi": {"shape": list(chi.shape), "desc": "susceptibility χ(t)"},
        "free_energy": {"shape": list(F.shape), "desc": "coarse-grained free energy F[m(t)]"},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[update] wrote metrics info into {json_path}")

    if remote_dir is not None:
        os.system(f"cp {out_npz} {remote_dir}/")
        os.system(f"cp {json_path} {remote_dir}/")
        print(f"[remote copy] to {remote_dir}")



def main(data_path: str, n_rounds: int = 20,
         beta: float = 1.0, h_field: float = 0.0,
         J0: float = 1.0, eps: float = 0.03125,
         remote_dir: Optional[str] = None):
    """
    Loop over all rounds, compute susceptibility & free energy, and update npz/json.
    """
    # Parse prefix
    m = re.match(r"^(.*_seed.*)/round\d+/(.*_seed.*)_round\d+\.npz$", data_path)
    if not m:
        raise ValueError(f"data_path format not recognized: {data_path}")
    dir_prefix = m.group(1)
    file_prefix = m.group(2)

    # Build kernel for free energy
    # Guess M from JSON of round0
    json_path0 = Path(f"{dir_prefix}/round0/{file_prefix}_round0.json")
    with open(json_path0, "r", encoding="utf-8") as f:
        meta = json.load(f)
    L = int(meta["L"])
    block = int(meta.get("B", 1))
    M = L // block
    print(f"[info] L={L}, block={block}, M={M}, eps={eps}")
    kernel_k = build_kernel_fft(M, eps)

    # Process each round
    for r in range(n_rounds):
        if remote_dir is not None:
            remote_round_dir = Path(remote_dir) / f"round{r}"
            ensure_dir(remote_round_dir)
            print(f"[remote] ensure dir {remote_round_dir}")
        else:
            remote_round_dir = None
        npz_path = Path(f"{dir_prefix}/round{r}/{file_prefix}_round{r}.npz")
        json_path = Path(f"{dir_prefix}/round{r}/{file_prefix}_round{r}.json")
        if not npz_path.exists() or not json_path.exists():
            print(f"⚠️ Missing files for round {r}, skipping")
            continue
        augment_npz_with_metrics(npz_path, json_path, beta, h_field, J0, kernel_k, remote_round_dir)
        print(f"✅ Processed round {r}")

if __name__ == "__main__":
    base_dir = "data/ct_glauber"
    remote_dir = "/mnt/nfs/homes/shiqi_w/MesoscopicModel/data/shared/ct_glauber"
    # Data path
    h = 0
    T = 1
    # epsilon = 0.0001
    epsilon = 0.03125
    # epsilon = 0.015625
    L_scale = 2
    ell = 32 * L_scale
    L = 1024 * L_scale
    block = 8 * L_scale
    t_end = 20.0
    stem = f"ct_glauber_L{L}_ell{ell}_sigma1.2_tau1_m00.1_T{T}_J1_h{h}_tend{t_end}_dt0.01_block{block}_kernelgaussian_epsilon{epsilon}_seed0"
    data_path = os.path.join(
        base_dir,
        stem,
        f"round0",
        f"ct_glauber_L{L}_ell{ell}_sigma1.2_tau1_m00.1_T{T}_J1_h{h}_"
        f"tend{t_end}_dt0.01_block{block}_kernelgaussian_epsilon{epsilon}_seed0_round0.npz"
    )
    n_rounds = 20
    beta_val = 1.0 / T
    h_field = float(h)
    J0 = 1.0
    eps_val = float(epsilon)
    main(data_path, n_rounds, beta_val, h_field, J0, eps_val, os.path.join(remote_dir, stem))
    print("✅ All rounds processed.")