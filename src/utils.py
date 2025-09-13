import random

import numpy as np
import torch
import torch.nn as nn
import re
import os


def set_seed_everywhere(seed=42):
    """Set random seed for reproducibility across all libraries"""

    # 1. Set the seed for Python's random module
    random.seed(seed)

    # 2. Set the seed for NumPy
    np.random.seed(seed)

    # 3. Set the seed for PyTorch (CPU and GPU)
    torch.manual_seed(seed)

    # 4. If using CUDA, also set the seed for GPU operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 5. Ensure deterministic behavior for certain PyTorch operations (optional but useful)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def critical_temp_2d(J: float = 1.0) -> float:
    """Onsager Tc for 2D square-lattice NN Ising."""
    return 2.0 / np.log(1.0 + np.sqrt(2.0)) * J


def count_trainable_params(module: nn.Module) -> int:
    """Count total number of trainable parameters (requires_grad=True) in module"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_all_params(module: nn.Module) -> int:
    """Count total number of all parameters in module (including non-trainable buffers/frozen parameters)"""
    return sum(p.numel() for p in module.parameters())


def print_metadata_info(
    metadata: dict,
    title: str = "Metadata Information",
    save_to_file: bool = True,
    output_dir: str = "results",
) -> None:
    """
    Print useful information from metadata dictionary in a formatted way.

    Args:
        metadata: Dictionary containing simulation metadata
        title: Title for the information section
        save_to_file: Whether to save the output to a text file
        output_dir: Directory to save the output file
    """
    import os
    from datetime import datetime

    # Collect all output lines
    output_lines = []

    def add_line(text: str):
        output_lines.append(text)
        print(text)

    add_line("\n" + "=" * 60)
    add_line(f"📊 {title}")
    add_line("=" * 60)

    # Lattice parameters
    add_line("🏗️  Lattice Configuration:")
    L = metadata.get("L", "N/A")
    M = metadata.get("M", "N/A")
    B = metadata.get("B", "N/A")
    add_line(f"  L (lattice size): {L}")
    add_line(f"  M (coarse-grained size): {M}")
    add_line(f"  B (block size): {B}")
    if L != "N/A" and M != "N/A" and B != "N/A":
        add_line(f"  Coarse-graining ratio: {L}/{M} = {L/M:.1f}")
    if M != "N/A":
        add_line(f"  Spatial resolution δ: {1 / M:.6f}")

    # Physical parameters
    add_line("\n🌡️  Physical Parameters:")
    T = metadata.get("T", "N/A")
    Tc = metadata.get("Tc", "N/A")
    T_frac = metadata.get("T_frac", "N/A")
    J = metadata.get("J", "N/A")
    h = metadata.get("h", "N/A")

    add_line(f"  T (temperature): {T}")
    add_line(f"  Tc (critical temperature; 2J/ln(1+sqrt(2))): {Tc}")
    add_line(f"  T_frac (T/Tc): {T_frac}")
    if T != "N/A" and T != 0:
        beta = 1.0 / T
        add_line(f"  β (beta = 1/T): {beta:.6f}")
    add_line(f"  J (coupling strength): {J}")
    add_line(f"  h (external field): {h}")

    # Additional parameters
    add_line("\n⚙️  Additional Parameters:")
    sigma = metadata.get("sigma", "N/A")
    tau = metadata.get("tau", "N/A")
    m0 = metadata.get("m0", "N/A")
    ell = metadata.get("ell", "N/A")
    R = metadata.get("R", "N/A")

    add_line(f"  σ (sigma): {sigma}")
    add_line(f"  τ (tau): {tau}")
    add_line(f"  m0 (initial magnetization): {m0}")
    add_line(f"  ell (characteristic length): {ell}")
    add_line(f"  R (interaction radius): {R}")

    # Simulation parameters
    add_line("\n🔄 Simulation Parameters:")
    t_end = metadata.get("t_end", "N/A")
    dt = metadata.get("snapshot_dt", "N/A")
    init = metadata.get("init", "N/A")
    dynamics = metadata.get("dynamics", "N/A")
    kernel = metadata.get("kernel", "N/A")

    add_line(f"  t_end (simulation time): {t_end}")
    add_line(f"  dt (time step): {dt}")
    add_line(f"  init (initialization): {init}")
    add_line(f"  dynamics: {dynamics}")
    add_line(f"  kernel: {kernel}")

    # Random seeds
    add_line("\n🎲 Random Seeds:")
    seed = metadata.get("seed", "N/A")
    add_line(f"  seed: {seed}")

    # Data information
    add_line("\n📈 Data Information:")
    data_level = metadata.get("data_level", "N/A")
    round_num = metadata.get("round", "N/A")
    add_line(f"  data_level: {data_level}")
    add_line(f"  round: {round_num}")

    # Summary
    add_line("\n📋 Summary:")
    if T != "N/A" and Tc != "N/A":
        temp_ratio = T / Tc if Tc != 0 else "N/A"
        add_line(f"  Temperature regime: T/Tc = {temp_ratio:.3f}")
        if temp_ratio != "N/A":
            if temp_ratio > 1.5:
                regime = "High temperature (disordered)"
            elif temp_ratio < 0.8:
                regime = "Low temperature (ordered)"
            else:
                regime = "Near critical (transition region)"
            add_line(f"  Physical regime: {regime}")

    if L != "N/A" and M != "N/A":
        add_line(f"  System size: {L}×{L} → {M}×{M} (coarse-grained)")

    add_line("=" * 60)

    # Save to file if requested
    if save_to_file:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metadata_info_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)

        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))

        print(f"\n💾 Metadata information saved to: {filepath}")


def compute_susceptibility_series(
    field_trajectory: np.ndarray, beta: float
) -> np.ndarray:
    """
    Compute susceptibility χ(t) from a field trajectory m(t,x) using spatial fluctuations:

    χ(t) = β N (⟨m^2⟩_x − ⟨m⟩_x^2), where N = M*M and ⟨·⟩_x is spatial average.

    Args:
        field_trajectory: array of shape (T, M, M)
        beta: inverse temperature β = 1/T

    Returns:
        chi: array of shape (T,) with susceptibility over time
    """
    if field_trajectory.ndim != 3:
        raise ValueError("field_trajectory must have shape (T, M, M)")
    Tn, M, _ = field_trajectory.shape
    N = float(M * M)
    m_mean = field_trajectory.reshape(Tn, -1).mean(axis=1)
    m2_mean = (field_trajectory**2).reshape(Tn, -1).mean(axis=1)
    var = m2_mean - m_mean**2
    chi = beta * N * var
    return chi


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_all_rounds(data_path: str, n_rounds: int = 20):
    """
    Given a reference data_path (e.g. round0),
    load magnetization and energy from round0..n_rounds-1.
    Returns (times, mags_all, Es_all).
    """
    # Strip off the "_roundX.npz"
    m = re.match(r"^(.*)_round(\d+)\.npz$", data_path)
    if not m:
        raise ValueError(f"data_path format not recognized: {data_path}")
    prefix = m.group(1)

    mags_all = []
    Es_all = []
    times = None

    for r in range(n_rounds):
        npz_path = f"{prefix}_round{r}.npz"
        if not os.path.exists(npz_path):
            print(f"⚠️ Missing {npz_path}")
            continue
        data = np.load(npz_path, allow_pickle=True)
        # Macro field or spins_meso (depending on what you saved)
        m_fields = data.get("spins_meso", data.get("spins"))
        if m_fields is None:
            raise KeyError("No spins_meso or spins in npz file.")
        if times is None:
            times = data["times"]
        mags = [np.mean(mf) for mf in m_fields]
        mags_all.append(mags)
        if "Es" in data:
            Es_all.append(data["Es"])

    mags_all = np.array(mags_all) if mags_all else None
    Es_all = np.array(Es_all) if Es_all else None
    return times, mags_all, Es_all


def export_npz_and_json_bundle(
    out_root: Path,
    stem: str,
    times: np.ndarray,
    snaps: np.ndarray = None,
    snaps_block_avg: np.ndarray = None,
    Ms: np.ndarray = None,
    Es: np.ndarray = None,
    meta: dict = None,
    block: int = 8,
    remote_dir: str = None,
):
    out_npz = out_root / f"{stem}.npz"
    out_json = out_root / f"{stem}.json"

    np.savez_compressed(
        out_npz, times=times, spins=snaps, spins_meso=snaps_block_avg, Ms=Ms, Es=Es
    )
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
