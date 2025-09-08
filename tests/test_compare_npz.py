# test_compare_npz.py
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import itertools
import textwrap
import sys
import math
import random

def build_path(base_dir: Path, T: int, h: int):
    stem = f"ct_glauber_L128_ell32_sigma1.2_tau1_m00.1_T{T}_J1_h{h}_tend2.0_dt0.01_block8_kernelgaussian_epsilon0.015625_seed0"
    return (base_dir / stem / "round0" / f"{stem}_round0.npz").resolve()

def load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    arr = np.load(path, allow_pickle=True)
    # 兼容你的保存字段命名
    times = arr["times"]
    Ms = arr["Ms"]
    Es = arr["Es"]
    spins_meso = arr["spins_meso"]  # (T_steps, M, M) 或 (T_steps, H, W)
    return dict(times=times, Ms=Ms, Es=Es, spins_meso=spins_meso)

def l2(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    n = min(len(a), len(b)) if a.ndim == 1 else min(a.shape[0], b.shape[0])
    return float(np.linalg.norm(a[:n] - b[:n]))

def series_stats(x):
    x = np.asarray(x)
    return dict(mean=float(np.mean(x)), std=float(np.std(x)), min=float(np.min(x)), max=float(np.max(x)))

def align_prefix(a, b):
    """对齐时间序列到共同的前缀长度，返回切片后的 (a, b)"""
    n = min(len(a), len(b))
    return a[:n], b[:n]

def pick_indices(n, k, seed=0):
    """从 [0, n-1] 随机挑选 k 个时间下标（无重复），并包含首尾"""
    random.seed(seed)
    idxs = set([0, n - 1]) if n >= 2 else set([0])
    while len(idxs) < min(k, n):
        idxs.add(random.randrange(0, n))
    return sorted(idxs)

def main():
    parser = argparse.ArgumentParser(
        description="Compare Ms, Es, and snap_block_avg across (T,h) settings.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--base_dir", type=str, default="data",
                        help="Root directory containing the 4 NPZ folders (default: data)")
    parser.add_argument("--sample_k", type=int, default=6,
                        help="Number of time points to sample for snap_block_avg per setting (default: 6)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling (default: 0)")
    parser.add_argument("--save", action="store_true", help="Save figures instead of showing them")
    parser.add_argument("--out_dir", type=str, default="results_compare", help="Directory to save figures")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    pairs = [("T1_h0", 1, 0), ("T1_h1", 1, 1), ("T4_h0", 4, 0), ("T4_h1", 4, 1)]
    data = {}

    print("[INFO] Loading files...")
    for name, T, h in pairs:
        p = build_path(base_dir, T, h)
        d = load_npz(p)
        data[name] = d
        print(f"  - {name}: {p}")

    # ------------- 数值对比（打印）-------------
    print("\n[INFO] Pairwise L2 differences on aligned prefixes:")
    for (n1, _, _), (n2, _, _) in itertools.combinations(pairs, 2):
        M1, M2 = align_prefix(data[n1]["Ms"], data[n2]["Ms"])
        E1, E2 = align_prefix(data[n1]["Es"], data[n2]["Es"])
        S1, S2 = data[n1]["spins_meso"], data[n2]["spins_meso"]
        tmin = min(S1.shape[0], S2.shape[0])
        S1a = S1[:tmin]
        S2a = S2[:tmin]
        # 对 snap_block_avg：展平后算 L2
        dS = float(np.linalg.norm(S1a - S2a))
        dM = float(np.linalg.norm(M1 - M2))
        dE = float(np.linalg.norm(E1 - E2))
        print(f"  {n1} vs {n2}: Ms={dM:.4e}, Es={dE:.4e}, snap_block_avg={dS:.4e}")

    # 简单统计
    print("\n[INFO] Basic stats:")
    for name, _, _ in pairs:
        Ms = data[name]["Ms"]
        Es = data[name]["Es"]
        sM = series_stats(Ms)
        sE = series_stats(Es)
        summary = textwrap.indent(
            textwrap.dedent(
                f"""\
                Ms: mean={sM['mean']:.6f}, std={sM['std']:.6f}, min={sM['min']:.6f}, max={sM['max']:.6f}
                Es: mean={sE['mean']:.6f}, std={sE['std']:.6f}, min={sE['min']:.6f}, max={sE['max']:.6f}
                """
            ), prefix="  "
        )
        print(f"  {name}:\n{summary}", end="")

    # ------------- 可视化：Ms / Es 时间序列 -------------
    # 统一对齐时间轴到共同长度（避免长度不同导致错位）
    min_len = min(len(data[n]["Ms"]) for n,_,_ in pairs)
    t_axes = {}
    for name, _, _ in pairs:
        t = data[name]["times"]
        if len(t) >= min_len:
            t_axes[name] = t[:min_len]
        else:
            # 若 times 缺失或长度异常，用索引代替
            t_axes[name] = np.arange(min_len)

    plt.figure(figsize=(10, 5))
    for name, _, _ in pairs:
        Ms = np.asarray(data[name]["Ms"])[:min_len]
        plt.plot(t_axes[name], Ms, label=name)
    plt.xlabel("time")
    plt.ylabel("Ms")
    plt.title("Magnetization (Ms) vs time")
    plt.legend()
    plt.tight_layout()
    if args.save:
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        plt.savefig(out / "Ms_timeseries.png", dpi=150)
        plt.close()
    else:
        plt.show()

    plt.figure(figsize=(10, 5))
    for name, _, _ in pairs:
        Es = np.asarray(data[name]["Es"])[:min_len]
        plt.plot(t_axes[name], Es, label=name)
    plt.xlabel("time")
    plt.ylabel("Es")
    plt.title("Energy (Es) vs time")
    plt.legend()
    plt.tight_layout()
    if args.save:
        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        plt.savefig(out / "Es_timeseries.png", dpi=150)
        plt.close()
    else:
        plt.show()

    # ------------- 可视化：snap_block_avg 随机抽样热力图 -------------
    # 每个设置抽 sample_k 个时间点（固定 seed），并画成网格图
    sample_k = args.sample_k
    seed = args.seed

    for name, _, _ in pairs:
        S = data[name]["spins_meso"]  # (T_steps, H, W)
        if S.ndim != 3:
            print(f"[WARN] {name}.spins_meso has shape {S.shape}, expected 3D (T,H,W). Skipping heatmaps.")
            continue
        Tsteps = S.shape[0]
        idxs = pick_indices(Tsteps, sample_k, seed=seed)

        # 布局：近似 sqrt 排列
        n = len(idxs)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        fig_w = 3 * cols
        fig_h = 3 * rows
        fig = plt.figure(figsize=(fig_w, fig_h))
        for i, t_idx in enumerate(idxs, start=1):
            ax = fig.add_subplot(rows, cols, i)
            im = ax.imshow(S[t_idx], origin="lower", aspect="equal", cmap="bwr", vmin=-1, vmax=1)
            ax.set_title(f"{name} | t_idx={t_idx}")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{name} - snap_block_avg samples (k={n})")
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if args.save:
            out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"{name}_snap_block_avg_samples.png", dpi=150)
            plt.close(fig)
        else:
            plt.show()

    print("\n[DONE] Comparison complete.")

if __name__ == "__main__":
    main()
