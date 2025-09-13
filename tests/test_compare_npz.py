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
import glob


def resolve_npz_from_stem(base_dir: Path, stem: str) -> Path:
    """
    约定结构：
      base_dir / <stem> / round0 / <stem>_round0.npz
    """
    p = (base_dir / stem / "round0" / f"{stem}_round0.npz").resolve()
    if not p.exists():
        raise FileNotFoundError(f"[stem] Cannot find npz: {p}")
    return p


def load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    arr = np.load(path, allow_pickle=True)
    # 兼容你的保存字段命名
    times = arr["times"]
    Ms = arr["Ms"]
    Es = arr["Es"]
    spins_meso = arr["spins_meso"]  # (T_steps, H, W)
    return dict(times=times, Ms=Ms, Es=Es, spins_meso=spins_meso)


def l2(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    n = min(len(a), len(b)) if a.ndim == 1 else min(a.shape[0], b.shape[0])
    return float(np.linalg.norm(a[:n] - b[:n]))


def series_stats(x):
    x = np.asarray(x)
    return dict(
        mean=float(np.mean(x)),
        std=float(np.std(x)),
        min=float(np.min(x)),
        max=float(np.max(x)),
    )


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
        description="Compare Ms, Es, and snap_block_avg across selectable stems/npz files.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="data/ct_glauber",
        help="Root directory that contains <stem>/round0/<stem>_round0.npz (default: data)",
    )
    # 选择对比对象的三种方式（可混合；顺序：npz > stems > glob）
    parser.add_argument(
        "--npz",
        type=str,
        nargs="*",
        default=[],
        help="Direct paths to .npz files (highest priority).",
    )
    parser.add_argument(
        "--stems",
        type=str,
        nargs="*",
        default=[],
        help="List of stem names under base_dir.",
    )
    parser.add_argument(
        "--glob",
        dest="pattern",
        type=str,
        default=None,
        help="Glob pattern under base_dir to auto-collect stems, e.g. 'ct_glauber_L128_*_seed0'.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=[],
        help="Optional labels for legend; must match the number/order of resolved inputs.",
    )
    parser.add_argument(
        "--sample_k",
        type=int,
        default=6,
        help="Number of time points to sample for snap_block_avg per setting (default: 6)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for sampling (default: 0)"
    )
    parser.add_argument(
        "--save", action="store_true", help="Save figures instead of showing them"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results_compare",
        help="Directory to save figures",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    # 1) 先收集 npz 文件（最高优先级）
    resolved_paths = []
    labels = []

    for p in args.npz:
        rp = Path(p).resolve()
        if not rp.exists():
            raise FileNotFoundError(f"[npz] Not found: {rp}")
        resolved_paths.append(rp)
        labels.append(rp.parent.parent.name if rp.parent.name == "round0" else rp.stem)

    # 2) 其次根据 stems 去拼路径
    for stem in args.stems:
        rp = resolve_npz_from_stem(base_dir, stem)
        resolved_paths.append(rp)
        labels.append(stem)

    # 3) 最后用 glob 自动收集 stems
    if args.pattern:
        # 匹配 base_dir 下的目录名
        pattern = str((base_dir / args.pattern))
        # 只取目录名，过滤不是目录的匹配
        stem_dirs = [Path(p) for p in glob.glob(pattern) if Path(p).is_dir()]
        # 去重：不要重复添加已在列表中的 stem
        existing_dirs = set(
            p.parent.parent for p in resolved_paths if p.parent.name == "round0"
        )
        for d in sorted(stem_dirs):
            if d in existing_dirs:
                continue
            stem = d.name
            try:
                rp = resolve_npz_from_stem(base_dir, stem)
                resolved_paths.append(rp)
                labels.append(stem)
            except FileNotFoundError:
                # 有些目录可能没有 round0/npz，跳过
                pass

    if not resolved_paths:
        print("[ERROR] No inputs. Provide at least one of --npz, --stems, or --glob.")
        sys.exit(1)

    # 如果用户给了 --labels，则覆盖默认 labels
    if args.labels:
        if len(args.labels) != len(resolved_paths):
            raise ValueError(
                f"--labels count ({len(args.labels)}) must match inputs ({len(resolved_paths)})."
            )
        labels = args.labels

    print("[INFO] Will compare the following runs/files:")
    for lab, p in zip(labels, resolved_paths):
        print(f"  - {lab}: {p}")

    # 载入数据
    data = {}  # label -> dict
    for lab, p in zip(labels, resolved_paths):
        d = load_npz(p)
        data[lab] = d

    # ------------- 数值对比（打印）-------------
    print("\n[INFO] Pairwise L2 differences on aligned prefixes:")
    for lab1, lab2 in itertools.combinations(labels, 2):
        M1, M2 = align_prefix(data[lab1]["Ms"], data[lab2]["Ms"])
        E1, E2 = align_prefix(data[lab1]["Es"], data[lab2]["Es"])
        S1, S2 = data[lab1]["spins_meso"], data[lab2]["spins_meso"]
        tmin = min(S1.shape[0], S2.shape[0])
        S1a = S1[:tmin]
        S2a = S2[:tmin]
        dS = float(np.linalg.norm(S1a - S2a))
        dM = float(np.linalg.norm(M1 - M2))
        dE = float(np.linalg.norm(E1 - E2))
        print(f"  {lab1} vs {lab2}: Ms={dM:.4e}, Es={dE:.4e}, snap_block_avg={dS:.4e}")

    # 简单统计
    print("\n[INFO] Basic stats:")
    for lab in labels:
        Ms = data[lab]["Ms"]
        Es = data[lab]["Es"]
        sM = series_stats(Ms)
        sE = series_stats(Es)
        summary = textwrap.indent(
            textwrap.dedent(
                f"""\
                Ms: mean={sM['mean']:.6f}, std={sM['std']:.6f}, min={sM['min']:.6f}, max={sM['max']:.6f}
                Es: mean={sE['mean']:.6f}, std={sE['std']:.6f}, min={sE['min']:.6f}, max={sE['max']:.6f}
                """
            ),
            prefix="  ",
        )
        print(f"  {lab}:\n{summary}", end="")

    # ------------- 可视化：Ms / Es 时间序列 -------------
    # 统一对齐时间轴到共同长度（避免长度不同导致错位）
    min_len = min(len(data[lab]["Ms"]) for lab in labels)
    t_axes = {}
    for lab in labels:
        t = data[lab]["times"]
        if len(t) >= min_len:
            t_axes[lab] = t[:min_len]
        else:
            t_axes[lab] = np.arange(min_len)

    plt.figure(figsize=(10, 5))
    for lab in labels:
        Ms = np.asarray(data[lab]["Ms"])[:min_len]
        plt.plot(t_axes[lab], Ms, label=lab)
    plt.xlabel("time")
    plt.ylabel("Ms")
    plt.title("Magnetization (Ms) vs time")
    plt.legend()
    plt.tight_layout()
    if args.save:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        plt.savefig(out / "Ms_timeseries.png", dpi=150)
        plt.close()
    else:
        plt.show()

    plt.figure(figsize=(10, 5))
    for lab in labels:
        Es = np.asarray(data[lab]["Es"])[:min_len]
        plt.plot(t_axes[lab], Es, label=lab)
    plt.xlabel("time")
    plt.ylabel("Es")
    plt.title("Energy (Es) vs time")
    plt.legend()
    plt.tight_layout()
    if args.save:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        plt.savefig(out / "Es_timeseries.png", dpi=150)
        plt.close()
    else:
        plt.show()

    # ------------- 可视化：spins_meso 随机抽样热力图 -------------
    sample_k = args.sample_k
    seed = args.seed

    for lab in labels:
        S = data[lab]["spins_meso"]  # (T_steps, H, W)
        if S.ndim != 3:
            print(
                f"[WARN] {lab}.spins_meso has shape {S.shape}, expected 3D (T,H,W). Skipping heatmaps."
            )
            continue
        Tsteps = S.shape[0]
        idxs = pick_indices(Tsteps, sample_k, seed=seed)

        n = len(idxs)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        fig_w = 3 * cols
        fig_h = 3 * rows
        fig = plt.figure(figsize=(fig_w, fig_h))
        for i, t_idx in enumerate(idxs, start=1):
            ax = fig.add_subplot(rows, cols, i)
            im = ax.imshow(
                S[t_idx], origin="lower", aspect="equal", cmap="bwr", vmin=-1, vmax=1
            )
            ax.set_title(f"{lab} | t_idx={t_idx}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{lab} - snap_block_avg samples (k={n})")
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if args.save:
            out = Path(args.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"{lab}_snap_block_avg_samples.png", dpi=150)
            plt.close(fig)
        else:
            plt.show()

    print("\n[DONE] Comparison complete.")


if __name__ == "__main__":
    main()
