#!/usr/bin/env python3
"""
Evaluate the loss using the exact Gaussian kernel (from epsilon) without training.

Data: MultiRound (R rounds) with time truncation to MAX_TIME_POINTS → shapes:
  - times: (T,)
  - m_all: (R, T, H, W)
  - dmdt_all: (R, T, H, W)

Model: Fixed kernel KxK derived from epsilon and M; use LocalKernelPointEvalPeriodic
with ForceTanhNet; run one epoch over the dataset (R batches), compute avg MSE loss.
"""

import os
import sys
import json
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn as nn

# Ensure we can import from src/
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))

from train import MultiRoundIsingDataset, prepare_training_data_multi  # noqa: E402
from networks import (
    LocalKernelPointEvalPeriodic,
    ForceTanhNet,
)  # noqa: E402
import matplotlib.pyplot as plt


def _compute_reference_kernel_KxK(meta: dict, data_path_str: str, K: int) -> np.ndarray:
    # Parse epsilon from path first; fallback to metadata keys
    eps = None
    m_eps = re.search(r"epsilon([0-9]*\.?[0-9]+)", data_path_str)
    if m_eps:
        eps = float(m_eps.group(1))
    if eps is None:
        eps = meta.get("R", None) or meta.get("epsilon", None)
    if eps is None:
        raise ValueError("Cannot determine epsilon from data path or metadata")

    M = int(meta["M"])  # coarse grid size (sets spatial step)
    dx = 1.0 / M

    assert K % 2 == 1, "KERNEL_SIZE should be odd"
    h = K // 2
    # Build a centered grid of coordinates in physical space around (0,0)
    di = np.arange(-h, h + 1, dtype=np.float32)
    dj = np.arange(-h, h + 1, dtype=np.float32)
    DI, DJ = np.meshgrid(di, dj, indexing="ij")
    RX = DI * dx
    RY = DJ * dx
    R2 = RX * RX + RY * RY

    if eps > 0:
        Kref = np.exp(-R2 / (2.0 * eps * eps)) / (2.0 * np.pi * eps * eps)
    else:
        Kref = np.zeros((K, K), dtype=np.float32)
        Kref[h, h] = 1.0

    # Normalize over the KxK domain for a fair comparison to learned kernel
    s = Kref.sum()
    if s != 0:
        Kref = Kref / s
    return Kref


class FixedKernelModule(nn.Module):
    """nn.Module that returns a fixed (1,1,K,K) kernel tensor."""

    def __init__(self, kernel_2d: np.ndarray):
        super().__init__()
        assert kernel_2d.ndim == 2 and kernel_2d.shape[0] == kernel_2d.shape[1]
        K = kernel_2d.shape[0]
        self.K = K
        k = torch.from_numpy(kernel_2d.astype(np.float32)).view(1, 1, K, K)
        self.register_buffer("kernel", k)

    def forward(self, device=None, dtype=None) -> torch.Tensor:
        k = self.kernel
        if device is not None:
            k = k.to(device)
        if dtype is not None:
            k = k.to(dtype)
        return k


class ExactKernelNetwork(nn.Module):
    """
    Network that uses a fixed Gaussian kernel and ForceTanhNet to predict dm/dt
    given current magnetization field m at sampled coords.
    """

    def __init__(self, fixed_kernel: np.ndarray, T: float = 1.0, h: float = 0.0):
        super().__init__()
        self.kernel = FixedKernelModule(fixed_kernel)
        self.kernel_eval = LocalKernelPointEvalPeriodic(kernel=self.kernel)
        self.force_net_type = "Tanh"
        self.force_net = ForceTanhNet(T=T, h=h)  # keep defaults

    def forward(self, m: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # Step 1: kernel evaluation
        kernel_output = self.kernel_eval(m, coords)  # (B,1)

        # Step 2: extract local m values at coords
        if m.dim() == 4:
            m = m.squeeze(1)
        B, H, W = m.shape
        coord_i = coords[:, 1].long().clamp(0, H - 1)
        coord_j = coords[:, 0].long().clamp(0, W - 1)
        m_values = []
        for b in range(B):
            m_values.append(m[b, coord_i[b], coord_j[b]])
        m_local = torch.stack(m_values).unsqueeze(1)  # (B,1)

        # Step 3: ForceTanhNet (h=0)
        dmdt = self.force_net(kernel_output, m_local)
        return dmdt


def main():
    torch.manual_seed(42)

    # Data path parameters (keep consistent with test_train.py)
    data_dir = "data/ct_glauber"
    h = 1
    T = 4
    epsilon = 0.0625
    L_scale = 1
    ell = 32 * L_scale
    L = 1024 * L_scale
    block = 8 * L_scale
    t_end = 20.0

    data_path = os.path.join(
        data_dir,
        f"ct_glauber_L{L}_ell{ell}_sigma1.2_tau1_m00.1_T{T}_J1_h{h}_tend{t_end}_dt0.01_block{block}_kernelgaussian_epsilon{epsilon}_seed0",
        "round0",
        f"ct_glauber_L{L}_ell{ell}_sigma1.2_tau1_m00.1_T{T}_J1_h{h}_tend{t_end}_dt0.01_block{block}_kernelgaussian_epsilon{epsilon}_seed0_round0.npz",
    )
    data_path = Path(data_path)
    assert data_path.exists(), f"Data file not found: {data_path}"

    # Time truncation
    max_time_points = int(os.environ.get("MAX_TIME_POINTS", 500))

    # Dataset (R,T,H,W) and dataloader
    dataset = MultiRoundIsingDataset(
        round0_file_path=str(data_path), n_rounds=20, max_time_points=max_time_points
    )
    print(
        f"MultiRound dataset: R={dataset.R}, T={dataset.T}, H={dataset.H}, W={dataset.W}"
    )
    dataloader = prepare_training_data_multi(dataset, train_batch_size=64)

    # Metadata for epsilon -> exact kernel
    meta_file = str(data_path).replace(".npz", ".json")
    with open(meta_file, "r") as f:
        metadata = json.load(f)

    KERNEL_SIZE = 21
    ref_kernel = _compute_reference_kernel_KxK(metadata, str(data_path), KERNEL_SIZE)
    print(
        f"Exact kernel sum={ref_kernel.sum():.6f}, min={ref_kernel.min():.6e}, max={ref_kernel.max():.6e}"
    )

    # Save exact kernel figure to results with kernel size in title (loss appended later)
    out_dir = ROOT_DIR / f"results/L{L}_h{h}_T{T}_eps{epsilon}/exact_kernel"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"exact_kernel_K{KERNEL_SIZE}_Len{max_time_points}.png"

    # Network with fixed kernel
    network = ExactKernelNetwork(ref_kernel, T=T, h=h)

    # Evaluate one epoch over the dataset (no training), compute avg loss
    loss_fn = nn.MSELoss()
    loss_list = []
    with torch.no_grad():
        for batch in dataloader:
            m_fields = batch["m_fields"]  # (B,H,W)
            coords = batch["coords"]  # (B,2)
            dmdt_values = batch["dmdt_values"]  # (B,1)

            preds = network(m_fields, coords)
            loss = loss_fn(preds, dmdt_values)
            loss_list.append(float(loss.item()))

    mean_loss = float(np.mean(loss_list)) if loss_list else float("nan")
    print(f"Exact-kernel loss (avg over 1 epoch): {mean_loss:.8f}")

    # Now write the exact-kernel figure with loss in the title
    try:
        plt.figure(figsize=(4, 4))
        im = plt.imshow(ref_kernel, cmap="viridis")
        plt.colorbar(im, fraction=0.046)
        plt.title(
            f"Exact Gaussian Kernel (K={KERNEL_SIZE})\nLoss(avg 1 epoch)={mean_loss:.6f}\nSelected Time Points={max_time_points}\n"
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved exact kernel figure with loss to: {fig_path}")
    except Exception as e:
        print(f"⚠️ Failed to save exact kernel figure with loss: {e}")


if __name__ == "__main__":
    main()
