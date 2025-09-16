#!/usr/bin/env python3
"""
Training harness using MultiRoundIsingDataset with TensorBoard logging, saved figures, and
kernel validation. Supports FREEZE option to only train kernel (freeze ForceNet).
"""

import os
import sys
import json
from pathlib import Path
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import imageio.v2 as imageio


# Ensure we can import from src/
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.append(str(ROOT_DIR / "src"))

from train import (  # noqa: E402
    MultiRoundIsingDataset,
    prepare_training_data_multi,
    NonlocalPDENetwork,
)
from plot import visualize_network  # noqa: E402


def _compute_reference_kernel_KxK(meta: dict, data_path_str: str, K: int) -> np.ndarray:
    eps = None
    m_eps = re.search(r"epsilon([0-9]*\.?[0-9]+)", data_path_str)
    if m_eps:
        eps = float(m_eps.group(1))
    if eps is None:
        eps = meta.get("R", None) or meta.get("epsilon", None)
    if eps is None:
        raise ValueError("Cannot determine epsilon from data path or metadata")

    M = int(meta["M"])  # coarse grid size
    dx = 1.0 / M

    assert K % 2 == 1, "KERNEL_SIZE should be odd"
    h = K // 2
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
    s = Kref.sum()
    if s != 0:
        Kref = Kref / s
    return Kref


def _tb_add_saved_image(writer: SummaryWriter, img_path: Path, tag: str, step: int):
    try:
        if not img_path.exists():
            print(f"  [TB] image not found: {img_path}")
            return
        img = imageio.imread(img_path)
        if img.ndim == 2:
            chw = np.expand_dims(img, 0)
        elif img.ndim == 3:
            chw = np.transpose(img, (2, 0, 1))
        else:
            print(f"  [TB] unsupported image ndim={img.ndim} for {img_path}")
            return
        writer.add_image(tag, chw, step)
    except Exception as e:
        print(f"  [TB] add_image failed for {img_path}: {e}")


def main():
    torch.manual_seed(42)

    print("\n=== Setting up data ===")
    data_dir = "data/ct_glauber"
    # Data path
    h = 1
    T = 4
    epsilon = 0.03125
    L_scale = 2
    ell = 32 * L_scale
    L = 1024 * L_scale
    block = 8 * L_scale
    t_end = 20.0
    data_path = os.path.join(
        data_dir,
        f"ct_glauber_L{L}_ell{ell}_sigma1.2_tau1_m00.1_T{T}_J1_h{h}_"
        f"tend{t_end}_dt0.01_block{block}_kernelgaussian_epsilon{epsilon}_seed0",
        "round0",
        f"ct_glauber_L{L}_ell{ell}_sigma1.2_tau1_m00.1_T{T}_J1_h{h}_"
        f"tend{t_end}_dt0.01_block{block}_kernelgaussian_epsilon{epsilon}_seed0_round0.npz",
    )
    data_path = Path(data_path)

    MAX_TIME_POINTS = 100
    FREEZE = True  # True: only train kernel; False: train kernel + ForceNet

    assert data_path.exists(), f"Data file not found: {data_path}"

    # Dataset and dataloader
    dataset = MultiRoundIsingDataset(
        round0_file_path=str(data_path), n_rounds=20, max_time_points=MAX_TIME_POINTS
    )
    print(
        f"MultiRound dataset: R={dataset.R}, T={dataset.T}, H={dataset.H}, W={dataset.W}"
    )
    dataloader = prepare_training_data_multi(dataset, train_batch_size=32)

    # Read metadata
    metadata_file = str(data_path).replace(".npz", ".json")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Model config
    KERNEL_SIZE = 15
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 100000
    FORCE_NET_TYPE = "Tanh"

    # Output dir includes L,h,T,eps and time usage
    time_steps_used = int(dataset.T)
    try:
        duration_used = float(dataset.times[time_steps_used - 1] - dataset.times[0])
    except Exception:
        duration_used = float("nan")

    OUTPUT_DIR = ROOT_DIR / (
        f"results/L{L}_h{h}_T{T}_eps{epsilon}/"
        f"nonlocal_pde_network_kernel{KERNEL_SIZE}_force{FORCE_NET_TYPE}_lr{LEARNING_RATE}_freeze{FREEZE}_"
        f"steps{time_steps_used}_dur{duration_used:.3f}"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    TB_DIR = OUTPUT_DIR / "tensorboard"
    writer = SummaryWriter(log_dir=str(TB_DIR))
    writer.add_text(
        "config",
        (
            f"kernel_size={KERNEL_SIZE}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}, "
            f"force={FORCE_NET_TYPE}, R={dataset.R}, T={dataset.T}, H={dataset.H}, W={dataset.W}, "
            f"steps_used={time_steps_used}, duration_used={duration_used:.6f}, FREEZE={FREEZE}"
        ),
        0,
    )

    print("\n=== Initializing model and dataloader ===")
    network = NonlocalPDENetwork(
        kernel_size=KERNEL_SIZE,
        hidden_sizes=[64, 64, 64],
        activation=nn.Tanh(),
        force_net_type=FORCE_NET_TYPE,
        T=T,
        h=h,
    )

    # FREEZE option: if True, only train kernel (freeze ForceNet)
    if FREEZE and hasattr(network, "force_net"):
        for p in network.force_net.parameters():
            p.requires_grad = False
        print("[FREEZE] Training kernel only; ForceNet is frozen.")
    else:
        print("[FREEZE] Training kernel + ForceNet.")

    optimizer = torch.optim.Adam(
        (p for p in network.parameters() if p.requires_grad), lr=LEARNING_RATE
    )

    # Reference kernel for validation
    ref_kernel = _compute_reference_kernel_KxK(metadata, str(data_path), KERNEL_SIZE)

    print("\n=== Training (with visualization, kernel validation & checkpoints) ===")
    VIZ_INTERVAL = int(os.environ.get("VIZ_INTERVAL", 1000))
    CKPT_INTERVAL = int(os.environ.get("CKPT_INTERVAL", 1000))

    for epoch in range(EPOCHS):
        loss_list = []
        for batch in dataloader:
            m_fields = batch["m_fields"]
            dmdt_values = batch["dmdt_values"]
            coords = batch["coords"]

            network.train()
            dmdt_pred = network(m_fields, coords)
            loss = nn.MSELoss()(dmdt_pred, dmdt_values)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss = float(np.mean(loss_list)) if loss_list else float("nan")
        print(f"Epoch {epoch:03d} | Loss: {mean_loss:.6f}")
        writer.add_scalar("train/loss", mean_loss, epoch)

        if epoch % VIZ_INTERVAL == 0:
            try:
                visualize_network(
                    network,
                    epoch,
                    KERNEL_SIZE,
                    BATCH_SIZE,
                    LEARNING_RATE,
                    save_dir=str(OUTPUT_DIR),
                )
                # Log saved network analysis figure
                tanh_name = f"network_analysis_tanh_kernel{KERNEL_SIZE}_batch{BATCH_SIZE}_lr{LEARNING_RATE:.0e}_epoch{epoch:06d}.png"
                mlp_name = f"network_analysis_mlp_kernel{KERNEL_SIZE}_batch{BATCH_SIZE}_lr{LEARNING_RATE:.0e}_epoch{epoch:06d}.png"
                tanh_path = OUTPUT_DIR / tanh_name
                mlp_path = OUTPUT_DIR / mlp_name
                if tanh_path.exists():
                    _tb_add_saved_image(
                        writer, tanh_path, "fig/network_analysis", epoch
                    )
                elif mlp_path.exists():
                    _tb_add_saved_image(writer, mlp_path, "fig/network_analysis", epoch)
            except Exception as e:
                print(f"  Visualization failed at epoch {epoch}: {e}")

            # Kernel comparison
            try:
                with torch.no_grad():
                    k_tensor = network.kernel()
                learned = k_tensor[0, 0].detach().cpu().numpy()
                s = learned.sum()
                if s != 0:
                    learned = learned / s
                out_cmp = OUTPUT_DIR / f"kernel_compare_epoch{epoch:06d}.png"
                # Save comparison fig
                import matplotlib.pyplot as plt

                err = learned - ref_kernel
                vmin = float(min(learned.min(), ref_kernel.min()))
                vmax = float(max(learned.max(), ref_kernel.max()))
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                im0 = axes[0].imshow(ref_kernel, cmap="viridis", vmin=vmin, vmax=vmax)
                axes[0].set_title("Reference Gaussian (KxK)")
                plt.colorbar(im0, ax=axes[0], fraction=0.046)
                im1 = axes[1].imshow(learned, cmap="viridis", vmin=vmin, vmax=vmax)
                axes[1].set_title("Learned Kernel (KxK)")
                plt.colorbar(im1, ax=axes[1], fraction=0.046)
                eabs = float(max(abs(err.min()), abs(err.max())))
                im2 = axes[2].imshow(err, cmap="RdBu_r", vmin=-eabs, vmax=eabs)
                axes[2].set_title("Error (learned - ref)")
                plt.colorbar(im2, ax=axes[2], fraction=0.046)
                for ax in axes:
                    ax.axis("off")
                fig.tight_layout()
                fig.savefig(out_cmp, dpi=150, bbox_inches="tight")
                plt.close(fig)

                # Log saved comparison fig
                _tb_add_saved_image(writer, out_cmp, "fig/kernel_compare", epoch)

                l2 = float(np.sqrt(np.mean((learned - ref_kernel) ** 2)))
                l1 = float(np.mean(np.abs(learned - ref_kernel)))
                print(
                    f"  Kernel L2: {l2:.6e}, L1: {l1:.6e}, sum(learned)={learned.sum():.6f}"
                )
            except Exception as e:
                print(f"  Kernel validation failed at epoch {epoch}: {e}")

        if epoch % CKPT_INTERVAL == 0:
            try:
                ckpt_path = OUTPUT_DIR / f"epoch{epoch}.pth"
                torch.save(network.state_dict(), str(ckpt_path))
                print(f"  Checkpoint saved: {ckpt_path}")
            except Exception as e:
                print(f"  Checkpoint save failed at epoch {epoch}: {e}")

        if epoch % 100 == 0:
            writer.flush()

    assert np.isfinite(np.mean(loss_list))
    writer.close()


if __name__ == "__main__":
    main()
