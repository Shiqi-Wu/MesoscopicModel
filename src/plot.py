#!/usr/bin/env python3
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import argparse

def plot_Ms_Es(times, Ms, Es, outdir: Path):
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True, constrained_layout=True)
    ax[0].plot(times, Ms, lw=1.5, label="Magnetization")
    ax[0].set_ylabel("Magnetization per spin")
    ax[0].grid(alpha=0.3)

    ax[1].plot(times, Es, lw=1.5, label="Energy", color="tab:orange")
    ax[1].set_ylabel("Energy per spin")
    ax[1].set_xlabel("Time")
    ax[1].grid(alpha=0.3)

    out_path = outdir / "Magnetization_Energy.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[save] {out_path}")

def plot_animation(times, spins, spins_meso, outdir: Path, interval=200):
    """生成动图，左边 spins，右边 spins_meso，统一显示大小"""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    im1 = axes[0].imshow(spins[0], cmap="grey", vmin=-1, vmax=1)
    im2 = axes[1].imshow(spins_meso[0], cmap="grey", vmin=-1, vmax=1)

    axes[0].set_title("Spins (micro)")
    axes[1].set_title("Spins (meso)")
    for ax in axes:
        ax.axis("off")

    def update(frame):
        im1.set_data(spins[frame])
        im2.set_data(spins_meso[frame])
        fig.suptitle(f"t = {times[frame]:.2f}")
        return im1, im2

    ani = animation.FuncAnimation(fig, update, frames=len(times),
                                  interval=interval, blit=False)

    try:
        mp4_path = outdir / "animation.mp4"
        ani.save(mp4_path, writer="ffmpeg", dpi=150)
        print(f"[save] {mp4_path}")
    except Exception as e:
        print(f"[warn] mp4 save failed ({e}), falling back to GIF")
        gif_path = outdir / "animation.gif"
        ani.save(gif_path, writer="pillow", dpi=100)
        print(f"[save] {gif_path}")

    print(f"[save] {mp4_path}")
    print(f"[save] {gif_path}")