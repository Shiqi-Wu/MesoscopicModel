"""
init_grf_ifft.py

Generate initial conditions for spin/PDE simulations using
Gaussian random fields (GRFs) via spectral synthesis.

Method:
- Sample complex Gaussian noise in Fourier domain with a
  prescribed power spectrum S(k).
- Impose Hermitian symmetry to ensure the field is real.
- Apply inverse FFT (IFFT) to obtain a real-space random field.
- Optionally rescale or threshold the field for use as an
  initial condition in Ising, Allen–Cahn, or other models.

Usage:
- Define system size (Lx, Ly).
- Provide desired power spectrum function S(k).
- Call generate_grf_field(...) to obtain a 2D numpy array
  representing the initial field configuration.

References:
- Spectral synthesis of Gaussian random fields.
- Applications in phase separation and nonlocal PDE models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Optional


# ---------- Spectral GRF (FFT) ----------
def _make_kgrid_rfft(L: int):
    """rFFT wavenumber grids (radians per lattice unit)."""
    kx = 2.0 * np.pi * np.fft.fftfreq(L)  # (L,)
    ky = 2.0 * np.pi * np.fft.rfftfreq(L)  # (L//2+1,)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return KX, KY


def grf_gaussian_spectrum_m_field(
    L: int,
    ell: float,
    sigma: float = 1.0,
    m0: float = 0.0,
    tau: float = 1.0,
    k_cut=None,
    butter_n=4,
    hard_cut=False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Spectral synthesis of a 2D GRF, then map to m(x) ∈ [-1,1]:
      1) φ = IFFT( sqrt(S(k)) * FFT(white) ), S(k) = σ^2 * exp( - (ell^2/2)|k|^2 )
      2) m = m0 + (1 - |m0|) * tanh( φ / τ )
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0, size=(L, L))  # white noise

    KX, KY = _make_kgrid_rfft(L)
    K2 = KX**2 + KY**2
    K = np.sqrt(K2)
    sqrt_S = sigma * np.exp(-0.25 * (ell**2) * K2)  # sqrt of Gaussian spectrum

    if k_cut is not None:
        if hard_cut:
            lp = (K <= float(k_cut)).astype(float)
        else:
            lp = 1.0 / (1.0 + (K / float(k_cut)) ** (2 * int(butter_n)))
        sqrt_S *= lp

    Wk = np.fft.rfftn(w)
    Phi_k = sqrt_S * Wk
    Phi_k[0, 0] = 0.0  # remove DC; mean controlled by m0

    phi = np.fft.irfftn(Phi_k, s=(L, L)).astype(np.float64)
    std = phi.std(ddof=0)
    if std > 0:
        phi *= sigma / std

    m = m0 + (1.0 - abs(m0)) * np.tanh(phi / max(tau, 1e-8))
    return np.clip(m, -1.0, 1.0)


# ---------- From m_field to spins ----------
def sample_spins_from_m_field(
    m_field: np.ndarray, seed: Optional[int] = None
) -> np.ndarray:
    """Given m(x) in [-1,1], sample σ∈{-1,+1} with P(σ=+1)=(1+m)/2 per site."""
    rng = np.random.default_rng(seed)
    m = np.clip(np.asarray(m_field, dtype=float), -1.0, 1.0)
    p = 0.5 * (1.0 + m)
    s = (rng.random(m.shape) < p).astype(np.int8)
    s[s == 0] = -1
    return s


# ---------- Coarse-grain (block average) ----------
def block_average(spin_2d: np.ndarray, block: int) -> np.ndarray:
    """
    Compute block-averaged field (coarse-grain).
    tiles = ceil(L/block). Edge blocks are truncated to remain inside [0,L).
    """
    L = spin_2d.shape[0]
    assert spin_2d.shape == (L, L)
    tiles = math.ceil(L / block)
    out = np.zeros((tiles, tiles), dtype=float)
    for i in range(tiles):
        for j in range(tiles):
            x0, x1 = i * block, min((i + 1) * block, L)
            y0, y1 = j * block, min((j + 1) * block, L)
            out[i, j] = np.mean(spin_2d[x0:x1, y0:y1])
    return out


# ---------- Plot helpers ----------
def plot_initial_and_coarse(
    spins_2d: np.ndarray,
    coarse_2d: np.ndarray,
    timestr: str = "t=0",
    save_path: Optional[str] = None,
):
    """Plot initial spins and coarse-grained field side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    ax = axes[0]
    im0 = ax.imshow(spins_2d, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(f"Initial spins ({timestr})")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    im1 = ax.imshow(coarse_2d, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title("Coarse-grained (block average)")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("block-avg spin")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    plt.close()


# Example: generate & plot
if __name__ == "__main__":
    # Lattice size
    L = 1024

    # Spectral parameters
    ell = 24.0  # correlation length (typical cluster size ~ ell)
    sigma = 1.2  # target std of φ (pre-tanh)
    tau = 1.0  # tanh strength (smaller -> closer to ±1)
    m0 = 0.1  # global mean magnetization
    seed = 0

    # 1) make m_field via spectral synthesis
    m_field = grf_gaussian_spectrum_m_field(
        L=L, ell=ell, sigma=sigma, m0=m0, tau=tau, seed=seed
    )

    # 2) sample Ising initial spins
    spins0 = sample_spins_from_m_field(m_field, seed=seed)

    # 3) coarse-grain (same as snaps_block_avg for a single frame)
    block = 8  # e.g., 8x8 blocks on a 1024x1024 lattice -> 128x128 coarse grid
    coarse0 = block_average(spins0, block=block)

    # 4) plot both
    plot_initial_and_coarse(
        spins0,
        coarse0,
        timestr="spectral init",
        save_path="results/init_grf_ifft_test/initial_and_coarse.png",
    )
