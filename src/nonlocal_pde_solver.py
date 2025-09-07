#!/usr/bin/env python3
"""
NONLOCAL Allen-Cahn PDE Solver

Core solver for NONLOCAL Allen-Cahn PDE:
∂t m = Γ(m)[∫ J(r-r') m(r') dr' - f'(m) + h]  (NONLOCAL with convolution)

The nonlocal term is a convolution integral instead of the local Laplacian.
This represents the original nonlocal interactions before gradient expansion.

Supports the same parameter sources as local solver:
1. Manual specification
2. Theory derivation from Ising simulation

Note: This solves the NONLOCAL Allen-Cahn equation. For local PDE,
see pde_solver.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import torch
from scipy import signal
import time
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

Array = np.ndarray


@dataclass
class NonlocalPDEParams:
    """
    Nonlocal PDE parameter container for nonlocal Allen-Cahn equation:
    ∂t m = Γ(m)[∫ J(r-r') m(r') dr' - f'(m) + h]
    """

    # Core nonlocal PDE parameters
    interaction_strength: float  # J₀ - interaction strength
    interaction_range: float  # ε - interaction range
    mobility_prefactor: float  # Γ₀ in Γ(m) = Γ₀(1-m²)
    h_field: float  # external field h
    potential_a: float = 1.0  # cubic term in f'(m) = am³ + bm
    potential_b: float = -1.0  # linear term

    # Grid parameters
    M: int = 128  # macro grid size
    Lx: float = 1.0  # domain size x
    Ly: float = 1.0  # domain size y

    # Time integration
    dt: Optional[float] = None  # time step (auto if None)
    steps: int = 1000  # number of steps
    record_every: int = 10  # recording frequency

    # Nonlocal kernel parameters
    kernel_type: str = "gaussian"  # "exponential" or "gaussian"
    kernel_cutoff: float = 5.0  # cutoff in units of interaction_range

    # Metadata
    source: str = "unknown"  # "manual" or "theory"
    source_info: Dict[str, Any] = None

    def __post_init__(self):
        if self.source_info is None:
            self.source_info = {}


class NonlocalParameterProvider:
    """Base class for nonlocal PDE parameter providers"""

    def get_params(self) -> NonlocalPDEParams:
        """Get nonlocal PDE parameters"""
        raise NotImplementedError

    def get_initial_condition(self) -> Array:
        """Get initial condition for PDE"""
        raise NotImplementedError

    def get_reference_data(self) -> Dict[str, Any]:
        """Get reference data for comparison"""
        raise NotImplementedError


class ManualNonlocalProvider(NonlocalParameterProvider):
    """Manual parameter specification for nonlocal PDE"""

    def __init__(
        self,
        interaction_strength: float = 1.0,
        interaction_range: float = 0.1,
        mobility_prefactor: float = 1.0,
        h_field: float = 0.0,
        M: int = 128,
        initial_condition: Optional[Array] = None,
        m0: float = 0.0,
        noise_amp: float = 0.1,
        seed: int = 0,
        **kwargs,
    ):
        self.interaction_strength = interaction_strength
        self.interaction_range = interaction_range
        self.mobility_prefactor = mobility_prefactor
        self.h_field = h_field
        self.M = M
        self.initial_condition = initial_condition
        self.m0 = m0
        self.noise_amp = noise_amp
        self.seed = seed
        self.extra_params = kwargs

    def get_params(self) -> NonlocalPDEParams:
        return NonlocalPDEParams(
            interaction_strength=self.interaction_strength,
            interaction_range=self.interaction_range,
            mobility_prefactor=self.mobility_prefactor,
            h_field=self.h_field,
            M=self.M,
            source="manual",
            source_info={"description": "Manual nonlocal parameters"},
            **self.extra_params,
        )

    def get_initial_condition(self) -> Array:
        if self.initial_condition is not None:
            return self.initial_condition.copy()

        rng = np.random.default_rng(self.seed)
        phi = np.full((self.M, self.M), self.m0, dtype=np.float32)
        if self.noise_amp > 0:
            phi += self.noise_amp * rng.standard_normal((self.M, self.M)).astype(
                np.float32
            )
        return np.clip(phi, -1.0, 1.0)

    def get_reference_data(self) -> Dict[str, Any]:
        return {"micro_spins": None, "macro_field": None, "times": None}


class TheoryNonlocalProvider(NonlocalParameterProvider):
    """Theory-derived nonlocal parameters from Ising simulation"""

    def __init__(self, data_path: str, frame_idx: int = 0, **kwargs):
        self.data_path = data_path
        self.frame_idx = frame_idx
        self.extra_params = kwargs

        # Load Ising data
        self._load_ising_data()

    def _load_ising_data(self):
        """Load Ising simulation data"""
        try:
            # Load .npz file
            data = np.load(self.data_path)

            # Load corresponding .json file
            json_path = self.data_path.replace(".npz", ".json")
            import json

            with open(json_path, "r") as f:
                self.meta = json.load(f)

            self.times = data["times"]
            self.spins = data["spins"]  # (T, L, L)
            self.spins_meso = data["spins_meso"]  # (T, M, M)

            if self.spins is None:
                raise ValueError(f"No spins data in {self.data_path}")
            if self.frame_idx >= self.spins.shape[0]:
                raise IndexError(f"frame_idx {self.frame_idx} >= {self.spins.shape[0]}")

            self.B = int(self.meta["block"])
            self.M = int(self.meta["M"])
            self.L = int(self.meta["L"])

        except Exception as e:
            raise ValueError(f"Failed to load data from {self.data_path}: {e}")

    def _derive_nonlocal_theory_params(self) -> Dict[str, float]:
        """Derive nonlocal PDE parameters from Ising theory"""
        T = float(self.meta["T"])
        h = float(self.meta["h"])
        J = float(self.meta["J"])

        # Theory derivation for nonlocal case
        beta = 1.0 / T
        lambda_rate = 1.0  # Corrected: 1 MCSS = 1/λ PDE time, λ=1
        mobility_prefactor = lambda_rate * beta

        # Nonlocal parameters: should match microscopic scale for theory consistency
        # interaction_range = ε = B/L (microscopic grid spacing)
        # This is the characteristic scale of the original Ising interactions
        epsilon = self.B / self.L  # microscopic spacing
        interaction_range = epsilon  # Use actual microscopic scale

        # interaction_strength = J (preserve original coupling strength)
        interaction_strength = J

        return {
            "T": T,
            "h": h,
            "J": J,
            "beta": beta,
            "lambda_rate": lambda_rate,
            "mobility_prefactor": mobility_prefactor,
            "interaction_strength": interaction_strength,
            "interaction_range": interaction_range,
        }

    def get_params(self) -> NonlocalPDEParams:
        theory_params = self._derive_nonlocal_theory_params()

        # Use the same time step settings as the original data
        snapshot_dt = float(self.meta["snapshot_dt"])
        t_end = float(self.meta["t_end"])
        steps = int(t_end / snapshot_dt)

        return NonlocalPDEParams(
            interaction_strength=theory_params["interaction_strength"],
            interaction_range=theory_params["interaction_range"],
            mobility_prefactor=theory_params["mobility_prefactor"],
            h_field=theory_params["h"],
            M=self.M,
            dt=snapshot_dt,  # Use the same dt as original data
            steps=steps,  # Use the same number of steps as original data
            record_every=1,  # Record every step to match original data
            source="theory",
            source_info={
                "data_path": self.data_path,
                "derived_params": theory_params,
                "description": f"Nonlocal theory from {self.data_path}",
            },
            **self.extra_params,
        )

    def get_initial_condition(self) -> Array:
        # Use the mesoscopic field directly
        return self.spins_meso[self.frame_idx].astype(np.float32)

    def get_reference_data(self) -> Dict[str, Any]:
        return {
            "micro_spins": self.spins,
            "macro_field": self.spins_meso,
            "times": self.times,
            "meta": self.meta,
        }


def create_nonlocal_parameter_provider(
    source: Union[str, Dict[str, Any]], **kwargs
) -> NonlocalParameterProvider:
    """
    Factory to create nonlocal parameter providers

    Args:
        source: "manual" or data path, or config dict
        **kwargs: Additional parameters

    Examples:
        create_nonlocal_parameter_provider("manual", interaction_strength=1.0)
        create_nonlocal_parameter_provider("data/ct_glauber_.../round0/ct_glauber_....npz")
    """
    if isinstance(source, str):
        if source == "manual":
            return ManualNonlocalProvider(**kwargs)
        else:
            return TheoryNonlocalProvider(data_path=source, **kwargs)
    elif isinstance(source, dict):
        source_type = source.get("type", "manual")
        source_kwargs = {**source, **kwargs}
        source_kwargs.pop("type", None)

        if source_type == "manual":
            return ManualNonlocalProvider(**source_kwargs)
        elif source_type == "theory":
            return TheoryNonlocalProvider(**source_kwargs)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
    else:
        raise ValueError(f"Invalid source: {source}")


# Core nonlocal solver functions


def _create_interaction_kernel(
    M: int,
    Lx: float,
    Ly: float,
    interaction_range: float,
    kernel_type: str = "gaussian",
    cutoff: float = 5.0,
) -> Array:
    """
    Create nonlocal interaction kernel J(r-r')

    Args:
        M: Grid size
        Lx, Ly: Domain size
        interaction_range: ε - characteristic interaction range
        kernel_type: "exponential" or "gaussian"
        cutoff: Cutoff in units of interaction_range

    Returns:
        Normalized interaction kernel in Fourier space (for efficiency)
    """
    dx = Lx / M
    dy = Ly / M

    # Create coordinate grids (centered at 0)
    x = np.fft.fftfreq(M, dx) * 2 * np.pi  # k_x
    y = np.fft.fftfreq(M, dy) * 2 * np.pi  # k_y
    kx, ky = np.meshgrid(x, y, indexing="ij")
    k = np.sqrt(kx**2 + ky**2)

    # Create kernel in k-space for different types
    if kernel_type == "exponential":
        # J(k) = J₀ / (1 + (k*ε)²) for exponential decay
        # This gives J(r) ∝ exp(-|r|/ε) in real space
        kernel_k = 1.0 / (1.0 + (k * interaction_range) ** 2)
    elif kernel_type == "gaussian":
        # J(k) = J₀ * exp(-(k*ε)²/2) for Gaussian
        # This gives J(r) ∝ exp(-r²/(2ε²)) in real space
        kernel_k = np.exp(-((k * interaction_range) ** 2) / 2.0)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    # Apply cutoff in k-space
    cutoff_k = cutoff / interaction_range
    kernel_k[k > cutoff_k] = 0.0

    # Simple normalization for numerical stability
    # For nonlocal Allen-Cahn, the key is that the kernel represents
    # the interaction shape, with strength controlled by interaction_strength
    # Normalize the maximum value to prevent numerical explosion

    if np.max(kernel_k) > 0:
        kernel_k /= np.max(kernel_k)  # Normalize max to 1

    return kernel_k


def _nonlocal_convolution_fft(m: Array, kernel_k: Array) -> Array:
    """
    Compute nonlocal convolution ∫ J(r-r') m(r') dr' using FFT

    Args:
        m: Field values
        kernel_k: Interaction kernel in Fourier space

    Returns:
        Convolution result
    """
    # FFT of field
    m_k = np.fft.fft2(m)

    # Convolution in k-space is multiplication
    result_k = kernel_k * m_k

    # Inverse FFT to get real space result
    result = np.fft.ifft2(result_k).real

    return result


def _gamma_mobility_nonlocal(m: Array, mobility_prefactor: float) -> Array:
    """Glauber mobility for nonlocal case: Γ(m) = Γ₀(1-m²)"""
    return mobility_prefactor * (1.0 - m * m)


def solve_nonlocal_allen_cahn(
    provider: NonlocalParameterProvider,
    method: str = "rk4",
    show_progress: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    NONLOCAL Allen-Cahn PDE solver

    Solves: ∂t m = Γ(m)[∫ J(r-r') m(r') dr' - f'(m) + h] with NONLOCAL convolution

    Args:
        provider: Nonlocal parameter provider (manual or theory-derived)
        method: Integration method ('euler' or 'rk4')
        show_progress: Whether to show progress updates
        seed: Random seed (currently unused)

    Returns:
        Solution dictionary with times, phi, params, reference_data
    """
    # Get parameters and initial condition
    params = provider.get_params()
    phi0 = provider.get_initial_condition()

    # Grid setup
    M = params.M
    dx = params.Lx / M
    dy = params.Ly / M

    # Create nonlocal interaction kernel
    kernel_k = _create_interaction_kernel(
        M,
        params.Lx,
        params.Ly,
        params.interaction_range,
        params.kernel_type,
        params.kernel_cutoff,
    )

    # Auto time step (more conservative for nonlocal case)
    if params.dt is None:
        # For nonlocal case, be much more conservative with time step
        max_mobility = params.mobility_prefactor
        max_interaction = params.interaction_strength

        # Very conservative time step for stability
        dt = (
            0.01 / (max_mobility + max_interaction)
            if (max_mobility + max_interaction) > 0
            else 1e-4
        )
        dt = min(0.005, dt)  # Much smaller max timestep
        dt = max(1e-6, dt)  # Reasonable min timestep
    else:
        dt = params.dt

    # Storage
    n_rec = params.steps // params.record_every
    times = np.empty(n_rec + 1, dtype=np.float32)
    snaps = np.empty((n_rec + 1, M, M), dtype=np.float32)

    # Initialize
    phi = phi0.copy()
    t = 0
    k = 0
    times[k] = 0
    snaps[k] = phi
    k += 1

    if show_progress:
        print(
            f"Nonlocal solver: dt={dt:.2e}, interaction_range={params.interaction_range:.2e}"
        )
        print(f"Grid size: {M}x{M}, Steps: {params.steps}, Method: {method.upper()}")

    start_time = time.time()

    # Time evolution: NONLOCAL Allen-Cahn: ∂t m = Γ(m)[∫ J(r-r') m(r') dr' - f'(m) + h]
    for step in range(params.steps):
        # Nonlocal convolution term
        nonlocal_term = _nonlocal_convolution_fft(phi, kernel_k)
        nonlocal_term *= params.interaction_strength

        # Mobility and potential
        mobility = _gamma_mobility_nonlocal(phi, params.mobility_prefactor)
        f_prime = params.potential_a * (phi**3) + params.potential_b * phi

        # RHS: Γ(m)[∫ J(r-r') m(r') dr' - f'(m) + h]
        rhs = mobility * (nonlocal_term - f_prime + params.h_field)

        # Alternative Glauber-Kac formulation (commented out):
        # beta = 1.0 / T  # where T comes from params
        # rhs = -phi + np.tanh(beta * nonlocal_term + beta * params.h_field)

        # Time step
        if method.lower() == "euler":
            phi = phi + dt * rhs
        elif method.lower() == "rk4":
            # Runge-Kutta 4th order
            k1 = rhs
            k2 = _gamma_mobility_nonlocal(
                phi + 0.5 * dt * k1, params.mobility_prefactor
            ) * (
                _nonlocal_convolution_fft(phi + 0.5 * dt * k1, kernel_k)
                * params.interaction_strength
                - params.potential_a * ((phi + 0.5 * dt * k1) ** 3)
                - params.potential_b * (phi + 0.5 * dt * k1)
                + params.h_field
            )
            k3 = _gamma_mobility_nonlocal(
                phi + 0.5 * dt * k2, params.mobility_prefactor
            ) * (
                _nonlocal_convolution_fft(phi + 0.5 * dt * k2, kernel_k)
                * params.interaction_strength
                - params.potential_a * ((phi + 0.5 * dt * k2) ** 3)
                - params.potential_b * (phi + 0.5 * dt * k2)
                + params.h_field
            )
            k4 = _gamma_mobility_nonlocal(phi + dt * k3, params.mobility_prefactor) * (
                _nonlocal_convolution_fft(phi + dt * k3, kernel_k)
                * params.interaction_strength
                - params.potential_a * ((phi + dt * k3) ** 3)
                - params.potential_b * (phi + dt * k3)
                + params.h_field
            )
            phi = phi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"Unknown method: {method}")

        phi = np.clip(phi, -1.0, 1.0)

        t += dt
        if (step + 1) % params.record_every == 0:
            times[k] = t
            snaps[k] = phi
            k += 1

            if show_progress and (step + 1) % (params.record_every * 10) == 0:
                elapsed = time.time() - start_time
                progress = 100.0 * (step + 1) / params.steps
                print(
                    f"  Step {step+1:4d}/{params.steps} ({progress:5.1f}%) - Elapsed: {elapsed:6.1f}s"
                )

    if show_progress:
        total_time = time.time() - start_time
        print(f"✅ Nonlocal solution completed in {total_time:.1f} seconds")

    return {
        "times": times,
        "phi": snaps,
        "params": params,
        "dt": dt,
        "kernel_k": kernel_k,  # Include kernel for analysis
        "reference_data": provider.get_reference_data(),
    }


# Convenience function for direct usage
def solve_nonlocal_pde_from_data(
    data_path: str, method: str = "rk4", show_progress: bool = True, **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to solve nonlocal PDE directly from data path

    Args:
        data_path: Path to .npz data file
        method: Integration method ('euler' or 'rk4')
        show_progress: Whether to show progress
        **kwargs: Additional parameters for TheoryNonlocalProvider

    Returns:
        Solution dictionary
    """
    provider = TheoryNonlocalProvider(data_path=data_path, **kwargs)
    return solve_nonlocal_allen_cahn(
        provider, method=method, show_progress=show_progress
    )
