#!/usr/bin/env python3
"""
Nonlocal Glauber–Kac PDE Solver (tanh form)

Core solver for the nonlocal Glauber–Kac evolution:
  ∂t m(t,x) = - m(t,x) + tanh( β [ (J * m)(t,x)·J0 + h ] )

where (J * m) denotes periodic convolution with a normalized interaction kernel J
of range ε. The kernel J is discretely normalized so that sum(J)·dx·dy = 1, and
the overall interaction strength is controlled by J0 (= interaction_strength).

Parameter sources:
1. Manual specification
2. Theory derivation from Ising simulation metadata

For the local PDE utilities (including Laplacian-based models), see pde_solver.py.
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
    Parameters for the nonlocal Glauber–Kac PDE:
      ∂t m = -m + tanh( β ( J0 (J * m) + h ) )

    Notes:
    - mobility_prefactor is preserved for backward compatibility but unused here
    - potential_a/potential_b are unused in the tanh formulation
    """

    # Core nonlocal PDE parameters
    interaction_strength: float  # J₀ - interaction strength
    interaction_range: float  # ε - interaction range
    mobility_prefactor: float  # kept for backward-compatibility (unused in tanh form)
    h_field: float  # external field h
    beta: float = 1.0  # inverse temperature β for tanh-based Glauber–Kac
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

        # Allow overrides from extra_params (without duplicating keywords)
        interaction_range = self.extra_params.get(
            "interaction_range", theory_params["interaction_range"]
        )
        interaction_strength = self.extra_params.get(
            "interaction_strength", theory_params["interaction_strength"]
        )
        beta = self.extra_params.get("beta", theory_params["beta"])
        h_field = self.extra_params.get("h_field", theory_params["h"])
        mobility_prefactor = self.extra_params.get(
            "mobility_prefactor", theory_params["mobility_prefactor"]
        )

        return NonlocalPDEParams(
            interaction_strength=interaction_strength,
            interaction_range=interaction_range,
            mobility_prefactor=mobility_prefactor,
            h_field=h_field,
            beta=beta,
            M=self.M,
            dt=snapshot_dt,  # Use the same dt as original data
            steps=steps,  # Use the same number of steps as original data
            record_every=1,  # Record every step to match original data
            source="theory",
            source_info={
                "data_path": self.data_path,
                "derived_params": theory_params,
                "description": f"Nonlocal theory from {self.data_path}",
                "overrides": {
                    k: v
                    for k, v in self.extra_params.items()
                    if k
                    in {
                        "interaction_range",
                        "interaction_strength",
                        "beta",
                        "h_field",
                        "mobility_prefactor",
                    }
                },
            },
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
    cutoff: float = 0.0,
) -> Array:
    """
    Create nonlocal interaction kernel in Fourier space via real-space construction.

    Steps:
      1) Build J(r) on the periodic grid using the shortest-image metric
      2) Normalize s.t. sum(J) * dx * dy = 1 (unit integral)
      3) ifftshift to zero-phase, then FFT to obtain kernel_k
      4) Optionally apply a high-k cutoff (rarely necessary)

    Returns:
        kernel_k: FFT2(J_r_rolled)
    """
    dx = Lx / M
    dy = Ly / M

    # periodic shortest distances in each axis
    grid = np.arange(M, dtype=np.float32)
    rx = np.minimum(grid, M - grid) * dx
    ry = rx  # same spacing in x and y
    RX, RY = np.meshgrid(rx, ry, indexing="ij")
    R = np.sqrt(RX * RX + RY * RY)

    # Real-space kernel
    if kernel_type == "gaussian":
        # Continuous normalized Gaussian then renormalize discretely
        eps2 = interaction_range * interaction_range
        Jr = np.exp(-(R * R) / (2.0 * eps2)) / (2.0 * np.pi * eps2)
    elif kernel_type == "exponential":
        # 2D isotropic exponential with continuous normalization ~ 1/(2π ε^2)
        # Using K(r) ∝ exp(-r/ε) / (2π ε^2), then renormalize discretely
        Jr = np.exp(-R / max(interaction_range, 1e-12))
        Jr[Jr < 1e-20] = 0.0
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    # Discrete normalization so that sum(Jr) = 1 (so conv(const) = const)
    sumJ = Jr.sum()
    if sumJ > 0:
        Jr = Jr / sumJ

    # Zero-phase alignment for cyclic convolution
    Jr_rolled = np.fft.ifftshift(Jr)
    kernel_k = np.fft.fft2(Jr_rolled)

    # Optional high-k cutoff
    if cutoff and cutoff > 0.0:
        kx = np.fft.fftfreq(M, d=dx) * 2.0 * np.pi
        ky = np.fft.fftfreq(M, d=dy) * 2.0 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        KK = np.sqrt(KX * KX + KY * KY)
        cutoff_k = cutoff / max(interaction_range, 1e-12)
        mask = KK > cutoff_k
        kernel_k[mask] = 0.0

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


def solve_nonlocal_allen_cahn(
    provider: NonlocalParameterProvider,
    method: str = "rk4",
    show_progress: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Nonlocal Glauber–Kac PDE solver (tanh form)

    Solves: ∂t m = -m + tanh( β ( J0 (J ⊛ m) + h ) ) with periodic convolution

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

    # Auto time step (conservative for tanh-based nonlocal case)
    if params.dt is None:
        # Conservative estimate using β and J0; ensures stability for stiff regimes
        max_interaction = max(1e-12, params.interaction_strength)
        dt = 0.01 / (params.beta * max_interaction + 1.0)
        dt = min(0.005, dt)
        dt = max(1e-6, dt)
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

    # Time evolution: Glauber–Kac: ∂t m = -m + tanh(β (J0 * (J ⊛ m) + h))
    for step in range(params.steps):

        def rhs(state: Array) -> Array:
            conv = _nonlocal_convolution_fft(state, kernel_k)
            field = params.interaction_strength * conv + params.h_field
            return -state + np.tanh(params.beta * field)

        # Time step
        if method.lower() == "euler":
            phi = phi + dt * rhs(phi)
        elif method.lower() == "rk4":
            # Runge-Kutta 4th order
            k1 = rhs(phi)
            k2 = rhs(phi + 0.5 * dt * k1)
            k3 = rhs(phi + 0.5 * dt * k2)
            k4 = rhs(phi + dt * k3)
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
