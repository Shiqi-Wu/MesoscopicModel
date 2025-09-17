#!/usr/bin/env python3
"""
Local PDE Solver (Allen–Cahn type from hydrodynamic tanh model)

Implements the local PDE obtained from the nonlocal Glauber–Kac model via
formal expansion (consistent with notes Eq. (21)):

  ∂t m = κ Δ m - f(m) + β h,

with f(m) = - r m - u m^3 (keeping leading orders), so that

  ∂t m = κ Δ m + r m + u m^3 + β h.

Parameter mapping (J0 ≡ 1, ∫J=1):
  κ = β (m2 / (2 d)) γ^2,  r = 1 - β,  u = β^3/3,
and for an isotropic Gaussian kernel with range ε=γ (d dimensions), m2 = d ε^2
⇒ κ = (β/2) ε^2.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import time
import os
import sys

# Ensure local imports work when called from tests
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


Array = np.ndarray


@dataclass
class LocalPDEParams:
    # Core PDE parameters
    kappa: float
    r: float
    u: float
    beta: float
    h_field: float
    # Interaction strength (from data/meta; used by tanh-local RHS)
    J0: float = 1.0
    # RHS mode: "poly" (default cubic AC) or "tanh_local" (tanh local-limit)
    rhs_mode: str = "poly"

    # Grid/domain
    M: int = 128
    Lx: float = 1.0
    Ly: float = 1.0

    # Time integration
    dt: Optional[float] = None
    steps: int = 1000
    record_every: int = 10

    # Metadata
    source: str = "unknown"
    source_info: Dict[str, Any] = None

    # Numerical control
    enforce_cfl: bool = True

    def __post_init__(self):
        if self.source_info is None:
            self.source_info = {}


class LocalParameterProvider:
    def get_params(self) -> LocalPDEParams:
        raise NotImplementedError

    def get_initial_condition(self) -> Array:
        raise NotImplementedError

    def get_reference_data(self) -> Dict[str, Any]:
        raise NotImplementedError


class ManualLocalProvider(LocalParameterProvider):
    def __init__(
        self,
        kappa: float,
        r: float,
        u: float,
        beta: float,
        h_field: float = 0.0,
        M: int = 128,
        initial_condition: Optional[Array] = None,
        seed: int = 0,
        **kwargs,
    ):
        self.kappa = kappa
        self.r = r
        self.u = u
        self.beta = beta
        self.h_field = h_field
        self.M = M
        self.initial_condition = initial_condition
        self.seed = seed
        self.extra = kwargs

    def get_params(self) -> LocalPDEParams:
        # Avoid passing rhs_mode twice; extract it before forwarding extras
        extras = dict(self.extra)
        rhs_mode = str(extras.pop("rhs_mode", "poly"))
        J0_val = float(extras.pop("J0", 1.0))
        return LocalPDEParams(
            kappa=self.kappa,
            r=self.r,
            u=self.u,
            beta=self.beta,
            h_field=self.h_field,
            M=self.M,
            J0=J0_val,
            rhs_mode=rhs_mode,
            source="manual",
            source_info={"description": "Manual local PDE params"},
            **extras,
        )

    def get_initial_condition(self) -> Array:
        if self.initial_condition is not None:
            return self.initial_condition.copy()
        rng = np.random.default_rng(self.seed)
        m0 = rng.standard_normal((self.M, self.M)).astype(np.float32) * 0.01
        return np.clip(m0, -1.0, 1.0)

    def get_reference_data(self) -> Dict[str, Any]:
        return {"macro_field": None, "times": None, "meta": {}}


class TheoryLocalProvider(LocalParameterProvider):
    """Derive κ,r,u from data metadata using notes' mapping (J0=1)."""

    def __init__(self, data_path: str, frame_idx: int = 0, **kwargs):
        self.data_path = data_path
        self.frame_idx = frame_idx
        self.extra = kwargs
        self._load_ising_data()

    def _load_ising_data(self):
        try:
            data = np.load(self.data_path)
            json_path = self.data_path.replace(".npz", ".json")
            import json

            with open(json_path, "r") as f:
                self.meta = json.load(f)

            self.times = data["times"]
            # Use mesoscopic field if available, else coarse-grain must be handled outside
            self.spins_meso = data.get("spins_meso")
            self.M = int(self.meta["M"])  # macro size
            self.B = int(self.meta.get("block", 1))
            self.L = int(self.meta.get("L", self.M * self.B))
        except Exception as e:
            raise ValueError(f"Failed to load data for local provider: {e}")

    def _map_params_from_meta(self) -> Dict[str, float]:
        T = float(self.meta["T"])  # absolute temperature
        beta = 1.0 / T
        h = float(self.meta["h"]) if "h" in self.meta else 0.0
        J0 = float(self.meta.get("J", 1.0))

        # Use epsilon (interaction range) from filename/meta when available
        eps = None
        if "epsilon" in self.meta:
            eps = float(self.meta["epsilon"])  # optional
        else:
            # Try parse from path: "..._epsilon0.015625_..."
            import re

            m = re.search(r"epsilon([0-9]*\.?[0-9]+)", self.data_path)
            if m:
                eps = float(m.group(1))
            else:
                # Fallback: block/L as microscopic spacing (rough scale)
                eps = self.B / self.L

        # Local limit of tanh-Glauber–Kac (keep up to cubic in m):
        # ∂t m ≈ κ Δ m + (β J0 − 1) m − (β J0)^3 m^3 / 3 + β h
        # with κ = β J0 · (ε^2 / 2) for a Gaussian kernel in 2D.
        kappa = 0.5 * beta * J0 * (eps**2)
        r = beta * J0 - 1.0
        # Store a positive u and use a stabilizing minus sign in RHS
        u = (beta * J0) ** 3 / 3.0

        return {
            "beta": beta,
            "h": h,
            "J0": J0,
            "kappa": kappa,
            "r": r,
            "u": u,
            "eps": eps,
        }

    def get_params(self) -> LocalPDEParams:
        mapped = self._map_params_from_meta()
        snapshot_dt = float(self.meta["snapshot_dt"])
        t_end = float(self.meta["t_end"])
        steps = int(t_end / snapshot_dt)
        return LocalPDEParams(
            kappa=mapped["kappa"],
            r=mapped["r"],
            u=mapped["u"],
            beta=mapped["beta"],
            h_field=mapped["h"],
            M=self.M,
            J0=mapped["J0"],
            rhs_mode=str(self.extra.get("rhs_mode", "poly")),
            dt=snapshot_dt,
            steps=steps,
            record_every=1,
            source="theory",
            source_info={"data_path": self.data_path, "mapped": mapped},
        )

    def get_initial_condition(self) -> Array:
        if self.spins_meso is not None:
            return self.spins_meso[self.frame_idx].astype(np.float32)
        raise ValueError("spins_meso not found in data (expected mesoscopic field)")

    def get_reference_data(self) -> Dict[str, Any]:
        return {"macro_field": self.spins_meso, "times": self.times, "meta": self.meta}


def create_local_parameter_provider(
    source: str | Dict[str, Any], **kwargs
) -> LocalParameterProvider:
    if isinstance(source, str):
        if source == "manual":
            return ManualLocalProvider(**kwargs)
        else:
            return TheoryLocalProvider(data_path=source, **kwargs)
    elif isinstance(source, dict):
        src_type = source.get("type", "manual")
        cfg = {**source, **kwargs}
        cfg.pop("type", None)
        if src_type == "manual":
            return ManualLocalProvider(**cfg)
        elif src_type == "theory":
            return TheoryLocalProvider(**cfg)
        else:
            raise ValueError(f"Unknown source type: {src_type}")
    else:
        raise ValueError("Invalid source for local provider")


def _laplacian_fft(m: Array, Lx: float, Ly: float) -> Array:
    Mx, My = m.shape
    kx = np.fft.fftfreq(Mx, d=Lx / Mx) * 2.0 * np.pi
    ky = np.fft.fftfreq(My, d=Ly / My) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2 = KX * KX + KY * KY
    mk = np.fft.fft2(m)
    lap_k = -k2 * mk
    lap = np.fft.ifft2(lap_k).real
    return lap


def solve_local_pde(
    provider: LocalParameterProvider,
    method: str = "rk4",
    show_progress: bool = True,
) -> Dict[str, Any]:
    params = provider.get_params()
    m = provider.get_initial_condition().astype(np.float32)

    M = params.M
    Lx = params.Lx
    Ly = params.Ly
    dt_in = params.dt if params.dt is not None else 0.01

    # Maintain same physical time horizon; optionally enforce RK4 stability for diffusion part
    dx = Lx / M
    dy = Ly / M
    if getattr(params, "enforce_cfl", True):
        # k_max^2 ≈ (π/dx)^2 + (π/dy)^2 on periodic grid
        k2_max = (np.pi / dx) ** 2 + (np.pi / dy) ** 2
        rk4_alpha = 2.785  # stability extent on negative real axis for RK4
        safety = 0.5
        dt_cfl = safety * rk4_alpha / max(params.kappa * k2_max, 1e-12)
        dt = min(dt_in, dt_cfl)
    else:
        dt = dt_in

    t_end = params.steps * dt_in
    steps = int(np.ceil(t_end / dt))
    record_every = max(1, params.record_every)

    n_rec = steps // record_every
    times = np.empty(n_rec + 1, dtype=np.float32)
    traj = np.empty((n_rec + 1, M, M), dtype=np.float32)

    state = m.copy()
    t = 0.0
    k = 0
    times[k] = 0.0
    traj[k] = state
    k += 1

    if show_progress:
        if getattr(params, "rhs_mode", "poly") == "tanh_local":
            print(
                f"Local solver (tanh_local): dt_in={dt_in:.3e}, dt_eff={dt:.3e}, steps={steps}, kappa={params.kappa:.3e}, J0={params.J0:.3e}"
            )
        else:
            print(
                f"Local solver (poly): dt_in={dt_in:.3e}, dt_eff={dt:.3e}, steps={steps}, kappa={params.kappa:.3e}, r={params.r:.3e}, u={params.u:.3e}"
            )

    def rhs(s: Array) -> Array:
        lap = _laplacian_fft(s, Lx, Ly)
        mode = getattr(params, "rhs_mode", "poly").lower()
        if mode == "tanh_local":
            # tanh local-limit: ∂t m = -m + tanh( β(J0 m + h) + κ Δm )
            return -s + np.tanh(
                params.beta * (params.J0 * s + params.h_field) + params.kappa * lap
            )
        else:
            # Polynomial AC: ∂t m = κ Δm + r m − u m^3 + β h
            reaction = params.r * s - params.u * (s**3) + params.beta * params.h_field
            return params.kappa * lap + reaction

    start = time.time()
    for step in range(steps):
        if method.lower() == "euler":
            state = state + dt * rhs(state)
        elif method.lower() == "rk4":
            k1 = rhs(state)
            k2 = rhs(state + 0.5 * dt * k1)
            k3 = rhs(state + 0.5 * dt * k2)
            k4 = rhs(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"Unknown method: {method}")

        state = np.clip(state, -1.0, 1.0)
        t += dt
        if (step + 1) % record_every == 0:
            times[k] = t
            traj[k] = state
            k += 1
            if show_progress and (step + 1) % (record_every * 10) == 0:
                elapsed = time.time() - start
                print(f"  Step {step+1}/{steps}  t={t:.3f}  elapsed={elapsed:.1f}s")

    if show_progress:
        print(f"✅ Local PDE solution completed in {time.time() - start:.1f}s")

    return {
        "times": times,
        "phi": traj,
        "params": params,
        "dt": dt,
        "reference_data": provider.get_reference_data(),
    }
