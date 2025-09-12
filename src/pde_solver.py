"""
PDE solvers for nonlocal Glauber–Kac (tanh form) and local variants.

Nonlocal (tanh form):
  ∂_t m(t,x) = -m(t,x) + tanh(β (J_γ * m)(t,x) + β h)

Local variants provided for comparison:
  (A) Tanh local limit: ∂_t m = -m + tanh(β (κ ∇²m + h))
  (B) Classical AC:     ∂_t m = κ ∇²m - (a m^3 + b m) + h

Periodic boundary conditions are used throughout.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import warnings


class NonlocalAllenCahnSolver:
    """
    Solver for the nonlocal Allen-Cahn equation with periodic boundary conditions.

    The equation solved is:
    ∂_t m(t,x) = -m(t,x) + tanh(β(J_γ * m)(t,x) + βh)

    where J_γ is a Gaussian kernel:
    J_γ(x) = (1/(2πγ²)^(d/2)) * exp(-|x|²/(2γ²))
    """

    def __init__(
        self,
        spatial_resolution: float,
        gamma: float,
        beta: float,
        h: float = 0.0,
        device: str = "cpu",
        use_fft: bool = True,  # Use FFT for better performance
    ):
        """
        Initialize the nonlocal Allen-Cahn solver.

        Args:
            spatial_resolution: Spatial resolution δ = 1/M where M is the grid size
            gamma: Interaction radius γ for the Gaussian kernel
            beta: Inverse temperature β = 1/T
            h: External magnetic field
            device: Device to run computations on ('cpu' or 'cuda')
            use_fft: Whether to use FFT for convolution (faster for large kernels)
        """
        self.delta = spatial_resolution
        self.gamma = gamma
        self.beta = beta
        self.h = h
        self.device = device
        self.use_fft = use_fft

        # Validate parameters
        if gamma <= 0:
            raise ValueError("Gamma must be positive")
        if beta <= 0:
            raise ValueError("Beta must be positive")
        if spatial_resolution <= 0:
            raise ValueError("Spatial resolution must be positive")

    def create_gaussian_kernel(self, grid_size: int) -> torch.Tensor:
        """
        Create a Gaussian Kac-type kernel for convolution.

        The kernel is: J_γ(x) = (1/(2πγ²)^(d/2)) * exp(-|x|²/(2γ²))

        In discrete form on a grid with spacing δ:
        J_γ(x_i, y_j) = (δ²/(2πγ²)) * exp(-(x_i² + y_j²)/(2γ²))

        Args:
            grid_size: Size of the spatial grid (M x M)

        Returns:
            Kernel tensor of shape (1, 1, kernel_size, kernel_size)
            for periodic convolution
        """
        # Determine kernel size based on gamma and grid spacing
        # Use 6*gamma as cutoff (covers 99.7% of Gaussian mass)
        kernel_radius = int(6 * self.gamma / self.delta) + 1
        kernel_size = 2 * kernel_radius + 1

        # Ensure kernel is not larger than grid
        kernel_size = min(kernel_size, grid_size)
        kernel_radius = kernel_size // 2

        # Create coordinate grids in physical units
        # For kernel coordinates, we use the original delta
        x = (
            torch.arange(
                -kernel_radius,
                kernel_radius + 1,
                device=self.device,
                dtype=torch.float32,
            )
            * self.delta
        )
        y = (
            torch.arange(
                -kernel_radius,
                kernel_radius + 1,
                device=self.device,
                dtype=torch.float32,
            )
            * self.delta
        )
        X, Y = torch.meshgrid(x, y, indexing="ij")

        # Distance squared in physical units
        r_squared = X**2 + Y**2

        # Gaussian kernel (2D) with proper discrete normalization
        d = 2  # dimension
        # Continuous kernel: J_γ(x) = (1/(2πγ²)^(d/2)) * exp(-|x|²/(2γ²))
        normalization = (2 * np.pi * self.gamma**2) ** (d / 2)
        kernel_continuous = torch.exp(-r_squared / (2 * self.gamma**2)) / normalization

        # For discrete grid: multiply by δ² to get proper discrete kernel
        # This ensures that the discrete kernel sum ≈ 1 (mass conservation)
        kernel = kernel_continuous * (self.delta**2)

        # Reshape for convolution (batch, channels, height, width)
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        return kernel

    def periodic_convolution(
        self, field: torch.Tensor, kernel: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform periodic convolution using FFT for better performance.

        Args:
            field: Input field of shape (batch, channels, height, width)
            kernel: Kernel of shape (1, 1, kernel_h, kernel_w)

        Returns:
            Convolved field with same shape as input
        """
        # Get dimensions
        batch_size, channels, height, width = field.shape
        kernel_h, kernel_w = kernel.shape[2], kernel.shape[3]

        # Pad the field for periodic convolution
        pad_h = kernel_h // 2
        pad_w = kernel_w // 2

        # Pad with periodic boundary conditions
        field_padded = F.pad(field, (pad_w, pad_w, pad_h, pad_h), mode="circular")

        # Perform convolution
        # Reshape for batch convolution
        field_reshaped = field_padded.view(
            batch_size * channels, 1, field_padded.shape[2], field_padded.shape[3]
        )
        kernel_reshaped = kernel.view(1, 1, kernel_h, kernel_w)

        # Convolution
        convolved = F.conv2d(field_reshaped, kernel_reshaped, padding=0)

        # Reshape back
        convolved = convolved.view(batch_size, channels, height, width)

        return convolved

    def periodic_convolution_fft(
        self, field: torch.Tensor, kernel: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform periodic convolution using FFT (faster for large kernels).

        Args:
            field: Input field of shape (batch, channels, height, width)
            kernel: Kernel of shape (1, 1, kernel_h, kernel_w)

        Returns:
            Convolved field with same shape as input
        """
        # Get dimensions
        batch_size, channels, height, width = field.shape
        kernel_h, kernel_w = kernel.shape[2], kernel.shape[3]

        # Create zero-padded kernel of same size as field
        kernel_padded = torch.zeros(
            batch_size, channels, height, width, device=field.device, dtype=field.dtype
        )

        # Place kernel in center
        start_h = (height - kernel_h) // 2
        start_w = (width - kernel_w) // 2
        end_h = start_h + kernel_h
        end_w = start_w + kernel_w

        # Ensure we don't go out of bounds
        end_h = min(end_h, height)
        end_w = min(end_w, width)
        actual_kernel_h = end_h - start_h
        actual_kernel_w = end_w - start_w

        kernel_padded[:, :, start_h:end_h, start_w:end_w] = kernel[
            :, :, :actual_kernel_h, :actual_kernel_w
        ]

        # FFT convolution
        field_fft = torch.fft.fft2(field, dim=(-2, -1))
        kernel_fft = torch.fft.fft2(kernel_padded, dim=(-2, -1))

        # Element-wise multiplication and inverse FFT
        convolved_fft = field_fft * kernel_fft
        convolved = torch.fft.ifft2(convolved_fft, dim=(-2, -1)).real

        return convolved

    def compute_rhs(self, m: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        Compute the right-hand side of the nonlocal Allen-Cahn equation.

        RHS = -m + tanh(β(J_γ * m) + βh)

        Args:
            m: Current magnetization field
            kernel: Gaussian kernel for convolution

        Returns:
            Right-hand side of the PDE
        """
        # Compute nonlocal interaction: J_γ * m
        if self.use_fft:
            nonlocal_interaction = self.periodic_convolution_fft(m, kernel)
        else:
            nonlocal_interaction = self.periodic_convolution(m, kernel)

        # Compute the argument of tanh
        argument = self.beta * nonlocal_interaction + self.beta * self.h

        # Compute RHS
        rhs = -m + torch.tanh(argument)

        return rhs

    def solve_euler_step(
        self, m: torch.Tensor, kernel: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """
        Perform one Euler time step.

        Args:
            m: Current magnetization field
            kernel: Gaussian kernel
            dt: Time step size

        Returns:
            Updated magnetization field
        """
        rhs = self.compute_rhs(m, kernel)
        return m + dt * rhs

    def solve_rk4_step(
        self, m: torch.Tensor, kernel: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """
        Perform one Runge-Kutta 4th order time step.

        Args:
            m: Current magnetization field
            kernel: Gaussian kernel
            dt: Time step size

        Returns:
            Updated magnetization field
        """
        # Compute k1
        k1 = self.compute_rhs(m, kernel)

        # Compute k2
        m_temp = m + 0.5 * dt * k1
        k2 = self.compute_rhs(m_temp, kernel)

        # Compute k3
        m_temp = m + 0.5 * dt * k2
        k3 = self.compute_rhs(m_temp, kernel)

        # Compute k4
        m_temp = m + dt * k3
        k4 = self.compute_rhs(m_temp, kernel)

        # Combine
        return m + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def solve(
        self,
        initial_field: Union[np.ndarray, torch.Tensor],
        dt: float,
        t_end: float,
        method: str = "rk4",
        save_trajectory: bool = False,
        show_progress: bool = True,
        progress_interval: int = 50,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Solve the nonlocal Allen-Cahn equation.

        Args:
            initial_field: Initial magnetization field (M x M array)
            dt: Time step size
            t_end: Total simulation time
            method: Integration method ('euler' or 'rk4')
            save_trajectory: Whether to save the full trajectory
            show_progress: Whether to show progress updates
            progress_interval: How often to show progress (every N steps)

        Returns:
            If save_trajectory=False: Final field
            If save_trajectory=True: (final_field, trajectory)
        """
        # Convert to torch tensor if needed
        if isinstance(initial_field, np.ndarray):
            m = torch.from_numpy(initial_field).float().to(self.device)
        else:
            m = initial_field.float().to(self.device)

        # Ensure correct shape: (1, 1, M, M) for convolution
        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.dim() == 3:
            m = m.unsqueeze(0)

        # Create kernel
        grid_size = m.shape[-1]  # Assuming square grid
        kernel = self.create_gaussian_kernel(grid_size)

        # Time stepping
        n_steps = int(t_end / dt)

        if show_progress:
            print(f"Starting PDE solution:")
            print(f"  Grid size: {m.shape[-1]}x{m.shape[-1]}")
            print(f"  Time steps: {n_steps}")
            print(f"  Method: {method.upper()}")
            print(f"  Device: {self.device}")
            print(f"  Progress updates every {progress_interval} steps")

        if save_trajectory:
            trajectory = []
            trajectory.append(m.squeeze().cpu().numpy())

        # Choose integration method
        if method == "euler":
            step_func = self.solve_euler_step
        elif method == "rk4":
            step_func = self.solve_rk4_step
        else:
            raise ValueError(f"Unknown method: {method}")

        # Time integration with progress tracking
        import time

        start_time = time.time()

        for step in range(n_steps):
            m = step_func(m, kernel, dt)

            if save_trajectory:
                trajectory.append(m.squeeze().cpu().numpy())

            # Show progress
            if show_progress and (step + 1) % progress_interval == 0:
                elapsed = time.time() - start_time
                progress = (step + 1) / n_steps * 100
                steps_per_sec = (step + 1) / elapsed
                eta = (n_steps - step - 1) / steps_per_sec if steps_per_sec > 0 else 0

                print(
                    f"  Step {step+1:4d}/{n_steps} ({progress:5.1f}%) - "
                    f"Elapsed: {elapsed:6.1f}s - "
                    f"Rate: {steps_per_sec:5.1f} steps/s - "
                    f"ETA: {eta:6.1f}s"
                )

        if show_progress:
            total_time = time.time() - start_time
            print(f"✅ Solution completed in {total_time:.1f} seconds")

        if save_trajectory:
            return m.squeeze(), np.array(trajectory)
        else:
            return m.squeeze()


def load_initial_condition_from_data(data_path: str) -> Tuple[np.ndarray, dict]:
    """
    Load initial condition from Glauber dynamics data.

    Args:
        data_path: Path to the .npz data file

    Returns:
        Tuple of (initial_magnetization, metadata)
    """
    import json

    # Load data
    data = np.load(data_path)
    json_path = data_path.replace(".npz", ".json")

    with open(json_path, "r") as f:
        metadata = json.load(f)

    # Get initial magnetization from coarse-grained spins
    spins_meso = data["spins_meso"]  # Shape: (time_steps, M, M)
    initial_magnetization = spins_meso[0]  # First time step

    return initial_magnetization, metadata


def create_solver_from_metadata(
    metadata: dict, device: str = "cpu", use_fft: bool = False
) -> NonlocalAllenCahnSolver:
    """
    Create a solver instance from simulation metadata.

    Args:
        metadata: Dictionary containing simulation parameters
        device: Device to run computations on
        use_fft: Whether to use FFT for convolution

    Returns:
        Configured NonlocalAllenCahnSolver instance
    """
    # Extract parameters
    M = metadata["M"]
    # For domain [0,1] x [0,1], we need delta = 1/(M-1)
    # This ensures the last grid point is at x=1
    spatial_resolution = 1.0 / (M - 1)
    try:
        gamma = metadata["R"]
    except:
        gamma = metadata["epsilon"]
    T = metadata["T"]
    beta = 1.0 / T
    h = metadata["h"]

    return NonlocalAllenCahnSolver(
        spatial_resolution=spatial_resolution,
        gamma=gamma,
        beta=beta,
        h=h,
        device=device,
        use_fft=use_fft,
    )


class LocalAllenCahnSolver:
    """
    Local Allen-Cahn solver for comparison with nonlocal version.

    Two local forms are supported via parameters:
      (A) Glauber–Kac local limit (tanh-form):
          ∂_t m = -m + tanh(β(κ ∇²m + h))
      (B) Classical Allen–Cahn (polynomial potential):
          ∂_t m = κ ∇²m - (a m^3 + b m) + h
    You can choose by setting use_tanh_rhs=True (A) or False (B).
    """

    def __init__(
        self,
        spatial_resolution: float,
        beta: float,
        h: float = 0.0,
        kappa: float = 1.0,
        a: float = 1.0,
        b: float = -1.0,
        use_tanh_rhs: bool = False,
        device: str = "cpu",
    ):
        """
        Initialize the local Allen-Cahn solver.

        Args:
            spatial_resolution: Spatial resolution δ = 1/M where M is the grid size
            beta: Inverse temperature β = 1/T
            h: External magnetic field
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.delta = spatial_resolution
        self.beta = beta
        self.h = h
        self.kappa = kappa
        self.a = a
        self.b = b
        self.use_tanh_rhs = use_tanh_rhs
        self.device = device

        # Validate parameters
        if beta <= 0:
            raise ValueError("Beta must be positive")
        if spatial_resolution <= 0:
            raise ValueError("Spatial resolution must be positive")

    def create_laplacian_kernel(self) -> torch.Tensor:
        """
        Create a discrete Laplacian kernel for 2D periodic boundary conditions.

        The discrete Laplacian is:
        ∇²f ≈ (f_{i+1,j} + f_{i-1,j} + f_{i,j+1} + f_{i,j-1} - 4f_{i,j}) / δ²

        Returns:
            Laplacian kernel tensor of shape (1, 1, 3, 3)
        """
        # Create 3x3 Laplacian kernel
        kernel = torch.zeros(1, 1, 3, 3, device=self.device, dtype=torch.float32)

        # Laplacian stencil: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        kernel[0, 0, 0, 1] = 1.0  # top
        kernel[0, 0, 1, 0] = 1.0  # left
        kernel[0, 0, 1, 1] = -4.0  # center
        kernel[0, 0, 1, 2] = 1.0  # right
        kernel[0, 0, 2, 1] = 1.0  # bottom

        # Scale by 1/δ²
        kernel = kernel / (self.delta**2)

        return kernel

    def compute_rhs(
        self, m: torch.Tensor, laplacian_kernel: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the right-hand side of the local Allen-Cahn equation.

        Two options:
          (A) use_tanh_rhs=True:  RHS = -m + tanh(β(κ ∇²m + h))
          (B) use_tanh_rhs=False: RHS = κ ∇²m - (a m^3 + b m) + h

        Args:
            m: Current magnetization field
            laplacian_kernel: Laplacian kernel for convolution

        Returns:
            Right-hand side of the PDE
        """
        # Compute Laplacian: ∇²m
        # Use circular padding for periodic boundary conditions
        m_padded = F.pad(m, (1, 1, 1, 1), mode="circular")
        laplacian = F.conv2d(m_padded, laplacian_kernel, padding=0)

        if self.use_tanh_rhs:
            argument = self.beta * (self.kappa * laplacian + self.h)
            rhs = -m + torch.tanh(argument)
        else:
            rhs = self.kappa * laplacian - (self.a * m**3 + self.b * m) + self.h

        return rhs

    def solve_rk4_step(
        self, m: torch.Tensor, laplacian_kernel: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """
        Perform one Runge-Kutta 4th order time step.

        Args:
            m: Current magnetization field
            laplacian_kernel: Laplacian kernel
            dt: Time step size

        Returns:
            Updated magnetization field
        """
        # Compute k1
        k1 = self.compute_rhs(m, laplacian_kernel)

        # Compute k2
        m_temp = m + 0.5 * dt * k1
        k2 = self.compute_rhs(m_temp, laplacian_kernel)

        # Compute k3
        m_temp = m + 0.5 * dt * k2
        k3 = self.compute_rhs(m_temp, laplacian_kernel)

        # Compute k4
        m_temp = m + dt * k3
        k4 = self.compute_rhs(m_temp, laplacian_kernel)

        # Combine
        return m + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def solve(
        self,
        initial_field: Union[np.ndarray, torch.Tensor],
        dt: float,
        t_end: float,
        save_trajectory: bool = False,
        show_progress: bool = True,
        progress_interval: int = 50,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Solve the local Allen-Cahn equation.

        Args:
            initial_field: Initial magnetization field (M x M array)
            dt: Time step size
            t_end: Total simulation time
            save_trajectory: Whether to save the full trajectory
            show_progress: Whether to show progress updates
            progress_interval: How often to show progress (every N steps)

        Returns:
            If save_trajectory=False: Final field
            If save_trajectory=True: (final_field, trajectory)
        """
        # Convert to torch tensor if needed
        if isinstance(initial_field, np.ndarray):
            m = torch.from_numpy(initial_field).float().to(self.device)
        else:
            m = initial_field.float().to(self.device)

        # Ensure correct shape: (1, 1, M, M) for convolution
        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.dim() == 3:
            m = m.unsqueeze(0)

        # Create Laplacian kernel
        laplacian_kernel = self.create_laplacian_kernel()

        # Time stepping
        n_steps = int(t_end / dt)

        if show_progress:
            print(f"Starting Local PDE solution:")
            print(f"  Grid size: {m.shape[-1]}x{m.shape[-1]}")
            print(f"  Time steps: {n_steps}")
            print(f"  Method: RK4")
            print(f"  Device: {self.device}")
            print(f"  Progress updates every {progress_interval} steps")

        if save_trajectory:
            trajectory = []
            trajectory.append(m.squeeze().cpu().numpy())

        # Time integration with progress tracking
        import time

        start_time = time.time()

        for step in range(n_steps):
            m = self.solve_rk4_step(m, laplacian_kernel, dt)

            if save_trajectory:
                trajectory.append(m.squeeze().cpu().numpy())

            # Show progress
            if show_progress and (step + 1) % progress_interval == 0:
                elapsed = time.time() - start_time
                progress = (step + 1) / n_steps * 100
                steps_per_sec = (step + 1) / elapsed
                eta = (n_steps - step - 1) / steps_per_sec if steps_per_sec > 0 else 0

                print(
                    f"  Step {step+1:4d}/{n_steps} ({progress:5.1f}%) - "
                    f"Elapsed: {elapsed:6.1f}s - "
                    f"Rate: {steps_per_sec:5.1f} steps/s - "
                    f"ETA: {eta:6.1f}s"
                )

        if show_progress:
            total_time = time.time() - start_time
            print(f"✅ Local solution completed in {total_time:.1f} seconds")

        if save_trajectory:
            return m.squeeze(), np.array(trajectory)
        else:
            return m.squeeze()


def create_local_solver_from_metadata(
    metadata: dict, device: str = "cpu"
) -> LocalAllenCahnSolver:
    """
    Create a local solver instance from simulation metadata.

    Args:
        metadata: Dictionary containing simulation parameters
        device: Device to run computations on

    Returns:
        Configured LocalAllenCahnSolver instance
    """
    # Extract parameters
    M = metadata["M"]
    # For domain [0,1] x [0,1], we need delta = 1/(M-1)
    # This ensures the last grid point is at x=1
    spatial_resolution = 1.0 / (M - 1)
    T = metadata["T"]
    beta = 1.0 / T
    h = metadata["h"]

    return LocalAllenCahnSolver(
        spatial_resolution=spatial_resolution,
        beta=beta,
        h=h,
        kappa=1.0,
        a=1.0,
        b=-1.0,
        use_tanh_rhs=False,
        device=device,
    )


def create_local_ac_solver_from_metadata(
    metadata: dict,
    *,
    device: str = "cpu",
    kappa: float | None = None,
    a: float | None = None,
    b: float | None = None,
    use_tanh_rhs: bool | None = None,
) -> LocalAllenCahnSolver:
    """
    Factory with convenient overrides for LocalAllenCahnSolver.

    Args:
        metadata: simulation metadata (contains M, T, h)
        device: torch device
        kappa, a, b, use_tanh_rhs: optional overrides
    """
    M = metadata["M"]
    spatial_resolution = 1.0 / (M - 1)
    T = metadata["T"]
    beta = 1.0 / T
    h = metadata["h"]

    solver = LocalAllenCahnSolver(
        spatial_resolution=spatial_resolution,
        beta=beta,
        h=h,
        kappa=kappa if kappa is not None else 1.0,
        a=a if a is not None else 1.0,
        b=b if b is not None else -1.0,
        use_tanh_rhs=use_tanh_rhs if use_tanh_rhs is not None else False,
        device=device,
    )
    return solver
