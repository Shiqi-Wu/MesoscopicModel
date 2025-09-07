#!/usr/bin/env python3
"""
Visualize Gaussian kernels in [0,1] domain with different gamma and kernel sizes.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append("src")
from networks import GaussianKernel


def create_gaussian_kernel_01_domain(kernel_size, gamma, M=128, center=(0.5, 0.5)):
    """
    Create Gaussian kernel in [0,1] domain based on grid cell centers.

    Args:
        kernel_size: Size of kernel (e.g., 5, 7, 9)
        gamma: Gaussian width parameter
        M: Grid size (M x M)
        center: Center point in [0,1] domain

    Returns:
        full_kernel: (M, M) array with kernel weights
    """
    # Create coordinate grids for all grid cells
    ys, xs = torch.meshgrid(
        torch.arange(M, dtype=torch.float32),
        torch.arange(M, dtype=torch.float32),
        indexing="ij",
    )

    # Convert grid indices to physical coordinates (cell centers)
    # Cell (i,j) has center at ((i+0.5)/M, (j+0.5)/M)
    coords_x = (xs + 0.5) / M
    coords_y = (ys + 0.5) / M

    # Calculate distances from center in physical coordinates
    center_x_phys = center[0]
    center_y_phys = center[1]

    distances_squared = (coords_x - center_x_phys) ** 2 + (
        coords_y - center_y_phys
    ) ** 2

    # Calculate Gaussian weights
    gaussian_weights = torch.exp(-distances_squared / (2.0 * gamma**2))

    # Apply kernel size constraint: only keep weights within kernel_size neighborhood
    # For center (0.5, 0.5) on M=128 grid, the exact center is at grid (63.5, 63.5)
    # We need to find the kernel_size x kernel_size neighborhood around this center

    # Calculate the exact center in grid coordinates
    center_x_grid = center[0] * M - 0.5  # This gives us the exact center
    center_y_grid = center[1] * M - 0.5

    k = (kernel_size - 1) // 2

    # Calculate symmetric bounds around the exact center
    start_x = max(0, int(center_x_grid - k + 0.5))  # Add 0.5 for proper rounding
    end_x = min(M, int(center_x_grid + k + 0.5))
    start_y = max(0, int(center_y_grid - k + 0.5))
    end_y = min(M, int(center_y_grid + k + 0.5))

    # Ensure we have exactly kernel_size x kernel_size if possible
    actual_size_x = end_x - start_x
    actual_size_y = end_y - start_y

    # Adjust if we're at the boundary
    if actual_size_x < kernel_size:
        if start_x == 0:
            end_x = min(M, start_x + kernel_size)
        else:
            start_x = max(0, end_x - kernel_size)

    if actual_size_y < kernel_size:
        if start_y == 0:
            end_y = min(M, start_y + kernel_size)
        else:
            start_y = max(0, end_y - kernel_size)

    # Create mask for kernel neighborhood
    mask = torch.zeros(M, M, dtype=torch.bool)
    mask[start_y:end_y, start_x:end_x] = True

    # Apply mask and normalize
    masked_weights = gaussian_weights * mask.float()
    if masked_weights.sum() > 0:
        normalized_weights = masked_weights / masked_weights.sum()
    else:
        normalized_weights = masked_weights

    return normalized_weights.numpy()


def visualize_kernels_01_domain():
    """Visualize Gaussian kernels with different parameters in [0,1] domain."""
    print("🔍 Visualizing Gaussian Kernels in [0,1] Domain")
    print("=" * 60)

    # Parameters
    M = 128  # Grid size
    center = (0.5, 0.5)  # Fixed center point

    # Different kernel sizes and gamma values
    kernel_sizes = [5, 11, 21, 41]
    gamma_values = [i / M for i in range(1, 5)]

    print(f"Grid size: {M}×{M}")
    print(f"Center point: {center}")
    print(f"Kernel sizes: {kernel_sizes}")
    print(f"Gamma values: {gamma_values}")
    print()

    # Create subplot grid
    fig, axes = plt.subplots(len(gamma_values), len(kernel_sizes), figsize=(16, 12))

    for i, gamma in enumerate(gamma_values):
        for j, kernel_size in enumerate(kernel_sizes):
            # Create kernel
            full_kernel = create_gaussian_kernel_01_domain(
                kernel_size, gamma, M, center
            )

            # Plot on full [0,1] domain
            # Each pixel represents a grid cell, so extent should be [0,1] x [0,1]
            im = axes[i, j].imshow(
                full_kernel, cmap="viridis", extent=[0, 1, 0, 1], origin="lower"
            )
            axes[i, j].set_title(f"γ={gamma:.4f}, K={kernel_size}×{kernel_size}")
            axes[i, j].set_xlabel("x")
            axes[i, j].set_ylabel("y")

            # Add center point
            axes[i, j].plot(
                center[0],
                center[1],
                "r*",
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=1,
            )

            # Add colorbar
            plt.colorbar(im, ax=axes[i, j], fraction=0.046)

            # Print statistics
            non_zero_mask = full_kernel > 0
            non_zero_values = full_kernel[non_zero_mask]
            print(
                f"γ={gamma:.2f}, K={kernel_size}: "
                f"Max={full_kernel.max():.4f}, "
                f"Min(non-zero)={non_zero_values.min():.6f}, "
                f"Sum={full_kernel.sum():.6f}, "
                f"Non-zero points={non_zero_mask.sum()}"
            )

    plt.suptitle(
        "Gaussian Kernels in [0,1] Domain\n" f"Grid: {M}×{M}, Center: {center}",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig("results/gaussian_kernels_01_domain.png", dpi=150, bbox_inches="tight")
    print(f"\n📁 Visualization saved to: results/gaussian_kernels_01_domain.png")
    plt.show()


def show_kernel_properties():
    """Show properties of kernels in [0,1] domain."""
    print("\n📊 Kernel Properties in [0,1] Domain:")
    print("=" * 60)

    M = 128
    kernel_size = 7
    gamma_values = [0.01, 0.05, 0.1, 0.2, 0.5]

    print(f"Kernel size: {kernel_size}×{kernel_size}")
    print(f"Grid size: {M}×{M}")
    print()

    for gamma in gamma_values:
        kernel_weights = create_gaussian_kernel_01_domain(kernel_size, gamma, M)

        # Calculate effective radius (where weight drops to 1% of max)
        max_weight = kernel_weights.max()
        threshold = 0.01 * max_weight

        # Find effective radius
        center_idx = kernel_size // 2
        effective_radius = 0
        for r in range(center_idx + 1):
            for i in range(kernel_size):
                for j in range(kernel_size):
                    dist_from_center = np.sqrt(
                        (i - center_idx) ** 2 + (j - center_idx) ** 2
                    )
                    if dist_from_center <= r and kernel_weights[i, j] >= threshold:
                        effective_radius = max(effective_radius, r)

        # Convert to physical units
        effective_radius_phys = effective_radius / M

        print(f"γ = {gamma:.2f}:")
        print(f"  Max weight: {max_weight:.6f}")
        print(f"  Min weight: {kernel_weights.min():.8f}")
        print(f"  Effective radius: {effective_radius_phys:.4f} (physical units)")
        print(f"  Effective radius: {effective_radius} (grid units)")
        print()


def demonstrate_coordinate_scaling():
    """Demonstrate the coordinate scaling concept."""
    print("\n🔧 Coordinate Scaling Concept:")
    print("=" * 60)

    M = 128
    kernel_size = 5
    gamma = 0.1

    print(f"For M = {M} grid with {kernel_size}×{kernel_size} kernel:")
    print()

    # Show coordinate conversion
    k = (kernel_size - 1) // 2
    print("Grid coordinates → Physical coordinates:")
    print("-" * 40)

    for i in range(kernel_size):
        for j in range(kernel_size):
            grid_x, grid_y = j, i
            phys_x = grid_x / M
            phys_y = grid_y / M
            print(f"({grid_x:2d}, {grid_y:2d}) → ({phys_x:.4f}, {phys_y:.4f})")

    print(f"\nKey insight:")
    print(f"- Kernel operates in physical [0,1] domain")
    print(f"- Each grid point represents 1/{M} = {1/M:.4f} physical units")
    print(
        f"- Kernel size {kernel_size}×{kernel_size} covers {kernel_size/M:.4f}×{kernel_size/M:.4f} physical area"
    )


if __name__ == "__main__":
    import os

    os.makedirs("results", exist_ok=True)

    visualize_kernels_01_domain()
    show_kernel_properties()
    demonstrate_coordinate_scaling()
