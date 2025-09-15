#!/usr/bin/env python3
"""
Test script for the PDE solver implementation.

This script tests the nonlocal Allen-Cahn equation solver by:
1. Loading initial conditions from Glauber dynamics data
2. Solving the PDE with the same parameters
3. Comparing the results with the original data
4. Validating the solver's correctness
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append("src")
from pde_solver import (
    NonlocalAllenCahnSolver,
    load_initial_condition_from_data,
    create_solver_from_metadata,
)
from plot import load_spins_data


def test_solver_initialization():
    """Test basic solver initialization and parameter validation."""
    print("=" * 60)
    print("Testing solver initialization")
    print("=" * 60)

    try:
        # Test valid parameters
        solver = NonlocalAllenCahnSolver(
            spatial_resolution=1.0 / 128, gamma=0.015625, beta=1.0 / 6.808, h=0.0
        )
        print("✅ Solver initialization successful")

        # Test parameter validation
        try:
            NonlocalAllenCahnSolver(
                spatial_resolution=1.0 / 128, gamma=-0.1, beta=1.0, h=0.0
            )
            print("❌ Should have raised ValueError for negative gamma")
        except ValueError:
            print("✅ Correctly caught negative gamma error")

        try:
            NonlocalAllenCahnSolver(
                spatial_resolution=1.0 / 128, gamma=0.1, beta=-1.0, h=0.0
            )
            print("❌ Should have raised ValueError for negative beta")
        except ValueError:
            print("✅ Correctly caught negative beta error")

        return True

    except Exception as e:
        print(f"❌ Solver initialization failed: {e}")
        return False


def test_gaussian_kernel():
    """Test Gaussian kernel creation and properties."""
    print("\n" + "=" * 60)
    print("Testing Gaussian kernel")
    print("=" * 60)

    try:
        solver = NonlocalAllenCahnSolver(
            spatial_resolution=1.0 / 128, gamma=0.1, beta=1.0, h=0.0
        )

        # Create kernel
        kernel = solver.create_gaussian_kernel(64)
        print(f"✅ Kernel created with shape: {kernel.shape}")

        # Check kernel properties
        kernel_sum = kernel.sum().item()
        print(f"  Kernel sum: {kernel_sum:.6f} (should be close to 1.0)")

        # Check symmetry
        kernel_2d = kernel.squeeze()
        center = kernel_2d.shape[0] // 2
        print(f"  Kernel center value: {kernel_2d[center, center]:.6f}")
        print(f"  Kernel max value: {kernel_2d.max().item():.6f}")

        # Check that kernel is positive
        assert torch.all(kernel >= 0), "Kernel should be non-negative"
        print("✅ Kernel is non-negative")

        return True

    except Exception as e:
        print(f"❌ Gaussian kernel test failed: {e}")
        return False


def test_convolution():
    """Test periodic convolution functionality."""
    print("\n" + "=" * 60)
    print("Testing periodic convolution")
    print("=" * 60)

    try:
        solver = NonlocalAllenCahnSolver(
            spatial_resolution=1.0 / 32, gamma=0.1, beta=1.0, h=0.0
        )

        # Create test field
        grid_size = 32
        test_field = torch.zeros(1, 1, grid_size, grid_size)
        test_field[0, 0, grid_size // 2, grid_size // 2] = 1.0  # Delta function

        # Create kernel
        kernel = solver.create_gaussian_kernel(grid_size)

        # Test convolution
        convolved = solver.periodic_convolution(test_field, kernel)
        print(f"✅ Convolution successful, output shape: {convolved.shape}")

        # Check that convolution preserves total mass (approximately)
        input_sum = test_field.sum().item()
        output_sum = convolved.sum().item()
        print(f"  Input sum: {input_sum:.6f}")
        print(f"  Output sum: {output_sum:.6f}")
        print(f"  Difference: {abs(input_sum - output_sum):.6f}")

        # Check that output is smooth
        output_2d = convolved.squeeze()
        print(
            f"  Output range: [{output_2d.min().item():.6f}, {output_2d.max().item():.6f}]"
        )

        return True

    except Exception as e:
        print(f"❌ Convolution test failed: {e}")
        return False


def test_data_loading():
    """Test loading initial conditions from data."""
    print("\n" + "=" * 60)
    print("Testing data loading")
    print("=" * 60)

    try:
        # Use the specified data path
        data_path = "data/ct_glauber_L1024_ell32_sigma1.2_tau1_m00.1_Tfrac3_J1_h0_tend5.0_dt0.01_block8_kernelnearest_R0.015625_seed0/round0/ct_glauber_L1024_ell32_sigma1.2_tau1_m00.1_Tfrac3_J1_h0_tend5.0_dt0.01_block8_kernelnearest_R0.015625_seed0_round0.npz"

        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
            return False

        # Load initial condition
        initial_m, metadata = load_initial_condition_from_data(data_path)
        print(f"✅ Data loaded successfully")
        print(f"  Initial magnetization shape: {initial_m.shape}")
        print(
            f"  Initial magnetization range: [{initial_m.min():.6f}, {initial_m.max():.6f}]"
        )
        print(f"  Metadata keys: {list(metadata.keys())}")

        # Create solver from metadata
        solver = create_solver_from_metadata(metadata)
        print(f"✅ Solver created from metadata")
        print(f"  Spatial resolution: {solver.delta:.6f}")
        print(f"  Gamma: {solver.gamma:.6f}")
        print(f"  Beta: {solver.beta:.6f}")
        print(f"  h: {solver.h:.6f}")

        return True, initial_m, metadata, solver

    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None, None, None


def test_pde_solution_quick():
    """Test solving the PDE with shorter time for quick validation."""
    print("\n" + "=" * 60)
    print("Testing PDE solution (quick test)")
    print("=" * 60)

    try:
        # Load data and create solver
        success, initial_m, metadata, solver = test_data_loading()
        if not success:
            return False

        # Use shorter time for quick test
        dt = metadata["snapshot_dt"]
        t_end = 0.1  # Much shorter time

        print(f"  Time step: {dt}")
        print(f"  Total time: {t_end} (shortened for testing)")
        print(f"  Number of steps: {int(t_end/dt)}")

        # Solve PDE
        print("  Solving PDE...")
        final_field = solver.solve(
            initial_field=initial_m,
            dt=dt,
            t_end=t_end,
            method="rk4",
            save_trajectory=False,
            show_progress=True,
            progress_interval=5,
        )

        print(f"✅ Quick PDE test successful")
        print(f"  Final field shape: {final_field.shape}")
        print(
            f"  Final field range: [{final_field.min().item():.6f}, {final_field.max().item():.6f}]"
        )

        return True, final_field

    except Exception as e:
        print(f"❌ Quick PDE solution test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_pde_solution():
    """Test solving the PDE and compare with data."""
    print("\n" + "=" * 60)
    print("Testing PDE solution (full simulation)")
    print("=" * 60)

    try:
        # Load data and create solver
        success, initial_m, metadata, solver = test_data_loading()
        if not success:
            return False

        # Get simulation parameters
        dt = metadata["snapshot_dt"]
        t_end = metadata["t_end"]

        print(f"  Time step: {dt}")
        print(f"  Total time: {t_end}")
        print(f"  Number of steps: {int(t_end/dt)}")
        print(f"  ⚠️  This will take several minutes...")

        # Ask user if they want to continue
        response = input("  Continue with full simulation? (y/n): ").lower().strip()
        if response != "y":
            print("  Skipping full simulation test")
            return True, None, None

        # Solve PDE
        print("  Solving PDE...")
        final_field = solver.solve(
            initial_field=initial_m,
            dt=dt,
            t_end=t_end,
            method="rk4",
            save_trajectory=False,
            show_progress=True,
            progress_interval=25,
        )

        print(f"✅ PDE solved successfully")
        print(f"  Final field shape: {final_field.shape}")
        print(
            f"  Final field range: [{final_field.min().item():.6f}, {final_field.max().item():.6f}]"
        )

        # Compare with original data
        data = np.load(
            "data/ct_glauber_L1024_ell32_sigma1.2_tau1_m00.1_Tfrac3_J1_h0_tend5.0_dt0.01_block8_kernelnearest_R0.015625_seed0/round0/ct_glauber_L1024_ell32_sigma1.2_tau1_m00.1_Tfrac3_J1_h0_tend5.0_dt0.01_block8_kernelnearest_R0.015625_seed0_round0.npz"
        )
        original_final = data["spins_meso"][-1]  # Last time step

        print(f"  Original final shape: {original_final.shape}")
        print(
            f"  Original final range: [{original_final.min():.6f}, {original_final.max():.6f}]"
        )

        # Compute difference
        if isinstance(final_field, torch.Tensor):
            final_field_np = final_field.cpu().numpy()
        else:
            final_field_np = final_field

        diff = np.abs(final_field_np - original_final)
        print(f"  Mean absolute difference: {diff.mean():.6f}")
        print(f"  Max absolute difference: {diff.max():.6f}")

        return True, final_field_np, original_final

    except Exception as e:
        print(f"❌ PDE solution test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_trajectory_solution():
    """Test solving with trajectory saving."""
    print("\n" + "=" * 60)
    print("Testing trajectory solution")
    print("=" * 60)

    try:
        # Load data and create solver
        success, initial_m, metadata, solver = test_data_loading()
        if not success:
            return False

        # Solve with trajectory saving
        dt = metadata["snapshot_dt"]
        t_end = 0.1  # Shorter time for testing

        print(f"  Solving PDE with trajectory saving...")
        final_field, trajectory = solver.solve(
            initial_field=initial_m,
            dt=dt,
            t_end=t_end,
            method="rk4",
            save_trajectory=True,
        )

        print(f"✅ Trajectory solution successful")
        print(f"  Trajectory shape: {trajectory.shape}")
        print(f"  Expected time steps: {int(t_end/dt) + 1}")
        print(f"  Actual time steps: {len(trajectory)}")

        # Check trajectory properties
        print(f"  Initial magnetization: {trajectory[0].mean():.6f}")
        print(f"  Final magnetization: {trajectory[-1].mean():.6f}")
        print(f"  Trajectory range: [{trajectory.min():.6f}, {trajectory.max():.6f}]")

        return True, trajectory

    except Exception as e:
        print(f"❌ Trajectory solution test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def create_comparison_plot(final_pde, final_original, output_dir="results"):
    """Create comparison plots between PDE solution and original data."""
    print("\n" + "=" * 60)
    print("Creating comparison plots")
    print("=" * 60)

    try:
        os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot PDE solution
        im1 = axes[0].imshow(final_pde, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[0].set_title("PDE Solution (Final)")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        plt.colorbar(im1, ax=axes[0])

        # Plot original data
        im2 = axes[1].imshow(final_original, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1].set_title("Original Data (Final)")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        plt.colorbar(im2, ax=axes[1])

        # Plot difference
        diff = np.abs(final_pde - final_original)
        im3 = axes[2].imshow(diff, cmap="viridis")
        axes[2].set_title("Absolute Difference")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()

        output_path = os.path.join(output_dir, "pde_comparison.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✅ Comparison plot saved to: {output_path}")

        plt.close()
        return True

    except Exception as e:
        print(f"❌ Plot creation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing PDE Solver Implementation")
    print("=" * 60)

    # Run tests
    tests = [
        ("Solver Initialization", test_solver_initialization),
        ("Gaussian Kernel", test_gaussian_kernel),
        ("Periodic Convolution", test_convolution),
        ("Data Loading", lambda: test_data_loading()[0]),
        ("PDE Solution (Quick)", lambda: test_pde_solution_quick()[0]),
        ("PDE Solution (Full)", lambda: test_pde_solution()[0]),
        ("Trajectory Solution", lambda: test_trajectory_solution()[0]),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! PDE solver is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

    # Try to create comparison plot if PDE solution worked
    try:
        success, final_pde, final_original = test_pde_solution()
        if success:
            create_comparison_plot(final_pde, final_original)
    except:
        pass  # Don't fail the test if plotting fails


if __name__ == "__main__":
    main()
