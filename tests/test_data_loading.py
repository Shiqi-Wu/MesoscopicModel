#!/usr/bin/env python3
"""
Test script to load and examine data from the data/ directory
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.append("src")
from plot import load_spins_data
from train import IsingDataset


def load_single_data_file():
    """Load a single data file and examine its contents"""
    print("=" * 60)
    print("Loading single data file")
    print("=" * 60)

    # Find the first data file
    data_dir = Path("data")
    data_files = list(data_dir.glob("**/*.npz"))

    if not data_files:
        print("❌ No .npz files found in data/ directory")
        return

    # Use the first file
    npz_file = data_files[0]
    json_file = npz_file.with_suffix(".json")

    print(f"📁 Loading: {npz_file}")
    print(f"📁 Metadata: {json_file}")

    # Load data
    try:
        data = np.load(npz_file)
        with open(json_file, "r") as f:
            metadata = json.load(f)

        print(f"\n✅ Successfully loaded data!")
        print(f"📊 Data keys: {list(data.keys())}")
        print(f"📊 Metadata keys: {list(metadata.keys())}")

        # Print data shapes
        for key in data.keys():
            print(f"  {key}: {data[key].shape}")

        # Print metadata
        print(f"\n📋 Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

        return data, metadata

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None


def test_ising_dataset():
    """Test the IsingDataset class"""
    print("\n" + "=" * 60)
    print("Testing IsingDataset class")
    print("=" * 60)

    try:
        # Find data files
        data_dir = Path("data")
        data_files = list(data_dir.glob("**/*.npz"))

        if not data_files:
            print("❌ No .npz files found")
            return

        # Use the first file (without .npz extension)
        file_path = str(data_files[0].with_suffix(""))

        print(f"📁 Creating dataset with: {file_path}")

        # Create dataset
        dataset = IsingDataset(file_path=file_path)

        print(f"✅ Dataset created successfully!")
        print(f"📊 Number of data items: {len(dataset)}")

        # Get sample data
        if len(dataset) > 0:
            sample_data = dataset.get_sample_data(0)
            t, x, m, dmdt = sample_data

            print(f"\n📊 Sample data shapes:")
            print(f"  t (times): {t.shape}")
            print(f"  x (spins): {x.shape}")
            print(f"  m (magnetization): {m.shape}")
            print(f"  dmdt (dm/dt): {dmdt.shape}")

            # Print some statistics
            print(f"\n📈 Data statistics:")
            print(f"  Time range: {t.min().item():.3f} to {t.max().item():.3f}")
            print(
                f"  Magnetization (m) range: {m.min().item():.3f} to {m.max().item():.3f}"
            )
            print(f"  dm/dt range: {dmdt.min().item():.3f} to {dmdt.max().item():.3f}")

            return dataset, sample_data

    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def analyze_data(spins, times, metadata):
    """Analyze the loaded spins data without plotting"""
    print("\n" + "=" * 60)
    print("Analyzing data")
    print("=" * 60)

    try:
        print(f"📊 Data analysis:")
        print(f"  Time steps: {len(times)}")
        print(f"  Spatial dimensions: {spins.shape[1]} x {spins.shape[2]}")
        print(f"  Time range: {times[0]:.3f} to {times[-1]:.3f}")
        print(f"  Time step: {times[1] - times[0]:.6f}")

        # Analyze spins (convert to torch for analysis)
        spins_torch = torch.from_numpy(spins).float()

        # Calculate magnetization (mean over spatial dimensions)
        m_mean = spins_torch.mean(dim=(1, 2))
        print(
            f"  Spins range: {spins_torch.min().item():.3f} to {spins_torch.max().item():.3f}"
        )
        print(
            f"  Magnetization range: {m_mean.min().item():.3f} to {m_mean.max().item():.3f}"
        )
        print(f"  Average magnetization: {m_mean.mean().item():.3f}")
        print(f"  Magnetization std: {m_mean.std().item():.3f}")

        # Calculate dm/dt (time derivative of magnetization)
        dm_dt = torch.diff(m_mean)
        print(f"  dm/dt range: {dm_dt.min().item():.3f} to {dm_dt.max().item():.3f}")
        print(f"  Average dm/dt: {dm_dt.mean().item():.3f}")
        print(f"  dm/dt std: {dm_dt.std().item():.3f}")

        # Check for any NaN or Inf values
        has_nan = torch.isnan(spins_torch).any() or torch.isnan(dm_dt).any()
        has_inf = torch.isinf(spins_torch).any() or torch.isinf(dm_dt).any()
        print(f"  Contains NaN: {has_nan}")
        print(f"  Contains Inf: {has_inf}")

        # Additional analysis
        print(f"  Lattice size: {metadata.get('L', 'N/A')}")
        print(f"  Block size: {metadata.get('B', 'N/A')}")
        print(f"  Temperature: {metadata.get('T', 'N/A'):.3f}")
        print(f"  External field: {metadata.get('h', 'N/A'):.3f}")

        return True

    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function to test data loading"""
    print("🧪 Testing data loading from data/ directory")
    FILE_PATH = "data/ct_glauber_L1024_ell32_sigma1.2_tau1_m00.1_Tfrac3_J1_h0_tend5.0_dt0.01_block8_kernelnearest_R0.015625_seed0/round0/ct_glauber_L1024_ell32_sigma1.2_tau1_m00.1_Tfrac3_J1_h0_tend5.0_dt0.01_block8_kernelnearest_R0.015625_seed0_round0.npz"

    # Test 1: Load single file
    # data, metadata = load_single_data_file()
    spins, times, metadata = load_spins_data(FILE_PATH)

    print(metadata.keys())
    print(metadata["B"], metadata["M"], metadata["L"])
    assert metadata["B"] * metadata["M"] == metadata["L"], "B * M != L"
    print(metadata["J"], metadata["h"], metadata["T"])
    print(metadata["snapshot_dt"])

    # Test 2: Create spins animation
    # plot spins with plot func from plot.py
    from plot import create_spins_animation

    print("\n🎬 Creating spins animation...")
    create_spins_animation(
        spins=spins,
        times=times,
        metadata=metadata,
        output_path="results/test_animation.html",
        max_frames=20,
    )
    print("✅ Animation created successfully!")

    # Test 3: Analyze data
    analyze_data(spins, times, metadata)

    # Test 4: Test IsingDataset class
    # dataset, sample_data = test_ising_dataset()

    dataset = IsingDataset(file_path=FILE_PATH)
    print(f"✅ Dataset created successfully!")
    print(f"📊 Number of data items: {len(dataset)}")

    if len(dataset) > 0:
        sample_data = dataset.get_sample_data(0)
        t, x, m, dmdt = sample_data

        print(f"\n📊 Sample data shapes:")
        print(f"  t (times): {t.shape}")
        print(f"  x (spins): {x.shape}")
        print(f"  m (magnetization): {m.shape}")
        print(f"  dmdt (dm/dt): {dmdt.shape}")

    print("\n" + "=" * 60)
    print("✅ Data loading test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
