import os
import sys
from pathlib import Path

# Add src to path
sys.path.append("src")
from plot import create_spins_animation_from_data


def main():
    print("🎬 Testing optimized spins animation creation")

    # Find the first data file
    data_dir = Path("data")
    data_files = list(data_dir.glob("**/*.npz"))

    if not data_files:
        print("❌ No .npz files found in data/ directory")
        return

    data_file = str(data_files[0])
    print(f"📁 Using data file: {data_file}")

    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Create high quality animation
    output_path = "results/spins_animation.html"
    print(f"🎬 Creating high quality animation: {output_path}")

    try:
        create_spins_animation_from_data(
            data_file=data_file,
            output_path=output_path,
            max_frames=50,
        )
        print(f"✅ High quality animation created successfully!")

        # Check file size
        file_size = Path(output_path).stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        print(f"📊 File size: {file_size_mb:.2f} MB")
        print(f"🌐 Open {output_path} in your browser to view the animation")

    except Exception as e:
        print(f"❌ Error creating animation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
