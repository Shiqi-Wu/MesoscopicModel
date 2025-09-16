import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from networks import (
    ForceNet,
    ForceTanhNet,
    LocalKernelPointEvalPeriodic,
    ScaledTanh,
    SoftmaxKernel,
)
from plot import visualize_network
from utils import print_metadata_info


class IsingDataset(Dataset):
    """
    Dataset for loading Ising model simulation data
    File format: .npz with metadata in .json file
    """

    def __init__(self, file_path: str = ""):
        """
        Args:
            file_path: Path to the .npz file
        """
        self.file_path = file_path

        # Load all data
        self.data_list = []
        self._load_all_data()

    def _load_all_data(self):
        """Load all .npz files and extract t, x, m, dm"""

        try:
            data = np.load(self.file_path, allow_pickle=True)
            meta_data = json.load(open(self.file_path.replace(".npz", ".json")))

            # Extract time
            t = data["times"]

            # Prefer coarse data if available, otherwise coarse-grain spins
            m_fields = data.get("spins_meso", None)
            x = data.get("spins", None)

            # Print basic availability for debugging
            print(f"keys: {list(data.keys())}")
            if isinstance(x, np.ndarray):
                try:
                    print(f"spins shape: {x.shape}")
                except Exception:
                    pass
            if isinstance(m_fields, np.ndarray):
                print(f"spins_meso shape: {m_fields.shape}")
            print(f"times shape: {t.shape}")

            if m_fields is None:
                if x is None or x.size == 0:
                    raise KeyError("No spins_meso or valid spins in file.")
                # Coarse-grain spins to (T, M, M)
                Tn = t.shape[0]
                M = int(meta_data["M"])  # coarse grid size
                B = int(meta_data["B"])  # block size
                # Expect micro spins as (T, L, L) or flattenable
                if x.ndim == 3:
                    L = x.shape[1]
                    assert L == int(
                        meta_data["L"]
                    ), "L mismatch between data and metadata"
                    m_fields = x.reshape(Tn, M, B, M, B).mean(axis=(2, 4))
                else:
                    # Try to infer L from metadata and reshape
                    L = int(meta_data["L"])
                    x = x.reshape(Tn, L, L)
                    m_fields = x.reshape(Tn, M, B, M, B).mean(axis=(2, 4))

            # Compute dmdt over time with proper dt
            dt = meta_data.get("snapshot_dt", (t[1] - t[0]))
            dmdt = np.gradient(m_fields, dt, axis=0)

            # Validate data exists
            if any(arr is None for arr in [t, m_fields, dmdt]):
                print(f"Warning: Missing required data in {self.file_path}")
                print(f"Available keys: {list(data.keys())}")

            # Convert to tensors
            t_tensor = torch.from_numpy(t).float()
            # x is optional; store coarse field as m
            m_tensor = torch.from_numpy(m_fields).float()
            # if raw spins exist and are dense, keep a small placeholder to avoid memory blow-up
            if isinstance(x, np.ndarray) and x.ndim >= 1 and x.size > 0:
                try:
                    x_tensor = torch.from_numpy(x).float()
                except Exception:
                    x_tensor = torch.empty(0)
            else:
                x_tensor = torch.empty(0)
            dmdt_tensor = torch.from_numpy(dmdt).float()

            # Store data info
            data_info = {
                "file_path": self.file_path,
                "t": t_tensor,
                "x": x_tensor,
                "m": m_tensor,
                "dmdt": dmdt_tensor,
                "shape": m_tensor.shape,
            }

            print(f"Loaded {os.path.basename(self.file_path)}")
            print(f"  t shape: {t_tensor.shape}")
            if x_tensor.numel() > 0:
                print(f"  x shape: {x_tensor.shape}")
            print(f"  m shape: {m_tensor.shape}")
            print(f"  dmdt shape: {dmdt_tensor.shape}")

            self.data_list.append(data_info)

        except Exception as e:
            print(f"Error loading {self.file_path}: {e}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx) -> dict:
        """
        Returns a dictionary containing t, x, m, dmdt for the given index
        """
        return self.data_list[idx]

    def get_sample_data(
        self, idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get sample data for testing"""
        if idx >= len(self.data_list):
            idx = 0
        data = self.data_list[idx]
        return data["t"], data["x"], data["m"], data["dmdt"]


class MultiRoundIsingDataset(Dataset):
    """
    Load multiple rounds into a single dataset and expose stacked arrays:
    m_all: (R, T, H, W), dmdt_all: (R, T, H, W), times: (T,)

    Args:
        round0_file_path: path to a round0 .npz file; used to infer other rounds
        n_rounds: number of rounds to try to load
        max_time_points: if provided, truncate time dimension to first N points
    """

    def __init__(
        self,
        round0_file_path: str,
        n_rounds: int = 20,
        max_time_points: int | None = None,
    ) -> None:
        super().__init__()
        import re

        self.round0_file_path = round0_file_path
        self.n_rounds = int(n_rounds)
        self.max_time_points = (
            int(max_time_points) if max_time_points is not None else None
        )

        m = re.match(
            r"^(.*_seed.*)/round\d+/(.*_seed.*)_round\d+\.npz$", round0_file_path
        )
        if not m:
            raise ValueError(
                f"round0_file_path format not recognized: {round0_file_path}"
            )
        dir_prefix = m.group(1)
        file_prefix = m.group(2)

        m_list: list[np.ndarray] = []
        dmdt_list: list[np.ndarray] = []
        times_ref: np.ndarray | None = None
        H = W = None
        loaded_round_indices: list[int] = []

        for r in range(self.n_rounds):
            npz_path = f"{dir_prefix}/round{r}/{file_prefix}_round{r}.npz"
            if not os.path.exists(npz_path):
                continue

            data = np.load(npz_path, allow_pickle=True)
            meta_path = npz_path.replace(".npz", ".json")
            meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

            # Prefer spins_meso
            m_fields = data.get("spins_meso", None)
            if m_fields is None:
                x = data.get("spins", None)
                if x is None or not isinstance(x, np.ndarray) or x.size == 0:
                    print(f"⚠️ Missing field in {npz_path}; skipping round {r}")
                    continue
                Tn = x.shape[0]
                M = int(meta.get("M"))
                B = int(meta.get("B"))
                L = int(meta.get("L"))
                x = x.reshape(Tn, L, L)
                m_fields = x.reshape(Tn, M, B, M, B).mean(axis=(2, 4))

            times = data["times"]
            if (
                self.max_time_points is not None
                and times.shape[0] > self.max_time_points
            ):
                t_sel = self.max_time_points
                m_fields = m_fields[:t_sel]
                times = times[:t_sel]

            dt = meta.get("snapshot_dt", (times[1] - times[0]))
            dmdt_fields = np.gradient(m_fields, dt, axis=0)

            if times_ref is None:
                times_ref = times
                H, W = int(m_fields.shape[1]), int(m_fields.shape[2])
            m_list.append(m_fields)
            dmdt_list.append(dmdt_fields)
            loaded_round_indices.append(r)

        if not m_list:
            raise FileNotFoundError(f"No rounds loaded from {round0_file_path}")

        # Stack to (R, T, H, W)
        self.times = torch.from_numpy(times_ref).float()
        self.m_all = torch.from_numpy(np.stack(m_list, axis=0)).float()
        self.dmdt_all = torch.from_numpy(np.stack(dmdt_list, axis=0)).float()
        self.round_indices = loaded_round_indices
        self.R, self.T, self.H, self.W = self.m_all.shape

    def __len__(self) -> int:
        return self.R

    def __getitem__(self, idx: int) -> dict:
        r = int(idx) % self.R
        return {
            "times": self.times,  # (T,)
            "m": self.m_all[r],  # (T, H, W)
            "dmdt": self.dmdt_all[r],  # (T, H, W)
            "H": self.H,
            "W": self.W,
            "T": self.T,
            "round_index": self.round_indices[r],
        }


class NonlocalPDENetwork(nn.Module):
    """
    Network that uses GlobalKernelPointEvalPeriodic with SoftmaxKernel
    for predicting dm/dt given current magnetization field m
    """

    def __init__(
        self,
        kernel_size: int = 15,
        hidden_sizes: List[int] = [64, 64, 32],
        activation: nn.Module = nn.Tanh(),
        use_bias: bool = True,
        force_net_type: str = "MLP",
        T: float = 1.0,
        h: float = 0.0,
    ):
        """
        Args:
            kernel_size: Size of the learnable kernel (must be odd)
            hidden_sizes: Hidden layer sizes for the neural network part (only used for MLP)
            activation: Activation function (only used for MLP)
            use_bias: Whether to use bias in linear layers (only used for MLP)
            force_net_type: Type of force network - "MLP" or "Tanh"
        """
        super().__init__()

        # Learnable softmax kernel
        self.kernel = SoftmaxKernel(kernel_size=kernel_size)

        # Global kernel evaluation with periodic boundary
        self.kernel_eval = LocalKernelPointEvalPeriodic(kernel=self.kernel)

        # Store force net type
        self.force_net_type = force_net_type

        # Initialize force network based on type
        if force_net_type == "MLP":
            # Use ForceNet from networks.py for processing [m, y] features
            # Input: local m value (1) + kernel output (1) = 2 features
            self.force_net = ForceNet(
                input_dim=2,
                hidden_sizes=hidden_sizes,
                activation=activation,
                output_activation=ScaledTanh(scale=2.0),  # 2 * tanh output
            )
        elif force_net_type == "Tanh":
            # Use ForceTanhNet with parameterization F(I, m, h) = -Am + tanh(BI + Cm + Dh)
            # Input: I (kernel output), m (local magnetization), h (external field)
            self.force_net = ForceTanhNet(
                T=T,
                h=h,
            )
        else:
            raise ValueError(
                f"Unknown force_net_type: {force_net_type}. Must be 'MLP' or 'Tanh'"
            )

    def forward(self, m: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            m: Magnetization field (B, H, W) or (B, 1, H, W)
            coords: Coordinates where to evaluate (B, 2) in pixel coordinates

        Returns:
            dm_dt: Predicted time derivative (B, 1)
        """
        # Step 1: Get local kernel evaluation → y (nonlocal features)
        kernel_output = self.kernel_eval(m, coords)  # (B, 1)

        # Step 2: Get local magnetization values at coordinates → m
        if m.dim() == 4:
            m = m.squeeze(1)  # (B, H, W)
        elif m.dim() == 3:
            pass  # Already (B, H, W)

        B, H, W = m.shape

        # Use integer coordinate sampling for better accuracy (based on our analysis)
        coord_i = coords[:, 1].long().clamp(0, H - 1)  # y -> row indices
        coord_j = coords[:, 0].long().clamp(0, W - 1)  # x -> col indices

        # Extract m values at coordinates
        m_values = []
        for b in range(B):
            m_val = m[b, coord_i[b], coord_j[b]]  # Extract m at coordinate
            m_values.append(m_val)

        m_local = torch.stack(m_values).unsqueeze(1)  # (B, 1) m_value

        # Step 3: Prepare inputs based on force network type
        if self.force_net_type == "MLP":
            # For MLP: concatenate [kernel_output, m_local] to form 2D feature vector
            features = torch.cat(
                [kernel_output, m_local], dim=1
            )  # (B, 2) - [kernel_output, m_value]
            dm_dt = self.force_net(features)  # (B, 1)
        elif self.force_net_type == "Tanh":
            # For ForceTanhNet: F(I, m, h) = -Am + tanh(BI + Cm + Dh)
            dm_dt = self.force_net(kernel_output, m_local)  # (B, 1)
        else:
            raise ValueError(f"Unknown force_net_type: {self.force_net_type}")

        return dm_dt


def sample_random_coords(
    H: int, W: int, num_points: int, use_integer_coords: bool = True
) -> torch.Tensor:
    """
    Sample random coordinates from the grid

    Args:
        H, W: Height and width of the grid
        num_points: Number of random points to sample
        use_integer_coords: If True, sample integer coordinates; if False, continuous coordinates

    Returns:
        coords: (num_points, 2) tensor of [x, y] coordinates in pixel space
    """
    if use_integer_coords:
        # Sample integer coordinates (recommended based on our analysis)
        x_coords = torch.randint(0, W, (num_points,), dtype=torch.float32)
        y_coords = torch.randint(0, H, (num_points,), dtype=torch.float32)
    else:
        # Sample continuous coordinates
        x_coords = torch.rand(num_points) * (W - 1)
        y_coords = torch.rand(num_points) * (H - 1)

    coords = torch.stack([x_coords, y_coords], dim=1)
    return coords


def prepare_training_data(
    dataset: IsingDataset, train_batch_size: int = 40, use_integer_coords: bool = True
) -> DataLoader:
    """
    Prepare DataLoader for training with simple random time-space sampling

    Args:
        dataset: IsingDataset instance
        train_batch_size: Number of random (time, coord) samples per training batch
        use_integer_coords: If True, use integer coordinates (recommended for better accuracy)

    Returns:
        DataLoader instance
    """

    def collate_fn(batch):
        """
        Simple random sampling strategy:
        1. Randomly pick train_batch_size coordinates
        2. Randomly pick train_batch_size time indices
        3. Extract corresponding m and dmdt values
        """
        # Take the first item for sampling (assuming all items have similar structure)
        item = batch[0]  # We'll just use one data file and sample from it
        t, x, m, dmdt = item["t"], item["x"], item["m"], item["dmdt"]

        # Handle different data shapes
        if m.dim() == 3:  # (T, H, W)
            T, H, W = m.shape
            m_data, dmdt_data = m, dmdt
        elif m.dim() == 4:  # (T, B, H, W) - take first batch element
            T, B, H, W = m.shape
            m_data, dmdt_data = m[:, 0], dmdt[:, 0]  # (T, H, W)
        elif m.dim() == 2:  # (H, W) - single time step
            H, W = m.shape
            T = 1
            m_data = m.unsqueeze(0)  # (1, H, W)
            dmdt_data = dmdt.unsqueeze(0)
        else:
            # Handle 1D case - assume square grid
            total_size = m.numel()
            size = int(np.sqrt(total_size))
            m_data = m.view(1, size, size)
            dmdt_data = dmdt.view(1, size, size)
            T, H, W = 1, size, size

        # Step 1: Randomly sample train_batch_size coordinates
        coords = sample_random_coords(
            H, W, train_batch_size, use_integer_coords
        )  # (train_batch_size, 2)

        # Step 2: Randomly sample train_batch_size time indices
        time_indices = torch.randint(0, T, (train_batch_size,))  # (train_batch_size,)

        # Step 3: Extract m and dmdt values at (time_i, coord_i) pairs
        m_values = []
        dmdt_values = []
        m_fields = []
        dmdt_fields = []

        for i in range(train_batch_size):
            time_idx = time_indices[i].item()
            coord = coords[i]  # (2,) - [x, y]

            # Get the field at this time
            m_t = m_data[time_idx]  # (H, W)
            dmdt_t = dmdt_data[time_idx]  # (H, W)

            # Extract value at coordinate
            coord_i = coord[1].long().clamp(0, H - 1)  # y -> row
            coord_j = coord[0].long().clamp(0, W - 1)  # x -> col

            m_value = m_t[coord_i, coord_j]  # scalar
            dmdt_value = dmdt_t[coord_i, coord_j]  # scalar

            m_values.append(m_value)
            dmdt_values.append(dmdt_value)
            m_fields.append(m_t)
            dmdt_fields.append(dmdt_t)

        # Stack into batch tensors
        batch_dict = {
            "m_values": torch.stack(m_values).unsqueeze(1),  # (train_batch_size,1)
            "dmdt_values": torch.stack(dmdt_values).unsqueeze(
                1
            ),  # (train_batch_size,1)
            "coords": coords,  # (train_batch_size, 2)
            "time_indices": time_indices.unsqueeze(1),  # (train_batch_size,1)
            "m_fields": torch.stack(m_fields),  # (train_batch_size, H, W)
            "dmdt_fields": torch.stack(dmdt_fields),  # (train_batch_size, H, W)
            "H": H,
            "W": W,
            "T": T,
        }

        return batch_dict

    return DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)


def prepare_training_data_multi(
    dataset: MultiRoundIsingDataset,
    train_batch_size: int = 128,
    use_integer_coords: bool = True,
) -> DataLoader:
    """
    Prepare DataLoader for multi-round dataset.

    Sampling strategy per batch (size = train_batch_size):
      - sample random round indices r_i in [0, R)
      - sample random time indices t_i in [0, T)
      - sample coordinates per i, extract m and dmdt at (r_i, t_i, coord_i)
      - also return the full fields at those times for network forward
    """

    def collate_fn(_batch):
        # Use underlying stacked tensors
        m_all = dataset.m_all  # (R, T, H, W)
        dmdt_all = dataset.dmdt_all
        R, T, H, W = dataset.R, dataset.T, dataset.H, dataset.W

        coords = sample_random_coords(H, W, train_batch_size, use_integer_coords)
        round_indices = torch.randint(0, R, (train_batch_size,))
        time_indices = torch.randint(0, T, (train_batch_size,))

        m_values = []
        dmdt_values = []
        m_fields = []
        dmdt_fields = []

        for i in range(train_batch_size):
            r_i = round_indices[i].item()
            t_i = time_indices[i].item()
            coord = coords[i]

            m_t = m_all[r_i, t_i]
            dmdt_t = dmdt_all[r_i, t_i]

            ci = coord[1].long().clamp(0, H - 1)
            cj = coord[0].long().clamp(0, W - 1)
            m_val = m_t[ci, cj]
            dmdt_val = dmdt_t[ci, cj]

            m_values.append(m_val)
            dmdt_values.append(dmdt_val)
            m_fields.append(m_t)
            dmdt_fields.append(dmdt_t)

        batch_dict = {
            "m_values": torch.stack(m_values).unsqueeze(1),
            "dmdt_values": torch.stack(dmdt_values).unsqueeze(1),
            "coords": coords,
            "time_indices": time_indices.unsqueeze(1),
            "round_indices": round_indices.unsqueeze(1),
            "m_fields": torch.stack(m_fields),
            "dmdt_fields": torch.stack(dmdt_fields),
            "H": H,
            "W": W,
            "T": T,
            "R": R,
        }
        return batch_dict

    return DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)


if __name__ == "__main__":
    # Example usage
    torch.manual_seed(42)

    # Configuration
    DATA_DIR = "data/ct_glauber_L1024_ell32_sigma1.2_tau1_m00.1_Tfrac3_J1_h0_tend5.0_dt0.01_block8_kernelnearest_R0.015625_seed0/round0/ct_glauber_L1024_ell32_sigma1.2_tau1_m00.1_Tfrac3_J1_h0_tend5.0_dt0.01_block8_kernelnearest_R0.015625_seed0_round0.npz"  # Adjust this to your data directory
    KERNEL_SIZE = 3
    BATCH_SIZE = 40
    NUM_COORD_SAMPLES = 128
    LEARNING_RATE = 1e-3
    EPOCHS = 1000000
    FORCE_NET_TYPE = "Tanh"  # "MLP" or "Tanh"
    OUTPUT_DIR = f"results/nonlocal_pde_network_kernel{KERNEL_SIZE}_force{FORCE_NET_TYPE}_lr{LEARNING_RATE}"

    print("=" * 50)
    print("Nonlocal PDE Training Setup")
    print("=" * 50)

    # Load dataset
    dataset = IsingDataset(file_path=DATA_DIR)
    print(f"\nLoaded {len(dataset)} data files")

    if len(dataset) > 0:
        # Get sample data info
        sample_t, sample_x, sample_m, sample_dmdt = dataset.get_sample_data(0)
        print(f"\nSample data shapes:")
        print(f"  t: {sample_t.shape}")
        print(f"  x: {sample_x.shape}")
        print(f"  m: {sample_m.shape}")
        print(f"  dmdt: {sample_dmdt.shape}")

        # Print metadata information
        # Load metadata from the dataset
        import json

        metadata_file = DATA_DIR.replace(".npz", ".json")
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            print_metadata_info(
                metadata, "Dataset Metadata", save_to_file=True, output_dir=OUTPUT_DIR
            )
        except FileNotFoundError:
            print(f"⚠️  Metadata file not found: {metadata_file}")
    else:
        print("No data found")
        exit()

    # Initialize network
    print(
        f"\nInitializing network with kernel size {KERNEL_SIZE} and force net type {FORCE_NET_TYPE}"
    )
    network = NonlocalPDENetwork(
        kernel_size=KERNEL_SIZE,
        hidden_sizes=[64, 64, 64],
        force_net_type=FORCE_NET_TYPE,
        T=T,
        h=h,
    )

    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)

    dataloader = prepare_training_data(dataset, train_batch_size=BATCH_SIZE)
    print(f"\nDataLoader prepared with batch size {BATCH_SIZE}")

    for epoch in range(EPOCHS):
        loss_list = []
        for batch_idx, batch in enumerate(dataloader):

            m_values = batch["m_values"]
            dmdt_values = batch["dmdt_values"]
            coords = batch["coords"]
            time_indices = batch["time_indices"]
            m_fields = batch["m_fields"]
            dmdt_fields = batch["dmdt_fields"]

            network.train()
            dmdt_pred = network(m_fields, coords)
            loss = nn.MSELoss()(dmdt_pred, dmdt_values)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch:4d} | Loss: {np.mean(loss_list)}")
            # Visualize kernel and ForceNet analysis
            grad_k, grad_m = visualize_network(
                network, epoch, KERNEL_SIZE, BATCH_SIZE, LEARNING_RATE
            )
            print(f"  ForceNet sensitivity - ∂/∂k: {grad_k:.4f}, ∂/∂m: {grad_m:.4f}")
            if grad_m > grad_k:
                print(f"  Network relies more on local field m ({grad_m/grad_k:.2f}x)")
            else:
                print(
                    f"  Network relies more on kernel output k ({grad_k/grad_m:.2f}x)"
                )

        if epoch % 1000 == 0:
            torch.save(
                network.state_dict(),
                f"{OUTPUT_DIR}/epoch{epoch}.pth",
            )
