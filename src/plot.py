#!/usr/bin/env python3
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import argparse
import os
import torch
import matplotlib.animation as animation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def _coarse_grain_blocks(spins: np.ndarray, B: int) -> np.ndarray:
    """
    Coarse-grain spins by averaging over BxB blocks.

    Args:
        spins: (L, L) array with values in {-1, +1}
        B: block size for coarse-graining

    Returns:
        (M, M) array where M = L // B
    """
    L = spins.shape[0]
    M = L // B

    # Reshape to (M, B, M, B) and average over the BxB blocks
    coarse = spins[: M * B, : M * B].reshape(M, B, M, B).mean(axis=(1, 3))

    return coarse


def _choose_writer(out_path: str, fps: int = 10):
    """Choose appropriate writer based on file extension"""
    if out_path.endswith(".gif"):
        return "pillow"
    elif out_path.endswith(".mp4"):
        return "ffmpeg"
    else:
        return "pillow"  # default to gif


def plot_Ms_Es(times, Ms, Es, outdir: Path):
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True, constrained_layout=True)
    ax[0].plot(times, Ms, lw=1.5, label="Magnetization")
    ax[0].set_ylabel("Magnetization per spin")
    ax[0].grid(alpha=0.3)

    ax[1].plot(times, Es, lw=1.5, label="Energy", color="tab:orange")
    ax[1].set_ylabel("Energy per spin")
    ax[1].set_xlabel("Time")
    ax[1].grid(alpha=0.3)

    out_path = outdir / "Magnetization_Energy.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[save] {out_path}")


def plot_animation(times, spins, spins_meso, outdir: Path, interval=200, stride=10):
    """生成动图，左边 spins，右边 spins_meso，每 stride 帧取一帧"""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    im1 = axes[0].imshow(spins[0], cmap="coolwarm", vmin=-1, vmax=1)
    im2 = axes[1].imshow(spins_meso[0], cmap="coolwarm", vmin=-1, vmax=1)

    axes[0].set_title("Spins (micro)")
    axes[1].set_title("Spins (meso)")
    for ax in axes:
        ax.axis("off")

    def update(frame_idx):
        f = frame_idx
        im1.set_data(spins[f])
        im2.set_data(spins_meso[f])
        fig.suptitle(f"t = {times[f]:.2f}")
        return im1, im2

    # 每 stride 帧取一帧
    frame_indices = range(0, len(times), stride)

    ani = animation.FuncAnimation(
        fig, update, frames=frame_indices, interval=interval, blit=False
    )

    # try:
    #     mp4_path = outdir / "animation.mp4"
    #     ani.save(mp4_path, writer="ffmpeg", dpi=150)
    #     print(f"[save] {mp4_path}")
    # except Exception as e:
        # print(f"[warn] mp4 save failed ({e}), falling back to GIF")
        # gif_path = outdir / "animation.gif"
        # ani.save(gif_path, writer="pillow", dpi=100)
        # print(f"[save] {gif_path}")
    gif_path = outdir / "animation.gif"
    ani.save(gif_path, writer="pillow", dpi=100)
    print(f"[save] {gif_path}")



def visualize_network(
    network, epoch, kernel_size, batch_size, learning_rate, save_dir="results"
):
    """
    Visualize both kernel and ForceNet properties
    """
    # Check force network type
    force_net_type = getattr(network, 'force_net_type', 'MLP')
    
    if force_net_type == "Tanh":
        return visualize_network_tanh(network, epoch, kernel_size, batch_size, learning_rate, save_dir)
    else:
        return visualize_network_mlp(network, epoch, kernel_size, batch_size, learning_rate, save_dir)


def visualize_network_mlp(
    network, epoch, kernel_size, batch_size, learning_rate, save_dir="results"
):
    """
    Visualize MLP-based ForceNet properties
    """
    # Use fonts that support mathematical symbols
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Visualize kernel
    with torch.no_grad():
        kernel_tensor = (
            network.kernel()
        )  # Call forward() to get the actual kernel (1,1,K,K)
    kernel_weights = kernel_tensor[0, 0].detach().cpu().numpy()  # Extract (K,K) part
    im1 = axes[0, 0].imshow(kernel_weights, cmap="viridis")
    axes[0, 0].set_title(f"Softmax Kernel (Epoch {epoch})")
    axes[0, 0].axis("off")
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    # 2. ForceNet sensitivity analysis
    # Create test input ranges
    m_range = torch.linspace(-2, 2, 50)
    k_range = torch.linspace(-2, 2, 50)

    # Fix k=0, vary m to see dependency on m
    m_test = m_range.unsqueeze(1)  # (50, 1)
    k_test = torch.zeros_like(m_test)  # (50, 1)
    input_m_vary = torch.cat([k_test, m_test], dim=1)  # (50, 2) - [k=0, m_varying]

    with torch.no_grad():
        output_m_vary = network.force_net(input_m_vary)

    axes[0, 1].plot(
        m_range.numpy(),
        output_m_vary.numpy(),
        "b-",
        linewidth=2,
        label="ForceNet(k=0, m)",
    )
    axes[0, 1].set_xlabel("m value")
    axes[0, 1].set_ylabel("dm/dt output")
    axes[0, 1].set_title("ForceNet dependency on m")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Fix m=0, vary k to see dependency on k
    m_test_2 = torch.zeros(50, 1)  # (50, 1)
    k_test_2 = k_range.unsqueeze(1)  # (50, 1)
    input_k_vary = torch.cat([k_test_2, m_test_2], dim=1)  # (50, 2) - [k_varying, m=0]

    with torch.no_grad():
        output_k_vary = network.force_net(input_k_vary)

    axes[1, 0].plot(
        k_range.numpy(),
        output_k_vary.numpy(),
        "r-",
        linewidth=2,
        label="ForceNet(k, m=0)",
    )
    axes[1, 0].set_xlabel("k value (kernel output)")
    axes[1, 0].set_ylabel("dm/dt output")
    axes[1, 0].set_title("ForceNet dependency on k")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 3. Gradient analysis - which input does ForceNet care more about?
    # Sample some test points
    test_inputs = torch.randn(100, 2) * 0.5  # (100, 2) random inputs
    test_inputs.requires_grad_(True)

    outputs = network.force_net(test_inputs)
    loss = outputs.sum()  # Sum for gradient computation
    loss.backward()

    # Compute gradient magnitudes - NOTE: input order is [k, m]
    grad_k = test_inputs.grad[:, 0].abs()  # Gradient w.r.t. k (kernel output)
    grad_m = test_inputs.grad[:, 1].abs()  # Gradient w.r.t. m (local field)

    # Plot gradient comparison
    x_pos = np.arange(2)
    grad_means = [grad_k.mean().item(), grad_m.mean().item()]
    grad_stds = [grad_k.std().item(), grad_m.std().item()]

    bars = axes[1, 1].bar(
        x_pos, grad_means, yerr=grad_stds, capsize=5, color=["red", "blue"], alpha=0.7
    )
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(["∂output/∂k", "∂output/∂m"])
    axes[1, 1].set_ylabel("Average |gradient|")
    axes[1, 1].set_title("ForceNet Sensitivity Analysis")
    axes[1, 1].grid(True, alpha=0.3)

    # Add text annotation
    if grad_means[1] > grad_means[0]:  # grad_m > grad_k
        more_sensitive = "m (local field)"
        ratio = grad_means[1] / grad_means[0]  # grad_m / grad_k
    else:
        more_sensitive = "k (kernel output)"
        ratio = grad_means[0] / grad_means[1]  # grad_k / grad_m

    axes[1, 1].text(
        0.5,
        max(grad_means) * 0.8,
        f"More sensitive to {more_sensitive}\n(ratio: {ratio:.2f}x)",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
    )

    plt.tight_layout()

    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create detailed filename with parameters
    filename = f"network_analysis_mlp_kernel{kernel_size}_batch{batch_size}_lr{learning_rate:.0e}_epoch{epoch:06d}.png"
    filepath = os.path.join(save_dir, filename)

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()  # Close the figure to free memory instead of showing

    print(f"  Visualization saved: {filename}")

    return grad_means[0], grad_means[1]  # Return grad_k, grad_m


def visualize_network_tanh(
    network, epoch, kernel_size, batch_size, learning_rate, save_dir="results"
):
    """
    Visualize ForceTanhNet-based properties
    """
    # Use fonts that support mathematical symbols
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Visualize kernel
    with torch.no_grad():
        kernel_tensor = network.kernel()  # Call forward() to get the actual kernel (1,1,K,K)
    kernel_weights = kernel_tensor[0, 0].detach().cpu().numpy()  # Extract (K,K) part
    im1 = axes[0, 0].imshow(kernel_weights, cmap="viridis")
    axes[0, 0].set_title(f"Softmax Kernel (Epoch {epoch})")
    axes[0, 0].axis("off")
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    # 2. ForceTanhNet parameter visualization
    with torch.no_grad():
        A = network.force_net.A.item()
        B = network.force_net.B.item()
        C = network.force_net.C.item()
        D = network.force_net.D.item()

    # Plot parameter values
    param_names = ['A', 'B', 'C', 'D']
    param_values = [A, B, C, D]
    colors = ['red', 'blue', 'green', 'orange']
    
    bars = axes[0, 1].bar(param_names, param_values, color=colors, alpha=0.7)
    axes[0, 1].set_ylabel("Parameter Value")
    axes[0, 1].set_title("ForceTanhNet Parameters")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, param_values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')

    # 3. ForceTanhNet function visualization: F(I, m, h) = -Am + tanh(BI + Cm + Dh)
    # Create test input ranges
    I_range = torch.linspace(-2, 2, 50)  # kernel output range
    m_range = torch.linspace(-2, 2, 50)  # magnetization range
    h_range = torch.linspace(-1, 1, 50)  # external field range

    # Fix h=0, vary I and m
    I_mesh, m_mesh = torch.meshgrid(I_range, m_range, indexing='ij')
    h_fixed = torch.zeros_like(I_mesh)
    
    with torch.no_grad():
        # Reshape for batch processing
        I_flat = I_mesh.flatten().unsqueeze(1)  # (2500, 1)
        m_flat = m_mesh.flatten().unsqueeze(1)  # (2500, 1)
        h_flat = h_fixed.flatten().unsqueeze(1)  # (2500, 1)
        
        output_flat = network.force_net(I_flat, m_flat, h_flat)  # (2500, 1)
        output_mesh = output_flat.reshape(I_mesh.shape)

    im2 = axes[1, 0].contourf(I_mesh.numpy(), m_mesh.numpy(), output_mesh.numpy(), 
                              levels=20, cmap='RdBu_r')
    axes[1, 0].set_xlabel("I (kernel output)")
    axes[1, 0].set_ylabel("m (magnetization)")
    axes[1, 0].set_title("ForceTanhNet: F(I, m, h=0)")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    # 4. Parameter sensitivity analysis
    # Test gradient w.r.t. each parameter
    test_I = torch.tensor([[1.0]], requires_grad=True)
    test_m = torch.tensor([[0.5]], requires_grad=True)
    test_h = torch.tensor([[0.0]], requires_grad=True)
    
    with torch.no_grad():
        # Get current parameter values
        A_val = network.force_net.A.item()
        B_val = network.force_net.B.item()
        C_val = network.force_net.C.item()
        D_val = network.force_net.D.item()
        
        # Compute analytical gradients
        # F = -Am + tanh(BI + Cm + Dh)
        # ∂F/∂A = -m
        # ∂F/∂B = I * sech²(BI + Cm + Dh)
        # ∂F/∂C = m * sech²(BI + Cm + Dh)
        # ∂F/∂D = h * sech²(BI + Cm + Dh)
        
        tanh_arg = B_val * test_I + C_val * test_m + D_val * test_h
        sech_sq = 1.0 / torch.cosh(tanh_arg) ** 2
        
        grad_A = -test_m.item()
        grad_B = (test_I * sech_sq).item()
        grad_C = (test_m * sech_sq).item()
        grad_D = (test_h * sech_sq).item()

    # Plot parameter sensitivities
    sens_names = ['∂F/∂A', '∂F/∂B', '∂F/∂C', '∂F/∂D']
    sens_values = [abs(grad_A), abs(grad_B), abs(grad_C), abs(grad_D)]
    
    bars = axes[1, 1].bar(sens_names, sens_values, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel("|Gradient|")
    axes[1, 1].set_title("Parameter Sensitivity (I=1, m=0.5, h=0)")
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, sens_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create detailed filename with parameters
    filename = f"network_analysis_tanh_kernel{kernel_size}_batch{batch_size}_lr{learning_rate:.0e}_epoch{epoch:06d}.png"
    filepath = os.path.join(save_dir, filename)

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()  # Close the figure to free memory instead of showing

    print(f"  Visualization saved: {filename}")

    return sens_values[0], sens_values[1]  # Return grad_A, grad_B


def save_spins_animation(
    spins: np.ndarray,
    times: np.ndarray,
    out_path: str,
    fps: int = 10,
    cmap: str = "gray",
    dpi: int = 120,
    B: int | None = None,
    params: dict | None = None,
) -> None:
    """
    Save an animation of spin configurations over time.

    Args:
        spins: (n_frames, L, L) with values in {-1, +1}
        times: (n_frames,) corresponding to MCSS times
        out_path: path ending with .gif or .mp4
        fps: frames per second
        cmap: colormap for visualization
        dpi: dots per inch for output
        B: block size for coarse-graining (if None, only show micro spins)
        params: dictionary with parameters for title display
    """
    if spins is None:
        raise ValueError("spins=None: this run did not save spin snapshots.")

    n_frames, L, L2 = spins.shape
    assert L == L2, f"Expected square spins array, got {spins.shape}"

    # If B is provided and divides L, make a side-by-side animation (micro | coarse)
    side_by_side = B is not None and (L % int(B) == 0)

    # Prepare optional dynamic suptitle with parameters
    M_eff = None
    if side_by_side:
        M = L // int(B)
        M_eff = M
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(8, 4))
        imL = axL.imshow(spins[0], cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        axL.set_title(f"micro {M_eff}x{M_eff} with {B}x{B}")
        axL.axis("off")
        coarse0 = _coarse_grain_blocks(spins[0], int(B))
        imR = axR.imshow(coarse0, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        axR.set_title(f"coarse {M}x{M}")
        axR.axis("off")
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(4, 4))
        imL = ax.imshow(spins[0], cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"t={int(times[0])} MCSS")
        ax.axis("off")
        fig.tight_layout()

    title_text = None
    make_suptitle = None
    if params is not None:
        # Compose global title: t | M,B | T/beta | h
        M_txt = params.get("M", M_eff)
        B_txt = params.get("B", B)
        T_txt = params.get("T", None)
        beta_txt = params.get(
            "beta",
            (None if T_txt is None else (0.0 if T_txt == 0 else 1.0 / float(T_txt))),
        )
        h_txt = params.get("h", None)

        def make_suptitle_fn(t_val: int) -> str:
            parts = [
                f"t={t_val} MCSS",
                (f"M={int(M_txt)}" if M_txt is not None else None),
                (f"B={int(B_txt)}" if B_txt is not None else None),
                (
                    f"T={float(T_txt):.3f}"
                    if T_txt is not None
                    else (
                        f"beta={float(beta_txt):.3f}" if beta_txt is not None else None
                    )
                ),
                (f"h={float(h_txt):.3f}" if h_txt is not None else None),
                (
                    f"initseed={int(params.get('init_seed'))}"
                    if params.get("init_seed") is not None
                    else None
                ),
                (
                    f"dynseed={int(params.get('dyn_seed'))}"
                    if params.get("dyn_seed") is not None
                    else None
                ),
            ]
            return " | ".join([p for p in parts if p])

        make_suptitle = make_suptitle_fn
        title_text = fig.suptitle(make_suptitle(int(times[0])))
        try:
            fig.subplots_adjust(top=0.88)
        except Exception:
            pass

    def init_anim():
        # Return artists for animation initialization
        if side_by_side:
            arts = [imL, imR]
        else:
            arts = [imL]
        if title_text is not None and make_suptitle is not None:
            title_text.set_text(make_suptitle(int(times[0])))
            arts.append(title_text)
        return arts

    def update(frame_idx):
        if side_by_side:
            imL.set_data(spins[frame_idx])
            imR.set_data(_coarse_grain_blocks(spins[frame_idx], int(B)))
            arts = [imL, imR]
            if title_text is not None and make_suptitle is not None:
                title_text.set_text(make_suptitle(int(times[frame_idx])))
                arts.append(title_text)
            return arts
        else:
            imL.set_data(spins[frame_idx])
            arts = [imL]
            if title_text is not None and make_suptitle is not None:
                title_text.set_text(make_suptitle(int(times[frame_idx])))
                arts.append(title_text)
            return arts

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init_anim,
        interval=1000.0 / max(fps, 1),
        blit=False,
    )
    writer = _choose_writer(out_path, fps=fps)
    ani.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Animation saved to: {out_path}")


def save_spins_interactive_animation(
    spins: np.ndarray,
    times: np.ndarray,
    out_path: str,
    B: int | None = None,
    params: dict | None = None,
    cmap: str = "RdBu",
    max_frames: int = 100,
    downsample_factor: int = 4,
) -> None:
    """
    Save an interactive animation of spin configurations over time using Plotly.

    Args:
        spins: (n_frames, L, L) with values in {-1, +1}
        times: (n_frames,) corresponding to MCSS times
        out_path: path ending with .html
        B: block size for coarse-graining (if None, only show micro spins)
        params: dictionary with parameters for title display
        cmap: colormap for visualization
        max_frames: maximum number of frames to include (for file size control)
        downsample_factor: factor to downsample spatial resolution (e.g., 4 means 1024->256)
    """
    if spins is None:
        raise ValueError("spins=None: this run did not save spin snapshots.")

    n_frames, L, L2 = spins.shape
    assert L == L2, f"Expected square spins array, got {spins.shape}"

    # If B is provided and divides L, make a side-by-side animation (micro | coarse)
    side_by_side = B is not None and (L % int(B) == 0)

    if side_by_side:
        M = L // int(B)
        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[f"Micro {L}x{L}", f"Coarse {M}x{M}"],
            horizontal_spacing=0.1,
        )

        # Create frames for animation
        frames = []
        for i in range(n_frames):
            # Micro spins
            micro_spins = spins[i]
            # Coarse spins
            coarse_spins = _coarse_grain_blocks(spins[i], int(B))

            # Create heatmap traces
            micro_trace = go.Heatmap(
                z=micro_spins,
                colorscale=cmap,
                zmin=-1,
                zmax=1,
                showscale=False,
                name="Micro",
            )

            coarse_trace = go.Heatmap(
                z=coarse_spins,
                colorscale=cmap,
                zmin=-1,
                zmax=1,
                showscale=False,
                name="Coarse",
            )

            frame = go.Frame(
                data=[micro_trace, coarse_trace],
                name=str(i),
                layout=go.Layout(title=f"t={times[i]:.3f} MCSS"),
            )
            frames.append(frame)

        # Add initial traces
        micro_trace = go.Heatmap(
            z=spins[0],
            colorscale=cmap,
            zmin=-1,
            zmax=1,
            showscale=True,
            colorbar=dict(x=0.45, len=0.5),
            name="Micro",
        )

        coarse_trace = go.Heatmap(
            z=_coarse_grain_blocks(spins[0], int(B)),
            colorscale=cmap,
            zmin=-1,
            zmax=1,
            showscale=False,
            name="Coarse",
        )

        fig.add_trace(micro_trace, row=1, col=1)
        fig.add_trace(coarse_trace, row=1, col=2)

    else:
        # Single plot
        fig = go.Figure()

        # Create frames for animation
        frames = []
        for i in range(n_frames):
            trace = go.Heatmap(
                z=spins[i],
                colorscale=cmap,
                zmin=-1,
                zmax=1,
                showscale=True,
                name="Spins",
            )

            frame = go.Frame(
                data=[trace],
                name=str(i),
                layout=go.Layout(title=f"t={times[i]:.3f} MCSS"),
            )
            frames.append(frame)

        # Add initial trace
        trace = go.Heatmap(
            z=spins[0], colorscale=cmap, zmin=-1, zmax=1, showscale=True, name="Spins"
        )
        fig.add_trace(trace)

    # Add frames to figure
    fig.frames = frames

    # Create slider
    sliders = [
        dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(
                font=dict(size=20), prefix="Time: ", visible=True, xanchor="right"
            ),
            transition=dict(duration=300, easing="cubic-in-out"),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.1,
            y=0,
            steps=[
                dict(
                    args=[
                        [str(i)],
                        dict(frame=dict(duration=300, redraw=True), mode="immediate"),
                    ],
                    label=f"{times[i]:.3f}",
                    method="animate",
                )
                for i in range(
                    0, n_frames, max(1, n_frames // 20)
                )  # Show every 20th frame on slider
            ],
        )
    ]

    # Update layout
    fig.update_layout(
        title=(
            f"Spin Configuration Animation | {params.get('M', 'N/A')}x{params.get('B', 'N/A')} | T={params.get('T', 'N/A'):.3f} | h={params.get('h', 'N/A'):.3f}"
            if params
            else "Spin Configuration Animation"
        ),
        sliders=sliders,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list(
                    [
                        dict(
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=500, redraw=True),
                                    fromcurrent=True,
                                    transition=dict(
                                        duration=300, easing="quadratic-in-out"
                                    ),
                                ),
                            ],
                            label="Play",
                            method="animate",
                        ),
                        dict(
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                ),
                            ],
                            label="Pause",
                            method="animate",
                        ),
                    ]
                ),
                pad=dict(r=10, t=87),
                showactive=False,
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top",
            )
        ],
        width=800 if side_by_side else 400,
        height=400,
        margin=dict(l=50, r=50, t=100, b=50),
    )

    # Update axes
    if side_by_side:
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=1, col=1)
        fig.update_xaxes(showticklabels=False, row=1, col=2)
        fig.update_yaxes(showticklabels=False, row=1, col=2)
    else:
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

    # Save as HTML
    fig.write_html(out_path)
    print(f"Interactive animation saved to: {out_path}")


def create_spins_animation_from_data(
    data_file: str,
    output_path: str = "results/spins_animation.html",
    interactive: bool = True,
):
    """
    Create spins animation from a data file.

    Args:
        data_file: path to .npz data file
        output_path: path for output animation
        interactive: if True, create interactive HTML animation; if False, create GIF
    """
    # Load data
    data = np.load(data_file)
    metadata_file = data_file.replace(".npz", ".json")

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Extract data
    spins = data["spins"]  # (T, L, L) - time is first dimension
    times = data["times"]  # (T,)

    print(f"📊 Data shapes:")
    print(f"  spins: {spins.shape} (T, L, L)")
    print(f"  times: {times.shape} (T,)")
    print(f"  Time range: {times[0]:.3f} to {times[-1]:.3f}")

    # Get parameters
    B = metadata.get("B", 8)
    M = metadata.get("M", 128)
    T = metadata.get("T", None)
    h = metadata.get("h", 0.0)

    # Create parameters dict for title
    params = {
        "M": M,
        "B": B,
        "T": T,
        "h": h,
        "init_seed": metadata.get("seed", 0),
        "dyn_seed": metadata.get("seed", 0),
    }

    # Create animation
    if interactive:
        save_spins_interactive_animation(
            spins=spins,
            times=times,
            out_path=output_path,
            B=B,
            params=params,
            cmap="RdBu",
        )
    else:
        save_spins_animation(
            spins=spins,
            times=times,
            out_path=output_path,
            fps=10,
            cmap="RdBu_r",
            dpi=120,
            B=B,
            params=params,
        )


def save_spins_interactive_animation(
    spins: np.ndarray,
    times: np.ndarray,
    out_path: str,
    B: int | None = None,
    params: dict | None = None,
    cmap: str = "RdBu",
    max_frames: int = 50,
) -> None:
    """
    Save an interactive animation of spin configurations over time using Plotly.

    Args:
        spins: (n_frames, L, L) with values in {-1, +1}
        times: (n_frames,) corresponding to MCSS times
        out_path: path ending with .html
        B: block size for coarse-graining (if None, only show micro spins)
        params: dictionary with parameters for title display
        cmap: colormap for visualization
        max_frames: maximum number of frames to include (for file size control)
    """
    if spins is None:
        raise ValueError("spins=None: this run did not save spin snapshots.")

    n_frames, L, L2 = spins.shape
    assert L == L2, f"Expected square spins array, got {spins.shape}"

    print(f"📊 Original data: {n_frames} frames, {L}x{L} resolution")

    # Only take first max_frames to reduce file size, keep original spatial resolution
    if n_frames > max_frames:
        spins = spins[:max_frames]
        times = times[:max_frames]
        n_frames = max_frames
        print(f"📉 Using first {max_frames} frames for file size control")

    # If B is provided and divides L, make a side-by-side animation (micro | coarse)
    side_by_side = B is not None and (L % int(B) == 0)

    if side_by_side:
        M = L // int(B)
        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[f"Micro {L}x{L}", f"Coarse {M}x{M}"],
            horizontal_spacing=0.1,
        )

        # Create frames for animation
        frames = []
        for i in range(n_frames):
            # Micro spins
            micro_spins = spins[i]
            # Coarse spins
            coarse_spins = _coarse_grain_blocks(spins[i], int(B))

            # Create heatmap traces
            micro_trace = go.Heatmap(
                z=micro_spins,
                colorscale=cmap,
                zmin=-1,
                zmax=1,
                showscale=False,
                name="Micro",
            )

            coarse_trace = go.Heatmap(
                z=coarse_spins,
                colorscale=cmap,
                zmin=-1,
                zmax=1,
                showscale=False,
                name="Coarse",
            )

            frame = go.Frame(
                data=[micro_trace, coarse_trace],
                name=str(i),
                layout=go.Layout(
                    title=(
                        f"t={times[i]:.3f} MCSS | L={L} | M={params.get('M', 'N/A')} | B={B} | T={params.get('T', 'N/A'):.3f} | J={params.get('J', 'N/A'):.3f} | h={params.get('h', 'N/A'):.3f}"
                        if params
                        else f"t={times[i]:.3f} MCSS"
                    )
                ),
            )
            frames.append(frame)

        # Add initial traces with consistent color mapping and colorbars
        micro_trace = go.Heatmap(
            z=spins[0],
            colorscale=cmap,
            zmin=-1,
            zmax=1,
            showscale=True,
            colorbar=dict(x=0.25, len=0.4, title="Micro"),
            name="Micro",
        )

        coarse_trace = go.Heatmap(
            z=_coarse_grain_blocks(spins[0], int(B)),
            colorscale=cmap,
            zmin=-1,
            zmax=1,
            showscale=True,
            colorbar=dict(x=0.75, len=0.4, title="Coarse"),
            name="Coarse",
        )

        fig.add_trace(micro_trace, row=1, col=1)
        fig.add_trace(coarse_trace, row=1, col=2)

    else:
        # Single plot
        fig = go.Figure()

        # Create frames for animation
        frames = []
        for i in range(n_frames):
            trace = go.Heatmap(
                z=spins[i],
                colorscale=cmap,
                zmin=-1,
                zmax=1,
                showscale=True,
                name="Spins",
            )

            frame = go.Frame(
                data=[trace],
                name=str(i),
                layout=go.Layout(
                    title=(
                        f"t={times[i]:.3f} MCSS | L={L} | M={params.get('M', 'N/A')} | T={params.get('T', 'N/A'):.3f} | J={params.get('J', 'N/A'):.3f} | h={params.get('h', 'N/A'):.3f}"
                        if params
                        else f"t={times[i]:.3f} MCSS"
                    )
                ),
            )
            frames.append(frame)

        # Add initial trace
        trace = go.Heatmap(
            z=spins[0], colorscale=cmap, zmin=-1, zmax=1, showscale=True, name="Spins"
        )
        fig.add_trace(trace)

    # Add frames to figure
    fig.frames = frames

    # Create slider with fewer steps for smaller file size
    slider_steps = min(20, n_frames)  # Max 20 slider steps
    step_indices = np.linspace(0, n_frames - 1, slider_steps, dtype=int)

    sliders = [
        dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(
                font=dict(size=20), prefix="Time: ", visible=True, xanchor="right"
            ),
            transition=dict(duration=300, easing="cubic-in-out"),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.1,
            y=0,
            steps=[
                dict(
                    args=[
                        [str(i)],
                        dict(frame=dict(duration=300, redraw=True), mode="immediate"),
                    ],
                    label=f"{times[i]:.3f}",
                    method="animate",
                )
                for i in step_indices
            ],
        )
    ]

    # Update layout with proper aspect ratio
    if params:
        title_text = (
            f"Spin Configuration Animation | "
            f"L={params.get('L', 'N/A')}×{params.get('L', 'N/A')} | "
            f"M={params.get('M', 'N/A')} | "
            f"B={params.get('B', 'N/A')} | "
            f"T={params.get('T', 'N/A'):.3f} | "
            f"J={params.get('J', 'N/A'):.3f} | "
            f"h={params.get('h', 'N/A'):.3f} | "
            f"σ={params.get('sigma', 'N/A'):.3f} | "
            f"τ={params.get('tau', 'N/A'):.3f}"
        )
    else:
        title_text = "Spin Configuration Animation"

    fig.update_layout(
        title=title_text,
        sliders=sliders,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list(
                    [
                        dict(
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=500, redraw=True),
                                    fromcurrent=True,
                                    transition=dict(
                                        duration=300, easing="quadratic-in-out"
                                    ),
                                ),
                            ],
                            label="Play",
                            method="animate",
                        ),
                        dict(
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                ),
                            ],
                            label="Pause",
                            method="animate",
                        ),
                    ]
                ),
                pad=dict(r=10, t=87),
                showactive=False,
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top",
            )
        ],
        width=1200 if side_by_side else 600,
        height=600 if side_by_side else 600,
        margin=dict(l=50, r=50, t=100, b=50),
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent paper background
    )

    # Update axes with equal aspect ratio and no grid
    if side_by_side:
        # Left subplot (micro)
        fig.update_xaxes(
            showticklabels=False,
            row=1,
            col=1,
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
        )
        fig.update_yaxes(
            showticklabels=False,
            row=1,
            col=1,
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
        )
        # Right subplot (coarse)
        fig.update_xaxes(
            showticklabels=False,
            row=1,
            col=2,
            scaleanchor="y2",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
        )
        fig.update_yaxes(
            showticklabels=False,
            row=1,
            col=2,
            scaleanchor="x2",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
        )
    else:
        fig.update_xaxes(
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
        )
        fig.update_yaxes(
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
        )

    # Save as HTML with compression
    fig.write_html(out_path, config={"displayModeBar": False})
    print(f"Interactive animation saved to: {out_path}")


def load_spins_data(data_file: str):
    """
    Load spins data and metadata from files.

    Args:
        data_file: path to .npz data file

    Returns:
        tuple: (spins, times, metadata)
            - spins: (T, L, L) array with spin configurations
            - times: (T,) array with time values
            - metadata: dict with simulation parameters
    """
    # Load data
    data = np.load(data_file)
    metadata_file = data_file.replace(".npz", ".json")

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Extract data
    spins = data["spins"]  # (T, L, L) - time is first dimension
    times = data["times"]  # (T,)

    print(f"📊 Data shapes:")
    print(f"  spins: {spins.shape} (T, L, L)")
    print(f"  times: {times.shape} (T,)")
    print(f"  Time range: {times[0]:.3f} to {times[-1]:.3f}")

    # Print key parameters
    B = metadata.get("B", 8)
    M = metadata.get("M", 128)
    L = spins.shape[1]  # Actual lattice size
    T = metadata.get("T", None)
    h = metadata.get("h", 0.0)
    J = metadata.get("J", 1.0)

    print(f"🌡️ Actual T value from metadata: {T}")
    print(f"🌡️ T_frac: {metadata.get('T_frac', 'N/A')}")
    print(f"🌡️ Tc: {metadata.get('Tc', 'N/A')}")
    print(f"📐 Lattice: L={L}, M={M}, B={B}")
    print(f"⚙️ Parameters: J={J}, h={h}")

    return spins, times, metadata


def create_spins_animation(
    spins: np.ndarray,
    times: np.ndarray,
    metadata: dict,
    output_path: str = "results/spins_animation.html",
    max_frames: int = 50,
):
    """
    Create spins animation from loaded data.

    Args:
        spins: (T, L, L) array with spin configurations
        times: (T,) array with time values
        metadata: dict with simulation parameters
        output_path: path for output animation
        max_frames: maximum number of frames to include
    """
    # Get parameters
    B = metadata.get("B", 8)
    M = metadata.get("M", 128)
    L = spins.shape[1]  # Actual lattice size
    T = metadata.get("T", None)
    h = metadata.get("h", 0.0)
    J = metadata.get("J", 1.0)
    sigma = metadata.get("sigma", 1.0)
    tau = metadata.get("tau", 1.0)

    # Create parameters dict for title
    params = {
        "L": L,
        "M": M,
        "B": B,
        "T": T,
        "h": h,
        "J": J,
        "sigma": sigma,
        "tau": tau,
        "init_seed": metadata.get("seed", 0),
        "dyn_seed": metadata.get("seed", 0),
    }

    # Create animation
    save_spins_interactive_animation(
        spins=spins,
        times=times,
        out_path=output_path,
        B=B,
        params=params,
        cmap="RdBu",
        max_frames=max_frames,
    )


def create_spins_animation_from_data(
    data_file: str,
    output_path: str = "results/spins_animation.html",
    max_frames: int = 50,
):
    """
    Create spins animation from a data file (convenience function).

    Args:
        data_file: path to .npz data file
        output_path: path for output animation
        max_frames: maximum number of frames to include
    """
    # Load data
    spins, times, metadata = load_spins_data(data_file)

    # Create animation
    create_spins_animation(
        spins=spins,
        times=times,
        metadata=metadata,
        output_path=output_path,
        max_frames=max_frames,
    )
