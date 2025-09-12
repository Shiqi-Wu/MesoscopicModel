import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledTanh(nn.Module):
    """Custom activation: 2 * tanh(x)"""

    def __init__(self, scale=2.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return self.scale * torch.tanh(x)


class BaseNet(nn.Module):
    def __init__(
        self,
        input_dim=2,
        hidden_sizes=None,
        activation=nn.Tanh(),
        output_activation=nn.Tanh(),
        output_dim=1,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PhiNet(BaseNet):
    def __init__(
        self,
        input_dim=2,
        hidden_sizes=None,
        activation=nn.Tanh(),
        output_activation=nn.Tanh(),
    ):
        super().__init__(input_dim, hidden_sizes, activation, output_activation)

    def forward(self, x):
        return self.net(x)


class ForceNet(BaseNet):
    def __init__(
        self,
        input_dim=2,
        hidden_sizes=None,
        activation=nn.Tanh(),
        output_activation=nn.Tanh(),
    ):
        super().__init__(input_dim, hidden_sizes, activation, output_activation)

        assert input_dim == 2, "ForceNet only supports 2D input"

        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers)


class ForceTanhNet(nn.Module):
    """
    Force function with parameterization: F(I, m) = -Am + tanh(BI + Cm + Dh)
    where A, B, C, D are learnable parameters.

    Input: (I, m, h) where I is 2D field, m is scalar field, h is external field
    Output: scalar force value
    """

    def __init__(
        self,
        A_init: float = 1.0,
        B_init: float = 1.0,
        C_init: float = 1.0,
        D_init: float = 1.0,
    ):
        super().__init__()

        # Learnable parameters
        self.A = nn.Parameter(torch.tensor(float(A_init)))
        self.B = nn.Parameter(torch.tensor(float(B_init)))
        self.C = nn.Parameter(torch.tensor(float(C_init)))
        self.D = nn.Parameter(torch.tensor(float(D_init)))

    def forward(
        self, I: torch.Tensor, m: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: F(I, m, h) = -Am + tanh(BI + Cm + Dh)

        Args:
            I: 2D field tensor of shape (batch_size, 1) - scalar value of 2D field
            m: scalar field tensor of shape (batch_size, 1) - scalar field value
            h: external field tensor of shape (batch_size, 1) - external field value

        Returns:
            Force values of shape (batch_size, 1)
        """
        # All inputs are (batch_size, 1)
        BI = self.B * I  # (batch_size, 1)
        Cm = self.C * m * 0  # (batch_size, 1)
        Dh = self.D * h  # (batch_size, 1)

        # Compute tanh argument: BI + Cm + Dh
        tanh_arg = BI + Cm + Dh  # (batch_size, 1)
        Am = self.A * m  # (batch_size, 1)

        return -Am + torch.tanh(tanh_arg)  # (batch_size, 1)


# ----------------------------
# 1) 三种“非负 & 和为1”的核
# ----------------------------


class SoftmaxKernel(nn.Module):
    """
    完全可学习的 KxK 核（通过 softmax 约束：非负，和为1）
    - 初始化为高斯形状（通过设定 logits = -r^2/(2*sigma^2)，softmax 后为高斯核）
    返回形状 (1,1,K,K)
    """

    def __init__(
        self,
        kernel_size: int = 15,
        sigma_init: float | None = None,
        init_mode: str = "uniform",  # "gaussian" or "uniform"
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 必须为奇数"
        self.K = kernel_size

        if init_mode.lower() == "uniform":
            # 均匀起点（softmax 后≈1/K^2）
            logits = torch.zeros(self.K, self.K)
        else:
            # 高斯型 logits 初始化：softmax(logits) ∝ exp(logits) = 高斯
            k = (self.K - 1) // 2
            xs, ys = torch.meshgrid(
                torch.arange(-k, k + 1, dtype=torch.float32),
                torch.arange(-k, k + 1, dtype=torch.float32),
                indexing="ij",
            )
            sigma = float(self.K) / 4.0 if sigma_init is None else float(sigma_init)
            logits = -((xs**2 + ys**2) / (2.0 * sigma * sigma))  # (K,K)

        # 注意：这是可学习的 logits，训练中将被更新
        self.kernel_logits = nn.Parameter(logits)

    def forward(self, device=None, dtype=None) -> torch.Tensor:
        k = torch.softmax(self.kernel_logits.view(-1), dim=0).view(1, 1, self.K, self.K)
        if device is not None:
            k = k.to(device)
        if dtype is not None:
            k = k.to(dtype)
        return k  # (1,1,K,K)


class GaussianKernel(nn.Module):
    """
    各向同性高斯核（非负，和为1）。gamma 可学习或固定。
    返回形状 (1,1,K,K)
    """

    def __init__(
        self, kernel_size: int = 15, learn_gamma: bool = True, gamma_init: float = 2.5
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 必须为奇数"
        self.K = kernel_size
        if learn_gamma:
            self.log_gamma = nn.Parameter(torch.log(torch.tensor(float(gamma_init))))
            self.learnable = True
        else:
            self.register_buffer("gamma", torch.tensor(float(gamma_init)))
            self.learnable = False

    def forward(self, device=None, dtype=None) -> torch.Tensor:
        K = self.K
        k = (K - 1) // 2
        if device is None:
            device = self.gamma.device if not self.learnable else None
        if dtype is None:
            dtype = torch.float32

        ys, xs = torch.meshgrid(
            torch.arange(-k, k + 1, device=device, dtype=dtype),
            torch.arange(-k, k + 1, device=device, dtype=dtype),
            indexing="ij",
        )
        if self.learnable:
            gamma = torch.exp(self.log_gamma).clamp(min=1e-6)
        else:
            gamma = self.gamma.to(device=device, dtype=dtype).clamp(min=1e-6)

        g = torch.exp(-(xs**2 + ys**2) / (2.0 * gamma**2))
        g = (g / g.sum()).view(1, 1, K, K)
        return g  # (1,1,K,K)


class EllipticalGaussianKernel(nn.Module):
    """
    椭圆高斯核（可旋转），参数 (sigma_x, sigma_y, theta) 可学习或固定。
    非负，和为1。返回形状 (1,1,K,K)
    """

    def __init__(
        self,
        kernel_size: int = 15,
        learn_params: bool = True,
        sigmax_init: float = 3.0,
        sigmay_init: float = 1.5,
        theta_init: float = 0.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 必须为奇数"
        self.K = kernel_size
        if learn_params:
            self.log_sigma_x = nn.Parameter(torch.log(torch.tensor(float(sigmax_init))))
            self.log_sigma_y = nn.Parameter(torch.log(torch.tensor(float(sigmay_init))))
            self.theta = nn.Parameter(torch.tensor(float(theta_init)))
            self.learnable = True
        else:
            self.register_buffer("sigma_x", torch.tensor(float(sigmax_init)))
            self.register_buffer("sigma_y", torch.tensor(float(sigmay_init)))
            self.register_buffer("theta_b", torch.tensor(float(theta_init)))
            self.learnable = False

    def forward(self, device=None, dtype=None) -> torch.Tensor:
        K = self.K
        k = (K - 1) // 2
        if dtype is None:
            dtype = torch.float32

        ys, xs = torch.meshgrid(
            torch.arange(-k, k + 1, device=device, dtype=dtype),
            torch.arange(-k, k + 1, device=device, dtype=dtype),
            indexing="ij",
        )
        if self.learnable:
            sx = torch.exp(self.log_sigma_x).clamp(min=1e-6)
            sy = torch.exp(self.log_sigma_y).clamp(min=1e-6)
            th = self.theta
        else:
            sx = self.sigma_x.to(device=device, dtype=dtype).clamp(min=1e-6)
            sy = self.sigma_y.to(device=device, dtype=dtype).clamp(min=1e-6)
            th = self.theta_b.to(device=device, dtype=dtype)

        c, s = torch.cos(th), torch.sin(th)
        # 旋转坐标：R^T [x;y]，R = [[c,-s],[s,c]]
        x_r = c * xs + s * ys
        y_r = -s * xs + c * ys

        g = torch.exp(-(x_r**2) / (2.0 * sx**2) - (y_r**2) / (2.0 * sy**2))
        g = (g / g.sum()).view(1, 1, K, K)
        return g  # (1,1,K,K)


# -----------------------------------------
# 2) 周期边界求值器（整图卷积 / 局部 patch）
#    接受任一“核模块”，核必须输出 (1,1,K,K)
# -----------------------------------------


def _to_grid(coords_pix: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """像素坐标 (x,y) -> grid_sample 归一化坐标 [-1,1]，输出 (B,1,1,2)"""
    x, y = coords_pix[:, 0], coords_pix[:, 1]
    gx = 2.0 * x / max(W - 1, 1) - 1.0
    gy = 2.0 * y / max(H - 1, 1) - 1.0
    return torch.stack([gx, gy], dim=-1).view(-1, 1, 1, 2)


class GlobalKernelPointEvalPeriodic(nn.Module):
    """
    周期边界：整图卷积 + 在坐标处取样
    适合同一张图上要取很多点（先卷一次 y_map，再多点采样）
    输入:
      m: (B,H,W) 或 (B,1,H,W)  单通道
      coords_pix: (B,2)  像素坐标 [x_col,y_row]
    输出:
      (B,1)
    """

    def __init__(self, kernel: nn.Module):
        super().__init__()
        self.kernel_module = kernel
        self.K = kernel.K
        self.pad_k = self.K // 2

    def forward(self, m: torch.Tensor, coords_pix: torch.Tensor) -> torch.Tensor:
        if m.dim() == 3:
            m = m.unsqueeze(1)  # (B,1,H,W)
        B, C, H, W = m.shape
        assert C == 1, "只支持单通道输入 (C=1)"

        device, dtype = m.device, m.dtype
        kernel = self.kernel_module(device=device, dtype=dtype)  # (1,1,K,K)

        # 1) 周期扩展后卷积，得到整图 (J*m)
        m_pad = F.pad(
            m, (self.pad_k, self.pad_k, self.pad_k, self.pad_k), mode="circular"
        )
        y_map = F.conv2d(m_pad, kernel, bias=None, stride=1, padding=0)  # (B,1,H,W)

        # 2) 为边界双线性取样平滑：再 circular pad 1，并把坐标整体 +1
        y_pad = F.pad(y_map, (1, 1, 1, 1), mode="circular")  # (B,1,H+2,W+2)
        grid = _to_grid(coords_pix + 1.0, H + 2, W + 2).to(device=device, dtype=dtype)

        val = F.grid_sample(
            y_pad, grid, mode="bilinear", align_corners=True, padding_mode="border"
        )  # (B,1,1,1)
        return val.view(B, 1)


class LocalKernelPointEvalPeriodic(nn.Module):
    """
    周期边界：局部 patch + 与核内积
    适合只在少量点上求值（复杂度 O(B*K*K)，与 H,W 无关）
    输入/输出同上
    """

    def __init__(self, kernel: nn.Module):
        super().__init__()
        self.kernel_module = kernel
        self.K = kernel.K
        self.k = (self.K - 1) // 2

        # 预先构造相对位移网格（像素单位）
        dy, dx = torch.meshgrid(
            torch.arange(-self.k, self.k + 1),
            torch.arange(-self.k, self.k + 1),
            indexing="ij",
        )
        self.register_buffer("dx", dx.float())
        self.register_buffer("dy", dy.float())

    def forward(self, m: torch.Tensor, coords_pix: torch.Tensor) -> torch.Tensor:
        if m.dim() == 3:
            m = m.unsqueeze(1)  # (B,1,H,W)
        B, C, H, W = m.shape
        assert C == 1, "只支持单通道输入 (C=1)"

        device, dtype = m.device, m.dtype
        kernel = self.kernel_module(device=device, dtype=dtype)  # (1,1,K,K)

        # 1) circular pad 到四周 k 个像素
        m_eff = F.pad(m, (self.k, self.k, self.k, self.k), mode="circular")
        Hp, Wp = H + 2 * self.k, W + 2 * self.k

        # 2) 以 (x,y) 为中心构造 KxK 采样网格；整体平移 +k 对齐扩展图
        x0 = coords_pix[:, 0].view(B, 1, 1) + self.k
        y0 = coords_pix[:, 1].view(B, 1, 1) + self.k
        xs = x0 + self.dx  # (B,K,K)
        ys = y0 + self.dy  # (B,K,K)

        gx = 2.0 * xs / max(Wp - 1, 1) - 1.0
        gy = 2.0 * ys / max(Hp - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).to(device=device, dtype=dtype)  # (B,K,K,2)

        # 3) 双线性取 patch
        patch = F.grid_sample(
            m_eff, grid, mode="bilinear", align_corners=True, padding_mode="border"
        )  # (B,1,K,K)

        # 4) 与核逐元素乘积并求和 -> (B,1)
        out = (patch * kernel).sum(dim=(2, 3))  # (B,1)
        return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from utils import count_all_params, count_trainable_params, set_seed_everywhere

    set_seed_everywhere(0)

    BATCH_SIZE = 8
    # PhiNet
    print("=" * 20, "PhiNet", "=" * 20)
    model = PhiNet(
        input_dim=2,
        hidden_sizes=[64, 64],
        activation=nn.Tanh(),
        output_activation=nn.Tanh(),
    )
    sample_input = torch.randn(BATCH_SIZE, 2)
    output = model(sample_input)
    print("Sample output:", output)

    # ForceNet
    print("=" * 20, "ForceNet", "=" * 20)
    model = ForceNet(
        hidden_sizes=[64, 64], activation=nn.Tanh(), output_activation=None
    )
    sample_input = torch.randn(BATCH_SIZE, 2)
    output = model(sample_input)
    print("Sample output:", output)

    # ForceTanhNet
    print("=" * 20, "ForceTanhNet", "=" * 20)
    force_tanh = ForceTanhNet(A_init=1.0, B_init=1.0, C_init=1.0, D_init=1.0)

    # Test with correct input format: (B, 1) for all inputs
    I = torch.randn(BATCH_SIZE, 1)  # 2D field scalar values
    m = torch.randn(BATCH_SIZE, 1)  # scalar field values
    h = torch.randn(BATCH_SIZE, 1)  # external field values
    output = force_tanh(I, m, h)
    print("Output shape:", output.shape)
    print("Output sample:", output[:3].flatten())

    # Print learnable parameters
    print("Learnable parameters:")
    print(f"A: {force_tanh.A.item():.4f}")
    print(f"B: {force_tanh.B.item():.4f}")
    print(f"C: {force_tanh.C.item():.4f}")
    print(f"D: {force_tanh.D.item():.4f}")

    # Test parameter count
    total_params = sum(p.numel() for p in force_tanh.parameters())
    print(f"Total parameters: {total_params}")

    # 构造单通道网格 (B,H,W)
    H, W = 32, 32
    K1, K2, K3 = 9, 15, 21  # K 必须奇数
    m = torch.randn(BATCH_SIZE, H, W)

    # 一批像素坐标（靠近/跨越边界也OK）
    # Just for testing
    coords = torch.tensor(
        [
            [W - 5, 3],
            [0, 1],
            [W - 1, H - 8],
            [W - 11, H - 9],
            [0, 1],
            [W / 2, H / 2 - 8],
            [W - 11, H - 9],
            [W - 1, 3],
        ],
        dtype=torch.float32,
    )

    # --- 三种核 ---
    ker_soft = SoftmaxKernel(kernel_size=K1)
    ker_gauss = GaussianKernel(kernel_size=K2, learn_gamma=True, gamma_init=2.5)
    ker_ellip = EllipticalGaussianKernel(
        kernel_size=K3,
        learn_params=True,
        sigmax_init=3.0,
        sigmay_init=1.5,
        theta_init=0.0,
    )

    # 验证核性质（非负、和为1）
    with torch.no_grad():
        ks = ker_soft(dtype=torch.float32)
        kg = ker_gauss(dtype=torch.float32)
        ke = ker_ellip(dtype=torch.float32)
        print("softmax kernel: sum=", ks.sum().item(), "min=", ks.min().item())
        print("gauss   kernel: sum=", kg.sum().item(), "min=", kg.min().item())
        print("ellip   kernel: sum=", ke.sum().item(), "min=", ke.min().item())

    # --- 整图卷积 + 取样（周期） ---
    global_soft = GlobalKernelPointEvalPeriodic(kernel=ker_soft)
    y_g_soft = global_soft(m, coords)  # (B,1)
    # --- 局部 patch + 内积（周期） ---
    local_soft = LocalKernelPointEvalPeriodic(kernel=ker_soft)
    y_l_soft = local_soft(m, coords)  # (B,1)

    global_gauss = GlobalKernelPointEvalPeriodic(kernel=ker_gauss)
    y_g_gauss = global_gauss(m, coords)  # (B,1)

    # --- 局部 patch + 内积（周期） ---
    local_ellip = LocalKernelPointEvalPeriodic(kernel=ker_ellip)
    y_l_ellip = local_ellip(m, coords)  # (B,1)

    print("Global+Soft :", y_g_soft.shape, y_g_soft[:2].flatten())
    print("Global+Gauss:", y_g_gauss.shape, y_g_gauss[:2].flatten())
    print("Local +Ellip:", y_l_ellip.shape, y_l_ellip[:2].flatten())

    print("SoftmaxKernel num parameters:", count_all_params(ker_soft))
    print("GaussianKernel num parameters:", count_all_params(ker_gauss))
    print("EllipticalGaussianKernel num parameters:", count_all_params(ker_ellip))
    print(
        "GlobalKernelPointEvalPeriodic num parameters:", count_all_params(global_soft)
    )
    print("LocalKernelPointEvalPeriodic num parameters:", count_all_params(local_ellip))

    print("SoftmaxKernel trainable num parameters:", count_trainable_params(ker_soft))
    print("GaussianKernel trainable num parameters:", count_trainable_params(ker_gauss))
    print(
        "EllipticalGaussianKernel trainable num parameters:",
        count_trainable_params(ker_ellip),
    )
    print(
        "GlobalKernelPointEvalPeriodic trainable num parameters:",
        count_trainable_params(global_soft),
    )
    print(
        "LocalKernelPointEvalPeriodic trainable num parameters:",
        count_trainable_params(local_ellip),
    )

    # check if the result of global and local is the same
    print("Global and local result is the same:", y_g_soft.shape, y_l_soft.shape)
    print(y_g_soft, y_l_soft)
    print(
        "Global and local result is the similar:",
        (y_g_soft - y_l_soft).abs().max().item() < 1e-6,
    )

    # --- 可视化三种核 ---

    # 设置支持中文的字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Softmax 核
    axes[0].imshow(ks[0, 0].detach().numpy(), cmap="viridis")
    axes[0].set_title("Softmax 核")
    axes[0].axis("off")

    # 高斯核
    axes[1].imshow(kg[0, 0].detach().numpy(), cmap="viridis")
    axes[1].set_title("高斯核")
    axes[1].axis("off")

    # 椭圆高斯核
    im = axes[2].imshow(ke[0, 0].detach().numpy(), cmap="viridis")
    axes[2].set_title("椭圆高斯核")
    axes[2].axis("off")

    # 添加颜色条
    plt.colorbar(im, ax=axes.ravel().tolist())

    plt.tight_layout()
    plt.show()
