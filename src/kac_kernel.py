import numpy as np
from numpy.fft import fft2, ifft2

def build_kernel_fft(L, epsilon):
    """
    构造理论 Gaussian Kac kernel 的 FFT 形式:
    J_epsilon(x) = (2π ε^2)^(-1) exp(-||x||^2 / (2ε^2)), d=2
    """
    spacing = 1.0 / L   # 格点间距 (单位方格)
    kern = np.zeros((L, L), dtype=np.float64)

    # 构造格点坐标 (周期性)
    for dx in range(-L//2, L//2):
        for dy in range(-L//2, L//2):
            dist = spacing * np.sqrt(dx*dx + dy*dy)
            w = (1.0 / (2.0 * np.pi * epsilon**2)) * np.exp(-0.5 * (dist/epsilon)**2)
            kern[dx % L, dy % L] = w

    return fft2(kern)

def conv_spin(spin, kernel_fft):
    """
    对 spin 做周期卷积: hloc = conv(spin, kernel)
    - spin: (L,L) 自旋格点
    - kernel_fft: build_kernel_fft 返回的 FFT 核
    """
    return np.real(ifft2(fft2(spin) * kernel_fft))
