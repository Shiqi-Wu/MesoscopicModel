# kac_kernel.py
import numpy as np
from numpy.fft import fft2, ifft2

def build_kernel_fft(L, R_phys, kernel="gaussian", sigma=0.1):
    spacing = 1.0 / L
    R_lat = int(np.ceil(R_phys / spacing))

    kern = np.zeros((L, L), dtype=np.float64)
    offsets = []  # 用于存储邻域相对坐标和权重

    for dx in range(-R_lat, R_lat+1):
        for dy in range(-R_lat, R_lat+1):
            dist = spacing * np.sqrt(dx*dx + dy*dy)
            if dist > R_phys:
                continue
            if dx == 0 and dy == 0:
                continue
            if kernel == "gaussian":
                w = np.exp(-0.5 * (dist/sigma)**2)
            else:
                w = 1.0
            kern[dx % L, dy % L] = w
            offsets.append((dx, dy, w))

    kern /= kern.sum()
    norm = sum(w for _,_,w in offsets)
    offsets = [(dx,dy,w/norm) for dx,dy,w in offsets]

    return fft2(kern), offsets

def conv_spin(spin, kernel_fft):
    """
    对 spin 做周期卷积: hloc = conv(spin, kernel)
    - spin: (L,L) 自旋格点
    - kernel_fft: build_kernel_fft 返回的 FFT 核
    """
    return np.real(ifft2(fft2(spin) * kernel_fft))
