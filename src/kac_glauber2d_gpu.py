import numpy as np
import cupy as cp   # 用 GPU 加速
import numba
from numpy.fft import fft2, ifft2   # CPU 端备用


# -------------------------------
# Kernel 构造 (theoretical Gaussian)
# -------------------------------
def build_kernel_fft(L, epsilon, use_gpu=True):
    """
    构造理论 Gaussian Kac kernel 的 FFT 形式:
    J_epsilon(x) = (2π ε^2)^(-1) exp(-||x||^2 / (2ε^2)), d=2
    """
    spacing = 1.0 / L
    kern = np.zeros((L, L), dtype=np.float64)

    for dx in range(-L//2, L//2):
        for dy in range(-L//2, L//2):
            dist = spacing * np.sqrt(dx*dx + dy*dy)
            w = (1.0 / (2.0 * np.pi * epsilon**2)) * np.exp(-0.5 * (dist/epsilon)**2)
            kern[dx % L, dy % L] = w
    # 归一化
    kern /= kern.sum() 

    if use_gpu:
        return cp.fft.fft2(cp.asarray(kern))  # 返回 GPU 上的 FFT 核
    else:
        return fft2(kern)  # CPU 版本


# -------------------------------
# 卷积 (GPU / CPU)
# -------------------------------
def conv_spin(spin, kernel_fft, use_gpu=True):
    """
    周期卷积: hloc = conv(spin, kernel)
    - spin: (L,L) 自旋格点
    - kernel_fft: build_kernel_fft 返回的 FFT 核
    - use_gpu: 是否用 GPU
    """
    if use_gpu:
        spin_gpu = cp.asarray(spin)
        conv_val = cp.real(cp.fft.ifft2(cp.fft.fft2(spin_gpu) * kernel_fft))
        return cp.asnumpy(conv_val)   # 返回到 CPU
    else:
        return np.real(ifft2(fft2(spin) * kernel_fft))


# -------------------------------
# Numba Gillespie 内核 (不包含卷积)
# -------------------------------
@numba.njit(cache=True)
def _pick_site_by_rates(r, u):
    """按照 rates 累积分布抽样"""
    s = 0.0
    for i in range(r.size):
        s += r.flat[i]
        if s >= u:
            return i
    return r.size - 1


@numba.njit(cache=True)
def _glauber_ct_gillespie_step(spin, r, beta, hloc, J0, h):
    """
    Gillespie 单步 (Numba 加速)
    - spin: 当前自旋 (L,L)
    - r: 翻转率数组
    - hloc: 局域场 (需要外部更新)
    """
    L = spin.shape[0]
    N = L * L
    Rtot = np.sum(r)

    # Gillespie 时间增量
    u1 = np.random.rand()
    dt = -np.log(u1) / Rtot

    # 选翻转点
    u2 = np.random.rand() * Rtot
    i = _pick_site_by_rates(r, u2)
    x, y = divmod(i, L)

    # 翻转自旋
    spin[x, y] = -spin[x, y]

    return dt, x, y


# -------------------------------
# 主类 (外层控制逻辑)
# -------------------------------
class Glauber2DIsingKacGPU:
    def __init__(self, L, use_gpu=True):
        self.L = L
        self.area = L * L
        self.use_gpu = use_gpu

    def _calc_magnetization(self, spin):
        return np.sum(spin) / self.area

    def _calc_energy_kac_fft(self, spin, J0, h, kernel_fft):
        conv_val = conv_spin(spin, kernel_fft, use_gpu=self.use_gpu)
        E = -0.5 * J0 * np.sum(spin * conv_val) - h * np.sum(spin)
        return E / self.area
    
    def flip_rate(self, beta, hloc, spin):
        # hloc: 局域场 h_γ(x)
        arg = 2.0 * beta * hloc * spin
        # 数值稳定：限制范围避免溢出
        arg = np.clip(arg, -30, 30)
        return 1.0 / (1.0 + np.exp(arg))


    def simulate_kac(self, spin_init, beta, J0=1.0, h=0.0,
                     epsilon=0.1, t_end=50.0, snapshot_dt=0.1, return_snapshots=True):
        L = self.L
        spin = spin_init.copy().astype(np.int8)

        print(f"[simulate_kac_gpu] L={L}, epsilon={epsilon}, t_end={t_end}, GPU={self.use_gpu}")

        # 构造 kernel
        kernel_fft = build_kernel_fft(L, epsilon, use_gpu=self.use_gpu)

        # 初始场和 rates
        hloc = h + J0 * conv_spin(spin, kernel_fft, use_gpu=self.use_gpu)
        # stable version of Glauber rate
        r = self.flip_rate(beta, hloc, spin)

        # 输出存储
        max_snaps = int(np.ceil(t_end / snapshot_dt)) + 2
        times = np.zeros(max_snaps, np.float64)
        snaps = np.zeros((max_snaps, L, L), np.int8)

        t = 0.0
        next_snap = 0.0
        n_snaps = 0

        times[n_snaps] = 0.0
        snaps[n_snaps] = spin
        n_snaps += 1
        next_snap += snapshot_dt

        step = 0
        while t < t_end:
            dt, x, y = _glauber_ct_gillespie_step(spin, r, beta, hloc, J0, h)
            t += dt

            # 更新全局 hloc & rates (FFT on GPU/CPU)
            hloc = h + J0 * conv_spin(spin, kernel_fft, use_gpu=self.use_gpu)
            r = self.flip_rate(beta, hloc, spin)

            # 存快照
            while t >= next_snap and n_snaps < times.size:
                times[n_snaps] = next_snap
                snaps[n_snaps] = spin
                n_snaps += 1
                next_snap += snapshot_dt
                if next_snap > t_end:
                    break

            step += 1
            if step % 1000 == 0:
                print(f"  [info] t={t:.3f}, snaps={n_snaps}")

        times = times[:n_snaps]
        snaps = snaps[:n_snaps]

        # 计算 M, E
        Ms = np.array([self._calc_magnetization(snaps[k]) for k in range(n_snaps)])
        Es = np.array([self._calc_energy_kac_fft(snaps[k], J0, h, kernel_fft) for k in range(n_snaps)])

        if return_snapshots:
            return times, Ms, Es, snaps
        else:
            return times, Ms, Es
