import numpy as np
import cupy as cp
import numba

# -------------------------------
# 核的频域构造 (解析公式)
# -------------------------------
def build_kernel_k(M, Lx, Ly, epsilon, kernel_type="gaussian", use_gpu=False):
    """
    构造非局域相互作用核的傅里叶表示 \hat J(k)
    归一化: 保证 \hat J(0)=1  <=>  实空间 sum J = 1
    """
    dx = Lx / M
    dy = Ly / M
    kx = np.fft.fftfreq(M, dx) * 2 * np.pi
    ky = np.fft.fftfreq(M, dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)

    if kernel_type == "gaussian":
        # Fourier of Gaussian kernel
        kernel_k = np.exp(-0.5 * (K * epsilon)**2)
    elif kernel_type == "exponential":
        kernel_k = 1.0 / (1.0 + (K * epsilon)**2)
    else:
        raise ValueError("Unknown kernel_type")

    # 归一化: 确保 sum J = 1
    kernel_k /= kernel_k[0, 0]

    if use_gpu:
        return cp.asarray(kernel_k)
    return kernel_k

# -------------------------------
# 卷积 (傅里叶空间乘法) —— CPU/混合模式使用
# （GPU 快路径里我们直接用 CuPy FFT，避免来回拷贝）
# -------------------------------
def conv_spin(spin, kernel_k, use_gpu=False):
    if use_gpu:
        spin_k = cp.fft.fft2(cp.asarray(spin))
        result = cp.fft.ifft2(kernel_k * spin_k).real
        return cp.asnumpy(result)
    else:
        spin_k = np.fft.fft2(spin)
        result = np.fft.ifft2(kernel_k * spin_k).real
        return result

# -------------------------------
# Gillespie step (Numba, CPU 路径)
# -------------------------------
@numba.njit(cache=True)
def _pick_site_by_rates(r, u):
    s = 0.0
    for i in range(r.size):
        s += r.flat[i]
        if s >= u:
            return i
    return r.size-1

@numba.njit(cache=True)
def _glauber_ct_gillespie_step(spin, r):
    L = spin.shape[0]
    Rtot = np.sum(r)
    u1 = np.random.rand()
    dt = -np.log(u1) / Rtot
    u2 = np.random.rand() * Rtot
    i = _pick_site_by_rates(r, u2)
    x, y = divmod(i, L)
    spin[x, y] = -spin[x, y]
    return dt, x, y

# -------------------------------
# GPU 版：按 rates 采样翻转位置
# -------------------------------
def _gpu_pick_site_by_rates(r_cp, rng=None):
    """
    r_cp: cupy.ndarray (L,L), 非负
    返回: (dt, flat_index)
    """
    if rng is None:
        rng = cp.random
    Rtot = r_cp.sum()  # cupy scalar
    # 采样时间
    u1 = rng.random()  # cupy scalar
    dt = -cp.log(u1) / Rtot  # cupy scalar
    # 采样事件
    u2 = rng.random() * Rtot
    cdf = r_cp.ravel().cumsum()      # cupy array
    idx = int(cp.searchsorted(cdf, u2))  # python int
    return float(dt), idx

# -------------------------------
# 主类
# -------------------------------
class Glauber2DIsingKacGPU:
    def __init__(self, L, Lx=1.0, Ly=1.0, use_gpu=False):
        self.L = L
        self.Lx = Lx
        self.Ly = Ly
        self.area = L * L
        self.use_gpu = use_gpu

    def _calc_magnetization(self, spin):
        return np.sum(spin) / self.area

    def _calc_energy_fft(self, spin, J0, h, kernel_k, J00):
        conv_val = conv_spin(spin, kernel_k, use_gpu=False)  # 这里用 CPU 版，spin 为 numpy
        E = -0.5 * J0 * np.sum(spin * (conv_val - J00 * spin)) - h * np.sum(spin)
        return E / self.area

    def flip_rate(self, beta, hloc, spin):
        arg = 2 * beta * hloc * spin
        arg = np.clip(arg, -30, 30)
        return 1.0 / (1.0 + np.exp(arg))

    def simulate_kac(self, spin_init, beta, J0=1.0, h=0.0,
                     epsilon=0.1, kernel_type="gaussian",
                     t_end=5.0, snapshot_dt=0.1, return_snapshots=True, verbose_every=1000):

        # 频域核
        kernel_k = build_kernel_k(self.L, self.Lx, self.Ly, epsilon, kernel_type, use_gpu=self.use_gpu)

        # ===== GPU 快路径 =====
        if self.use_gpu:
            L = self.L
            spin_cp = cp.asarray(spin_init.astype(np.int8))

            kernel_k_cp = kernel_k if isinstance(kernel_k, cp.ndarray) else cp.asarray(kernel_k)

            kern_real_cp = cp.fft.ifft2(kernel_k_cp).real
            kern_real_cp /= kern_real_cp.sum()
            J00 = float(kern_real_cp[0, 0].get())  # Python float，后续 CPU 也可用
            J00_cp = cp.float64(J00)

            spin_k = cp.fft.fft2(spin_cp)
            conv_val_cp = cp.fft.ifft2(kernel_k_cp * spin_k).real
            hloc_cp = h + J0 * (conv_val_cp - J00_cp * spin_cp)
            r_cp = 1.0 / (1.0 + cp.exp(cp.clip(2.0 * beta * hloc_cp * spin_cp, -30, 30)))

            # 采样主循环
            t = 0.0
            next_snap = 0.0
            snap_list = []
            time_list = []

            time_list.append(0.0)
            snap_list.append(cp.asnumpy(spin_cp))
            next_snap += snapshot_dt

            steps = 0
            while t < t_end:
                dt, idx = _gpu_pick_site_by_rates(r_cp)
                t += dt
                x, y = divmod(idx, L)
                spin_cp[x, y] = -spin_cp[x, y]

                spin_k = cp.fft.fft2(spin_cp)
                conv_val_cp = cp.fft.ifft2(kernel_k_cp * spin_k).real
                hloc_cp = h + J0 * (conv_val_cp - J00_cp * spin_cp)
                r_cp = 1.0 / (1.0 + cp.exp(cp.clip(2.0 * beta * hloc_cp * spin_cp, -30, 30)))

                while t >= next_snap and next_snap <= t_end:
                    time_list.append(next_snap)
                    snap_list.append(cp.asnumpy(spin_cp))
                    next_snap += snapshot_dt
                    if next_snap > t_end:
                        break

                steps += 1
                if verbose_every and (steps % verbose_every == 0):
                    print(f"[GPU] t={t:.3f}/{t_end}, steps={steps}, next_snap={next_snap:.3f}", end="\r")

            times = np.array(time_list, dtype=np.float64)
            snaps = np.stack(snap_list, axis=0).astype(np.int8)

            kernel_k_np = cp.asnumpy(kernel_k_cp)

            Ms = np.array([self._calc_magnetization(snaps[i]) for i in range(snaps.shape[0])])
            Es = np.array([self._calc_energy_fft(snaps[i], J0, h, kernel_k_np, J00) for i in range(snaps.shape[0])])

            if not return_snapshots:
                return times, Ms, Es
            return times, Ms, Es, snaps

        # 自作用项 J(0) 从实空间核取
        kern_real = np.fft.ifft2(kernel_k).real
        kern_real /= kern_real.sum()
        J00 = float(kern_real[0, 0])

        spin = spin_init.copy().astype(np.int8)

        # 初始场
        conv_val = conv_spin(spin, kernel_k, use_gpu=False)
        hloc = h + J0 * (conv_val - J00 * spin)
        r = self.flip_rate(beta, hloc, spin)

        # 存储
        max_snaps = int(np.ceil(t_end / snapshot_dt)) + 2
        times = np.zeros(max_snaps)
        snaps = np.zeros((max_snaps, self.L, self.L), np.int8)

        t = 0.0
        next_snap = 0.0
        n_snaps = 0
        times[n_snaps] = 0.0
        snaps[n_snaps] = spin
        n_snaps += 1
        next_snap += snapshot_dt

        steps = 0
        while t < t_end:
            dt, x, y = _glauber_ct_gillespie_step(spin, r)
            t += dt
            conv_val = conv_spin(spin, kernel_k, use_gpu=False)
            hloc = h + J0 * (conv_val - J00 * spin)
            r = self.flip_rate(beta, hloc, spin)

            while t >= next_snap and n_snaps < times.size:
                times[n_snaps] = next_snap
                snaps[n_snaps] = spin
                n_snaps += 1
                next_snap += snapshot_dt
                if next_snap > t_end:
                    break

            steps += 1
            if verbose_every and (steps % verbose_every == 0):
                print(f"[CPU] t={t:.3f}/{t_end}, steps={steps}, next_snap={next_snap:.3f}", end="\r")

        times = times[:n_snaps]
        snaps = snaps[:n_snaps]

        Ms = np.array([self._calc_magnetization(snaps[i]) for i in range(n_snaps)])
        Es = np.array([self._calc_energy_fft(snaps[i], J0, h, kernel_k, J00) for i in range(n_snaps)])
        if not return_snapshots:
            return times, Ms, Es
        return times, Ms, Es, snaps
    
    def simulate_kac_tauleap(self, spin_init, beta, J0=1.0, h=0.0,
                         epsilon=0.1, kernel_type="gaussian",
                         t_end=5.0, snapshot_dt=0.1, eps_tau=0.01,
                         return_snapshots=True, verbose_every=1000):
        """
        Continuous-time Glauber dynamics with tau-leaping.
        - eps_tau: control parameter for tau selection.
        - verbose_every: print progress every N steps.
        """

        L = self.L
        spin_cp = cp.asarray(spin_init.astype(np.int8))

        # Kernel in frequency domain
        kernel_k_cp = build_kernel_k(L, self.Lx, self.Ly, epsilon, kernel_type, use_gpu=True)
        kern_real_cp = cp.fft.ifft2(kernel_k_cp).real
        kern_real_cp /= kern_real_cp.sum()
        J00 = float(kern_real_cp[0, 0].get())  # 一步到位转成 Python float
        J00_cp = kern_real_cp[0, 0]    

        t = 0.0
        next_snap = 0.0
        snap_list = []
        time_list = []

        time_list.append(0.0)
        snap_list.append(cp.asnumpy(spin_cp))
        next_snap += snapshot_dt

        steps = 0
        while t < t_end:
            # Compute local fields and rates
            spin_k = cp.fft.fft2(spin_cp)
            conv_val_cp = cp.fft.ifft2(kernel_k_cp * spin_k).real
            hloc_cp = h + J0 * (conv_val_cp - J00_cp * spin_cp)
            r_cp = 1.0 / (1.0 + cp.exp(cp.clip(2.0 * beta * hloc_cp * spin_cp, -30, 30)))

            # Rtot = r_cp.sum()
            r_max = r_cp.max()
            if r_max == 0:
                break

            # Determine tau
            tau = eps_tau / r_max
            if t + tau > t_end:
                tau = t_end - t

            # Poisson sampling of flips
            filps = cp.random.poisson(r_cp * tau)

            # Update spins
            spin_cp *= cp.where(filps % 2 == 1, -1, 1)
            t += tau

            # save snapshots
            while t >= next_snap and next_snap <= t_end:
                time_list.append(next_snap)
                snap_list.append(cp.asnumpy(spin_cp))
                next_snap += snapshot_dt
                if next_snap > t_end:
                    break

            steps += 1
            if verbose_every and (steps % verbose_every == 0):
                print(f"[GPU Tau-Leap] t={t:.3f}/{t_end}, steps={steps}, next_snap={next_snap:.3f}", end="\r")

        times = np.array(time_list, dtype=np.float64)
        snaps = np.stack(snap_list, axis=0).astype(np.int8)

        kernel_k_np = cp.asnumpy(kernel_k_cp)
        J00 = float(J00_cp.get())
        Ms = np.array([self._calc_magnetization(snaps[i]) for i in range(snaps.shape[0])])
        Es = np.array([self._calc_energy_fft(snaps[i], J0, h, kernel_k_np, J00) for i in range(snaps.shape[0])])

        if not return_snapshots:
            return times, Ms, Es
        return times, Ms, Es, snaps


# test functions
def conv_direct(spin, kern):
    """O(L^4) 直译定义卷积 (for testing small L)."""
    L = spin.shape[0]
    out = np.zeros_like(spin, dtype=float)
    for x0 in range(L):
        for y0 in range(L):
            s = 0.0
            for i in range(L):
                for j in range(L):
                    dx = (x0 - i) % L
                    dy = (y0 - j) % L
                    s += kern[dx, dy] * spin[i, j]
            out[x0, y0] = s
    return out

def test_eps_effect():
    L = 16
    spin = np.random.choice([-1, 1], size=(L, L))
    eps1, eps2 = 0.1, 0.3

    k1 = build_kernel_k(L, 1.0, 1.0, eps1, kernel_type="gaussian", use_gpu=False)
    k2 = build_kernel_k(L, 1.0, 1.0, eps2, kernel_type="gaussian", use_gpu=False)

    conv1 = conv_spin(spin, k1)
    conv2 = conv_spin(spin, k2)
    diff = np.linalg.norm(conv1 - conv2)
    print(f"[TEST] ||conv(eps={eps1}) - conv(eps={eps2})|| = {diff:.3e}")

if __name__ == "__main__":
    L = 8
    beta = 0.5
    eps = 0.2
    spin = np.random.choice([-1, 1], size=(L, L))

    # 构造核（频域）
    kernel_k = build_kernel_k(L, 1.0, 1.0, eps, kernel_type="gaussian", use_gpu=False)
    kern_real = np.fft.ifft2(kernel_k).real
    kern_real /= kern_real.sum()
    J00 = kern_real[0, 0]

    # ---- 卷积正确性测试 ----
    conv_fft = conv_spin(spin, kernel_k, use_gpu=False)
    conv_ref = conv_direct(spin, kern_real)
    err = np.linalg.norm(conv_fft - conv_ref)
    print(f"[TEST] ||conv_fft - conv_direct|| = {err:.3e}")

    # ---- 自作用项检查 ----
    hloc0 = conv_fft - J00 * spin
    x, y = 3, 4
    spin2 = spin.copy()
    spin2[x, y] *= -1
    conv2 = conv_spin(spin2, kernel_k, use_gpu=False)
    hloc1 = conv2 - J00 * spin2
    delta = abs(hloc0[x, y] - hloc1[x, y])
    print(f"[TEST] |Δ hloc at flipped site| = {delta:.3e}")

    test_eps_effect()
