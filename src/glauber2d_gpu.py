import numpy as np
import cupy as cp
import numba


def build_kernel_k(M, Lx, Ly, epsilon, kernel_type="gaussian", use_gpu=False):
    """Construct kernel in Fourier domain for Kac interaction.
    For nearest-neighbor, no kernel is needed (returns None).
    """
    if kernel_type == "nearest":
        return None

    dx = Lx / M
    dy = Ly / M
    kx = np.fft.fftfreq(M, dx) * 2 * np.pi
    ky = np.fft.fftfreq(M, dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)

    if kernel_type == "gaussian":
        kernel_k = np.exp(-0.5 * (K * epsilon)**2)
    elif kernel_type == "exponential":
        kernel_k = 1.0 / (1.0 + (K * epsilon)**2)
    else:
        raise ValueError("Unknown kernel_type")

    kernel_k /= kernel_k[0, 0]
    return cp.asarray(kernel_k) if use_gpu else kernel_k


def conv_spin(spin, kernel_k, use_gpu=False):
    """Convolution in Fourier space (only for Kac kernel)."""
    if kernel_k is None:
        raise ValueError("conv_spin should only be used for Kac kernels.")
    if use_gpu:
        spin_k = cp.fft.fft2(spin)
        return cp.fft.ifft2(kernel_k * spin_k).real
    else:
        spin_k = np.fft.fft2(spin)
        return np.fft.ifft2(kernel_k * spin_k).real


def nn_sum_cpu(spin):
    """Sum of 4 nearest neighbors (CPU)."""
    return (
        np.roll(spin, 1, axis=0) + np.roll(spin, -1, axis=0) +
        np.roll(spin, 1, axis=1) + np.roll(spin, -1, axis=1)
    )


def nn_sum_gpu(spin_cp):
    """Sum of 4 nearest neighbors (GPU)."""
    return (
        cp.roll(spin_cp, 1, axis=0) + cp.roll(spin_cp, -1, axis=0) +
        cp.roll(spin_cp, 1, axis=1) + cp.roll(spin_cp, -1, axis=1)
    )


@numba.njit(cache=True)
def _pick_site_by_rates(r, u):
    """Pick site index by cumulative rates (CPU)."""
    s = 0.0
    for i in range(r.size):
        s += r.flat[i]
        if s >= u:
            return i
    return r.size - 1


@numba.njit(cache=True)
def _glauber_ct_gillespie_step(spin, r):
    """One Gillespie step on CPU."""
    L = spin.shape[0]
    Rtot = np.sum(r)
    u1 = np.random.rand()
    dt = -np.log(u1) / Rtot
    u2 = np.random.rand() * Rtot
    i = _pick_site_by_rates(r, u2)
    x, y = divmod(i, L)
    spin[x, y] = -spin[x, y]
    return dt, x, y


def _gpu_pick_site_by_rates(r_cp, rng=None):
    """Pick site index by cumulative rates (GPU)."""
    if rng is None:
        rng = cp.random
    Rtot = r_cp.sum()
    u1 = rng.random()
    dt = -cp.log(u1) / Rtot
    u2 = rng.random() * Rtot
    cdf = r_cp.ravel().cumsum()
    idx = int(cp.searchsorted(cdf, u2))
    return float(dt), idx

    def simulate_kac_tauleap(self, spin_init, beta, J0=1.0, h=0.0,
                             epsilon=0.1, kernel_type="gaussian",
                             t_end=5.0, snapshot_dt=0.1, eps_tau=0.01,
                             return_mode="full", verbose_every=1000):
        """Tau-leaping simulation (Kac or nearest, CPU or GPU)."""
        # ... (code unchanged, just comments removed for brevity)

class Glauber2DIsingCT:
    def __init__(self, L, Lx=1.0, Ly=1.0, B=8, use_gpu=False):
        self.L = L
        self.Lx = Lx
        self.Ly = Ly
        self.area = L * L
        self.B = B
        self.use_gpu = use_gpu

    def _calc_magnetization(self, spin, use_gpu=False):
        return float(spin.sum() / self.area)

    def _calc_energy_fft(self, spin, J0, h, kernel_k, J00, use_gpu=False):
        """Energy for Kac interaction."""
        conv_val = conv_spin(spin, kernel_k, use_gpu=use_gpu)
        E = -0.5 * J0 * (spin * (conv_val - J00 * spin)).sum() - h * spin.sum()
        return float(E / self.area)

    def _calc_energy_nearest(self, spin, J, h, use_gpu=False):
        """Energy for nearest-neighbor interaction."""
        R = nn_sum_gpu(spin) if use_gpu else nn_sum_cpu(spin)
        E = -0.5 * J * (spin * R).sum() - h * spin.sum()
        return float(E / self.area)

    def _coarse_grain(self, spin):
        """Block-spin averaging for coarse-graining."""
        L = spin.shape[0]
        assert L % self.B == 0
        M = L // self.B
        spin_blocks = spin.reshape(M, self.B, M, self.B)
        return spin_blocks.mean(axis=(1, 3))

    def flip_rate(self, beta, hloc, spin):
        """Compute Glauber flip rate on CPU."""
        arg = 2 * beta * hloc * spin
        arg = np.clip(arg, -30, 30)
        return 1.0 / (1.0 + np.exp(arg))

    def _local_field_cpu(self, spin, J0, h, kernel_k, J00, kernel_type):
        """Local field calculation on CPU."""
        if kernel_type == "nearest":
            R = nn_sum_cpu(spin)
            return h + J0 * R
        else:
            conv_val = conv_spin(spin, kernel_k, use_gpu=False)
            return h + J0 * (conv_val - J00 * spin)

    def _local_field_gpu(self, spin_cp, J0, h, kernel_k_cp, J00_cp, kernel_type):
        """Local field calculation on GPU."""
        if kernel_type == "nearest":
            R = nn_sum_gpu(spin_cp)
            return h + J0 * R
        else:
            spin_k = cp.fft.fft2(spin_cp)
            conv_val_cp = cp.fft.ifft2(kernel_k_cp * spin_k).real
            return h + J0 * (conv_val_cp - J00_cp * spin_cp)

    def simulate(self, spin_init, beta, J0=1.0, h=0.0,
                     epsilon=0.1, kernel_type="gaussian",
                     t_end=5.0, snapshot_dt=0.1,
                     return_mode="full", verbose_every=1000):
        kernel_k = build_kernel_k(self.L, self.Lx, self.Ly, epsilon, kernel_type, use_gpu=self.use_gpu)

        # GPU 
        if self.use_gpu:
            L = self.L
            spin_cp = cp.asarray(spin_init.astype(np.int8))

            if kernel_type == "nearest":
                J00_cp = 0.0
            else:
                kernel_k_cp = kernel_k
                kern_real_cp = cp.fft.ifft2(kernel_k_cp).real
                kern_real_cp /= kern_real_cp.sum()
                J00_cp = kern_real_cp[0, 0]

            # init
            if kernel_type == "nearest":
                hloc_cp = self._local_field_gpu(spin_cp, J0, h, None, 0.0, kernel_type)
            else:
                hloc_cp = self._local_field_gpu(spin_cp, J0, h, kernel_k_cp, J00_cp, kernel_type)
            r_cp = 1.0 / (1.0 + cp.exp(cp.clip(2.0 * beta * hloc_cp * spin_cp, -30, 30)))

            t, next_snap = 0.0, 0.0
            time_list, Ms_list, Es_list = [0.0], [], []
            snap_list = [] if return_mode != "none" else None

            Ms_list.append(float(spin_cp.mean()))
            if kernel_type == "nearest":
                Es_list.append(self._calc_energy_nearest(spin_cp, J0, h, use_gpu=True))
            else:
                Es_list.append(self._calc_energy_fft(spin_cp, J0, h, kernel_k_cp, J00_cp, use_gpu=True))
            if return_mode == "full":
                snap_list.append(spin_cp.get())
            elif return_mode == "coarse":
                snap_list.append(self._coarse_grain(spin_cp).get())
            next_snap += snapshot_dt

            steps = 0
            while t < t_end:
                dt, idx = _gpu_pick_site_by_rates(r_cp)
                t += dt
                x, y = divmod(idx, L)
                spin_cp[x, y] = -spin_cp[x, y]

                # 更新本地场与速率
                if kernel_type == "nearest":
                    hloc_cp = self._local_field_gpu(spin_cp, J0, h, None, 0.0, kernel_type)
                else:
                    hloc_cp = self._local_field_gpu(spin_cp, J0, h, kernel_k_cp, J00_cp, kernel_type)
                r_cp = 1.0 / (1.0 + cp.exp(cp.clip(2.0 * beta * hloc_cp * spin_cp, -30, 30)))

                while t >= next_snap and next_snap <= t_end:
                    time_list.append(next_snap)
                    Ms_list.append(float(spin_cp.mean()))
                    if kernel_type == "nearest":
                        Es_list.append(self._calc_energy_nearest(spin_cp, J0, h, use_gpu=True))
                    else:
                        Es_list.append(self._calc_energy_fft(spin_cp, J0, h, kernel_k_cp, J00_cp, use_gpu=True))
                    if return_mode == "full":
                        snap_list.append(spin_cp.get())
                    elif return_mode == "coarse":
                        snap_list.append(self._coarse_grain(spin_cp).get())
                    next_snap += snapshot_dt

                steps += 1
                if verbose_every and (steps % verbose_every == 0):
                    print(f"[GPU] t={t:.3f}/{t_end}, steps={steps}", end="\r")

            times = np.array(time_list)
            Ms = np.array(Ms_list)
            Es = np.array(Es_list)
            if return_mode == "none":
                return times, Ms, Es
            else:
                snaps = np.stack(snap_list, axis=0)
                return times, Ms, Es, snaps

        else:
            L = self.L
            spin = spin_init.copy().astype(np.int8)

            if kernel_type == "nearest":
                J00 = 0.0
            else:
                kern_real = np.fft.ifft2(kernel_k).real
                kern_real /= kern_real.sum()
                J00 = kern_real[0, 0]

            # init
            hloc = self._local_field_cpu(spin, J0, h, kernel_k, J00, kernel_type)
            r = self.flip_rate(beta, hloc, spin)

            t, next_snap = 0.0, 0.0
            time_list, Ms_list, Es_list = [0.0], [], []
            snap_list = [] if return_mode != "none" else None

            Ms_list.append(spin.mean())
            if kernel_type == "nearest":
                Es_list.append(self._calc_energy_nearest(spin, J0, h, use_gpu=False))
            else:
                Es_list.append(self._calc_energy_fft(spin, J0, h, kernel_k, J00, use_gpu=False))
            if return_mode == "full":
                snap_list.append(spin.copy())
            elif return_mode == "coarse":
                snap_list.append(self._coarse_grain(spin).copy())
            next_snap += snapshot_dt

            steps = 0
            while t < t_end:
                dt, x, y = _glauber_ct_gillespie_step(spin, r)
                t += dt
                hloc = self._local_field_cpu(spin, J0, h, kernel_k, J00, kernel_type)
                r = self.flip_rate(beta, hloc, spin)

                while t >= next_snap and next_snap <= t_end:
                    time_list.append(next_snap)
                    Ms_list.append(spin.mean())
                    if kernel_type == "nearest":
                        Es_list.append(self._calc_energy_nearest(spin, J0, h, use_gpu=False))
                    else:
                        Es_list.append(self._calc_energy_fft(spin, J0, h, kernel_k, J00, use_gpu=False))
                    if return_mode == "full":
                        snap_list.append(spin.copy())
                    elif return_mode == "coarse":
                        snap_list.append(self._coarse_grain(spin).copy())
                    next_snap += snapshot_dt

                steps += 1
                if verbose_every and (steps % verbose_every == 0):
                    print(f"[CPU] t={t:.3f}/{t_end}, steps={steps}", end="\r")

            times = np.array(time_list)
            Ms = np.array(Ms_list)
            Es = np.array(Es_list)
            if return_mode == "none":
                return times, Ms, Es
            else:
                snaps = np.stack(snap_list, axis=0)
                return times, Ms, Es, snaps

    def simulate_tauleap(self, spin_init, beta, J0=1.0, h=0.0,
                             epsilon=0.1, kernel_type="gaussian",
                             t_end=5.0, snapshot_dt=0.1, eps_tau=0.01,
                             return_mode="full", verbose_every=1000):
        """Tau-leaping simulation (Kac or nearest, CPU or GPU)."""

        L = self.L
        use_gpu = self.use_gpu

        if use_gpu:
            xp = cp
            spin = cp.asarray(spin_init.astype(np.int8))
            kernel_k = build_kernel_k(L, self.Lx, self.Ly, epsilon, kernel_type, use_gpu=True)
            if kernel_type == "nearest":
                J00 = 0.0
            else:
                kern_real = cp.fft.ifft2(kernel_k).real
                kern_real /= kern_real.sum()
                J00 = kern_real[0, 0]
        else:
            xp = np
            spin = spin_init.copy().astype(np.int8)
            kernel_k = build_kernel_k(L, self.Lx, self.Ly, epsilon, kernel_type, use_gpu=False)
            if kernel_type == "nearest":
                J00 = 0.0
            else:
                kern_real = np.fft.ifft2(kernel_k).real
                kern_real /= kern_real.sum()
                J00 = kern_real[0, 0]

        t, next_snap = 0.0, 0.0
        time_list, Ms_list, Es_list = [0.0], [], []
        snap_list = [] if return_mode != "none" else None

        if use_gpu:
            Ms_list.append(float(spin.mean()))
            if kernel_type == "nearest":
                Es_list.append(self._calc_energy_nearest(spin, J0, h, use_gpu=True))
            else:
                Es_list.append(self._calc_energy_fft(spin, J0, h, kernel_k, J00, use_gpu=True))
            if return_mode == "full":
                snap_list.append(spin.get())
            elif return_mode == "coarse":
                snap_list.append(self._coarse_grain(spin).get())
        else:
            Ms_list.append(spin.mean())
            if kernel_type == "nearest":
                Es_list.append(self._calc_energy_nearest(spin, J0, h, use_gpu=False))
            else:
                Es_list.append(self._calc_energy_fft(spin, J0, h, kernel_k, J00, use_gpu=False))
            if return_mode == "full":
                snap_list.append(spin.copy())
            elif return_mode == "coarse":
                snap_list.append(self._coarse_grain(spin).copy())
        next_snap += snapshot_dt

        steps = 0
        while t < t_end:
            # rates
            if use_gpu:
                if kernel_type == "nearest":
                    hloc = h + J0 * nn_sum_gpu(spin)
                else:
                    spin_k = cp.fft.fft2(spin)
                    conv_val = cp.fft.ifft2(kernel_k * spin_k).real
                    hloc = h + J0 * (conv_val - J00 * spin)
                r = 1.0 / (1.0 + cp.exp(cp.clip(2.0 * beta * hloc * spin, -30, 30)))
                r_max = float(r.max())
            else:
                if kernel_type == "nearest":
                    hloc = h + J0 * nn_sum_cpu(spin)
                else:
                    conv_val = conv_spin(spin, kernel_k, use_gpu=False)
                    hloc = h + J0 * (conv_val - J00 * spin)
                arg = 2.0 * beta * hloc * spin
                arg = np.clip(arg, -30, 30)
                r = 1.0 / (1.0 + np.exp(arg))
                r_max = float(r.max())

            if r_max == 0.0:
                break

            tau = min(eps_tau / r_max, t_end - t)

            if use_gpu:
                flips = cp.random.poisson(r * tau)
                spin *= cp.where(flips % 2 == 1, -1, 1)
            else:
                flips = np.random.poisson(r * tau)
                spin *= np.where(flips % 2 == 1, -1, 1)

            t += tau

            while t >= next_snap and next_snap <= t_end:
                time_list.append(next_snap)
                if use_gpu:
                    Ms_list.append(float(spin.mean()))
                    if kernel_type == "nearest":
                        Es_list.append(self._calc_energy_nearest(spin, J0, h, use_gpu=True))
                    else:
                        Es_list.append(self._calc_energy_fft(spin, J0, h, kernel_k, J00, use_gpu=True))
                    if return_mode == "full":
                        snap_list.append(spin.get())
                    elif return_mode == "coarse":
                        snap_list.append(self._coarse_grain(spin).get())
                else:
                    Ms_list.append(spin.mean())
                    if kernel_type == "nearest":
                        Es_list.append(self._calc_energy_nearest(spin, J0, h, use_gpu=False))
                    else:
                        Es_list.append(self._calc_energy_fft(spin, J0, h, kernel_k, J00, use_gpu=False))
                    if return_mode == "full":
                        snap_list.append(spin.copy())
                    elif return_mode == "coarse":
                        snap_list.append(self._coarse_grain(spin).copy())
                next_snap += snapshot_dt

            steps += 1
            if verbose_every and (steps % verbose_every == 0):
                tag = "TauLeap-GPU" if use_gpu else "TauLeap-CPU"
                print(f"[{tag}] t={t:.3f}/{t_end}, steps={steps}", end="\r")

        times = np.array(time_list)
        Ms = np.array(Ms_list)
        Es = np.array(Es_list)
        if return_mode == "none":
            return times, Ms, Es
        else:
            snaps = np.stack(snap_list, axis=0)
            return times, Ms, Es, snaps
