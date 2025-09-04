import numba
import numpy as np
import math

@numba.njit(cache=True)
def _build_nn(L):
    up = np.empty(L*L, np.int64)
    dn = np.empty(L*L, np.int64)
    lf = np.empty(L*L, np.int64)
    rt = np.empty(L*L, np.int64)
    for x in range(L):
        xp = x+1 if x+1 < L else 0
        xm = x-1 if x-1 >= 0 else L-1
        for y in range(L):
            yp = y+1 if y+1 < L else 0
            ym = y-1 if y-1 >= 0 else L-1
            k  = x*L + y
            up[k] = xp*L + y
            dn[k] = xm*L + y
            rt[k] = x*L + yp
            lf[k] = x*L + ym
    return up, dn, lf, rt

@numba.njit(cache=True)
def _local_field(k, spin, up, dn, lf, rt, J, h):
    return J*(spin[up[k]] + spin[dn[k]] + spin[lf[k]] + spin[rt[k]]) + h

@numba.njit(cache=True)
def _rate(k, spin, up, dn, lf, rt, beta, J, h):
    Hi = _local_field(k, spin, up, dn, lf, rt, J, h)
    return np.exp(-beta * Hi * spin[k]) / (2.0 * np.cosh(beta * Hi))

@numba.njit(cache=True)
def _pick_site_by_rates(r, R, rng):
    thresh = rng * R
    s = 0.0
    for i in range(r.size):
        s += r[i]
        if s >= thresh:
            return i
    return r.size - 1

@numba.njit(cache=True)
def _glauber_ct_gillespie(spin, L, beta, J, h,
                          t_end, snapshot_dt,
                          times_out, snaps_out):
    N = L*L
    up, dn, lf, rt = _build_nn(L)

    r = np.empty(N, np.float64)
    for k in range(N):
        r[k] = _rate(k, spin, up, dn, lf, rt, beta, J, h)
    R = r.sum()

    t = 0.0
    next_snap_t = 0.0
    n_snaps = 0

    times_out[n_snaps] = 0.0
    snaps_out[n_snaps, :, :] = spin.reshape(L, L)
    n_snaps += 1
    next_snap_t += snapshot_dt

    step = 0
    while t < t_end and R > 0.0:
        u1 = np.random.rand()
        dt = -np.log(u1) / R
        t += dt

        while t >= next_snap_t and n_snaps < times_out.size:
            times_out[n_snaps] = next_snap_t
            snaps_out[n_snaps, :, :] = spin.reshape(L, L)
            n_snaps += 1
            next_snap_t += snapshot_dt
            if next_snap_t > t_end:
                break

        u2 = np.random.rand()
        i = _pick_site_by_rates(r, R, u2)

        spin[i] = -spin[i]

        idxs = np.empty(5, np.int64)
        idxs[0] = i
        idxs[1] = up[i]
        idxs[2] = dn[i]
        idxs[3] = lf[i]
        idxs[4] = rt[i]
        for j in range(5):
            k = idxs[j]
            old = r[k]
            r[k] = _rate(k, spin, up, dn, lf, rt, beta, J, h)
            R += r[k] - old
        step += 1
        if step % 10000 == 0:
            print("simulate progress:", t, step, n_snaps)

    if n_snaps < times_out.size and (n_snaps == 0 or times_out[n_snaps-1] < t_end):
        times_out[n_snaps] = min(t, t_end)
        snaps_out[n_snaps, :, :] = spin.reshape(L, L)
        n_snaps += 1
    return n_snaps

# Kac Kernel
def build_kac_csr_with_aff(L, R_phys, kernel="gaussian", sigma=0.1, include_self=False):
    spacing = 1.0 / L
    R_lat = int(np.ceil(R_phys / spacing))
    N = L*L

    nbr_ptr = [0]
    nbr_idx = []
    nbr_w   = []
    deg_aff = np.zeros(N, np.int64)

    for x in range(L):
        for y in range(L):
            weights = []
            indices = []
            for dx in range(-R_lat, R_lat+1):
                for dy in range(-R_lat, R_lat+1):
                    if dx==0 and dy==0 and not include_self:
                        continue
                    u = (x+dx) % L
                    v = (y+dy) % L
                    dist = spacing * math.sqrt(dx*dx + dy*dy)
                    if dist > R_phys:
                        continue
                    if kernel=="gaussian":
                        w = math.exp(-0.5*(dist/sigma)**2)
                    else:
                        w = 1.0
                    indices.append(u*L+v)
                    weights.append(w)
            s = sum(weights)
            weights = [w/s for w in weights]
            for j,w in zip(indices, weights):
                nbr_idx.append(j)
                nbr_w.append(w)
                deg_aff[j] += 1
            nbr_ptr.append(len(nbr_idx))

    nbr_ptr = np.array(nbr_ptr, dtype=np.int64)
    nbr_idx = np.array(nbr_idx, dtype=np.int64)
    nbr_w   = np.array(nbr_w, dtype=np.float64)

    # 反向邻接
    aff_ptr = np.zeros(N+1, np.int64)
    aff_ptr[1:] = np.cumsum(deg_aff)
    M2 = aff_ptr[-1]
    aff_idx = np.zeros(M2, np.int64)
    aff_w   = np.zeros(M2, np.float64)
    cursor = np.zeros(N, np.int64)

    for i in range(N):
        for p in range(nbr_ptr[i], nbr_ptr[i+1]):
            j = nbr_idx[p]
            pos = aff_ptr[j] + cursor[j]
            aff_idx[pos] = i
            aff_w[pos]   = nbr_w[p]
            cursor[j]   += 1

    return nbr_ptr, nbr_idx, nbr_w, aff_ptr, aff_idx, aff_w

@numba.njit(cache=True)
def calc_energy_kac(spin, L, J0, h, nbr_ptr, nbr_idx, nbr_w):
    N = L*L
    E = 0.0
    for i in range(N):
        si = spin.flat[i]
        for p in range(nbr_ptr[i], nbr_ptr[i+1]):
            j = nbr_idx[p]
            E += -0.5 * J0 * nbr_w[p] * si * spin.flat[j]
    E += -h * np.sum(spin)
    return E / N

# main class
class Glauber2DIsingCT:
    def __init__(self, size):
        self.L = size
        self.area = self.L * self.L

    def _calc_energy(self, spin, J=1.0, h=0.0):
        R = (np.roll(spin, 1, 0) + np.roll(spin, -1, 0) +
             np.roll(spin, 1, 1) + np.roll(spin, -1, 1))
        return (-J*np.sum(R*spin)/2.0 - h*np.sum(spin)) / self.area

    def _calc_energy_kac(self, spin, nbr_ptr, nbr_idx, nbr_w, J0=1.0, h=0.0):
        return calc_energy_kac(spin, self.L, J0, h, nbr_ptr, nbr_idx, nbr_w)

    def _calc_magnetization(self, spin):
        return np.sum(spin) / self.area

    def calc_temperature_ratios(self, beta, J=1.0, model="ising2d"):
        T = 1.0 / beta
        if model == "ising2d":
            Tc = 2.0 * J / np.log(1.0 + np.sqrt(2.0))
        elif model == "kac":
            Tc = J
        else:
            raise ValueError("Unknown model type")
        return T, Tc, T/Tc

    def simulate(self, spin_init, beta, J=1.0, h=0.0,
                 t_end=500.0, snapshot_dt=0.01, return_snapshots=True):
        spin = spin_init.reshape(-1).copy()
        max_snaps = int(np.ceil(t_end/snapshot_dt)) + 2
        times = np.zeros(max_snaps, dtype=np.float64)
        snaps = np.zeros((max_snaps, self.L, self.L), dtype=np.int8)
        print(f"[simulate] start continuous-time Glauber NN, L={self.L}, t_end={t_end}, dt={snapshot_dt}")
        n_snaps = _glauber_ct_gillespie(spin, self.L, beta, J, h,
                                        t_end, snapshot_dt,
                                        times, snaps)
        times = times[:n_snaps]
        snaps = snaps[:n_snaps]

        Ms = np.array([self._calc_magnetization(snaps[k]) for k in range(n_snaps)])
        Es = np.array([self._calc_energy(snaps[k], J, h) for k in range(n_snaps)])

        if return_snapshots:
            return times, Ms, Es, snaps
        else:
            return times, Ms, Es

    def simulate_kac(self, spin_init, beta, J0=1.0, h=0.0,
                     R=0.2, kernel="gaussian", sigma=0.1,
                     t_end=500.0, snapshot_dt=0.01, return_snapshots=True):
        spin = spin_init.copy()
        N = self.L * self.L
        print(f"[simulate_kac] start CT Glauber Kac, L={self.L}, R={R}, kernel={kernel}, sigma={sigma}, t_end={t_end}")

        # 构建邻接表
        nbr_ptr, nbr_idx, nbr_w, aff_ptr, aff_idx, aff_w = build_kac_csr_with_aff(
            self.L, R, kernel=kernel, sigma=sigma)
        print(f"  [info] avg degree: {nbr_ptr[-1]/N:.2f}, max degree: {max(nbr_ptr[i+1]-nbr_ptr[i] for i in range(N))}")

        # 初始化局域场
        hloc = np.zeros(N, dtype=np.float64)
        for i in range(N):
            s = 0.0
            for p in range(nbr_ptr[i], nbr_ptr[i+1]):
                j = nbr_idx[p]
                s += nbr_w[p] * spin.flat[j]
            hloc[i] = h + J0 * s

        # 初始化速率
        r = np.zeros(N, dtype=np.float64)
        for i in range(N):
            r[i] = np.exp(-beta * hloc[i] * spin.flat[i]) / (2.0 * np.cosh(beta * hloc[i]))
        Rtot = r.sum()
        print(f"[simulate_kac] initial total rate Rtot={Rtot:.3e}")
        max_snaps = int(np.ceil(t_end/snapshot_dt)) + 2
        times = np.zeros(max_snaps, np.float64)
        snaps = np.zeros((max_snaps, self.L, self.L), np.int8)

        t = 0.0
        next_snap = 0.0
        n_snaps = 0
        times[n_snaps] = 0.0
        snaps[n_snaps] = spin
        n_snaps += 1
        next_snap += snapshot_dt

        rng = np.random.default_rng()
        while t < t_end and Rtot > 0:
            u1 = rng.random()
            dt = -np.log(u1) / Rtot
            t += dt

            while t >= next_snap and n_snaps < times.size:
                times[n_snaps] = next_snap
                snaps[n_snaps] = spin
                n_snaps += 1
                next_snap += snapshot_dt
                if next_snap > t_end:
                    break

            # Gillespie: 按速率选择翻转点
            u2 = rng.random() * Rtot
            s = 0.0
            i = 0
            step = 0
            for i in range(N):
                s += r[i]
                if s >= u2:
                    break

            old_s = spin.flat[i]
            spin.flat[i] = -old_s
            delta = spin.flat[i] - old_s

            # 增量更新：翻转点 i 影响的所有中心
            for p in range(aff_ptr[i], aff_ptr[i+1]):
                y = aff_idx[p]
                hloc[y] += J0 * aff_w[p] * delta
                old = r[y]
                r[y] = np.exp(-beta * hloc[y] * spin.flat[y]) / (2.0*np.cosh(beta*hloc[y]))
                Rtot += r[y] - old

            # 自身速率也要更新
            old = r[i]
            r[i] = np.exp(-beta * hloc[i] * spin.flat[i]) / (2.0*np.cosh(beta*hloc[i]))
            Rtot += r[i] - old

            step += 1
            if step % 10000 == 0:
                print(f"  [info] t={t:.3f}, Rtot={Rtot:.3e}, next_snap={next_snap:.3f}, n_snaps={n_snaps}")

        times = times[:n_snaps]
        snaps = snaps[:n_snaps]

        Ms = np.array([self._calc_magnetization(snaps[k]) for k in range(n_snaps)])
        Es = np.array([self._calc_energy_kac(snaps[k], nbr_ptr, nbr_idx, nbr_w, J0, h)
                       for k in range(n_snaps)])

        print(f"[simulate_kac] done, collected {n_snaps} snapshots")
        print(f"[simulate_kac] final magnetization={Ms[-1]:.4f}, energy={Es[-1]:.4f}")

        if return_snapshots:
            return times, Ms, Es, snaps
        else:
            return times, Ms, Es
