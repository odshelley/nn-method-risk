"""Finite-volume forward Kolmogorov (Fokker-Planck) operator for the Heston-type LSV density.

Density phi(t, x, v) of (ln S, V) with dX = -1/2 L^2 V dt + L sqrt(V) dW,
dV = kappa (theta - V) dt + xi sqrt(V) dB, d<W,B> = rho dt. Written in the scaled variable
phi = v^beta p, beta = 2 kappa theta / xi^2 - 1 (Cozma, Mariapragassam, Reisinger,
arXiv:1701.06001, Cor. 5), which is bounded at v = 0 even when the Feller condition fails.
The zero-flux condition of their Theorem 4 at v = 0 then holds identically.

Unknowns are cell masses m_ij (integral of phi over the cell); p is constant per cell.
Scharfetter-Gummel fluxes for the drift-diffusion parts, a conservative corner-average
stencil for the mixed derivative, ghost cells with p = 0 at the outer x faces and at v_max.
See docs/superpowers/specs/2026-09-08-pde-reference-design.md.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from ..simulate.dynamics import HestonParams


def bernoulli(w):
    """B(w) = w / (exp(w) - 1), B(0) = 1. Safe for any float w."""
    w = np.asarray(w, dtype=float)
    out = np.empty_like(w)
    small = np.abs(w) < 1e-8
    big = w > 700.0
    mid = ~(small | big)
    out[small] = 1.0 - 0.5 * w[small]
    out[big] = 0.0
    out[mid] = w[mid] / np.expm1(w[mid])
    return out if out.ndim else float(out)


def _faces(centres, first=None):
    f = np.empty(len(centres) + 1)
    f[1:-1] = 0.5 * (centres[1:] + centres[:-1])
    f[0] = first if first is not None else centres[0] - (f[1] - centres[0])
    f[-1] = centres[-1] + (centres[-1] - f[-2])
    return f


def _one_minus_pow(r, e):
    """1 - r**e, accurately, for r in [0, 1) and e > 0."""
    with np.errstate(divide="ignore"):
        lr = np.log(r)
    return np.where(r > 0, -np.expm1(e * lr), 1.0)


class FokkerPlanck:
    """Grid geometry, beta-weights and operator assembly for one (x_grid, v_grid, params)."""

    def __init__(self, x_grid, v_grid, params):
        self.params = hp = HestonParams.from_dict(params)
        self.x = x = np.asarray(x_grid, dtype=float)
        self.v = v = np.asarray(v_grid, dtype=float)
        if not (np.all(np.diff(x) > 0) and np.all(np.diff(v) > 0) and v[0] > 0):
            raise ValueError("x_grid and v_grid must be increasing and v_grid positive")
        self.nx, self.nv = len(x), len(v)
        self.x_faces = _faces(x)
        self.v_faces = Vf = _faces(v, first=0.0)
        self.dx = np.diff(self.x_faces)
        self.dv = np.diff(Vf)
        # beta weights: W_j = int_cell v^beta, expressed through ratios only
        b1 = 2.0 * hp.kappa * hp.theta / hp.xi ** 2   # beta + 1 > 0
        self.beta = b1 - 1.0
        r = Vf[:-1] / Vf[1:]
        d1 = _one_minus_pow(r, b1)
        self.gp = b1 / d1                                   # V_{j+1}^(b1) / W_j
        self.gm = b1 * (1.0 - d1) / d1                      # V_j^(b1) / W_j
        self.mean_v = b1 / (b1 + 1.0) * Vf[1:] * _one_minus_pow(r, b1 + 1.0) / d1   # W'_j / W_j
        self.mean_v2 = b1 / (b1 + 2.0) * Vf[1:] ** 2 * _one_minus_pow(r, b1 + 2.0) / d1
        # spacings between cell centres, with mirrored ghosts at the outer faces
        hx = np.empty(self.nx + 1)
        hx[1:-1] = np.diff(x)
        hx[0], hx[-1] = self.dx[0], self.dx[-1]
        self.hx = hx
        hv = np.empty(self.nv + 1)
        hv[1:-1] = np.diff(v)
        hv[0], hv[-1] = np.nan, 2.0 * (Vf[-1] - v[-1])   # face 0 carries no flux
        self.hv = hv

    # ----- state helpers -------------------------------------------------------------
    @property
    def shape(self):
        return (self.nx, self.nv)

    def initial(self, x0, v0):
        """Unit mass split bilinearly over the four cells around (x0, v0)."""
        m = np.zeros(self.shape)
        ix, wx = self._bracket(self.x, x0)
        jv, wv = self._bracket(self.v, v0)
        for i, a in ((ix, 1 - wx), (ix + 1, wx)):
            for j, b in ((jv, 1 - wv), (jv + 1, wv)):
                if a * b:
                    m[i, j] += a * b
        return m

    @staticmethod
    def _bracket(grid, val):
        if not grid[0] <= val <= grid[-1]:
            raise ValueError(f"{val} outside the grid [{grid[0]}, {grid[-1]}]")
        i = min(int(np.searchsorted(grid, val, side="right")) - 1, len(grid) - 2)
        w = (val - grid[i]) / (grid[i + 1] - grid[i])
        return i, float(w)

    def mass(self, m):
        return float(m.sum())

    def marginal(self, m):
        """Cell-averaged marginal density of X, per unit x."""
        return m.sum(axis=1) / self.dx

    def cond_mean_v(self, m, rel_floor=1e-12):
        """E[V | X = x_i]; cells with negligible marginal mass take the nearest defined value."""
        num = m @ self.mean_v
        den = m.sum(axis=1)
        ok = den > rel_floor * den.max()
        f = np.full(self.nx, np.nan)
        f[ok] = num[ok] / den[ok]
        if not ok.all():
            idx = np.arange(self.nx)
            f = np.interp(idx, idx[ok], f[ok])
        return f

    # ----- operator ------------------------------------------------------------------
    def operator(self, L):
        """Sparse A with dm/dt = A m for the leverage L (scalar or per x-cell)."""
        nx, nv, hp = self.nx, self.nv, self.params
        L = np.broadcast_to(np.asarray(L, dtype=float), (nx,)).copy()
        Lf = np.empty(nx + 1)
        Lf[1:-1], Lf[0], Lf[-1] = 0.5 * (L[:-1] + L[1:]), L[0], L[-1]
        L2dx = L ** 2 / self.dx
        Ldx = L / self.dx
        inv_dx = 1.0 / self.dx
        gp, gm, mean_v = self.gp, self.gm, self.mean_v
        cross = -0.5 * hp.rho * hp.xi
        rows, cols, vals = [], [], []

        def emit(row_i, row_j, sign, cell_i, cell_j, val):
            ok = ((row_i >= 0) & (row_i < nx) & (row_j >= 0) & (row_j < nv)
                  & (cell_i >= 0) & (cell_i < nx) & (cell_j >= 0) & (cell_j < nv))
            rows.append(np.ravel_multi_index((row_i[ok], row_j[ok]), self.shape))
            cols.append(np.ravel_multi_index((cell_i[ok], cell_j[ok]), self.shape))
            vals.append(sign * val[ok])

        def through_x_face(k, j, cell_i, cell_j, val):
            # flux through x-face k in row j: +into cell k, -out of cell k-1
            emit(k, j, +1.0, cell_i, cell_j, val)
            emit(k - 1, j, -1.0, cell_i, cell_j, val)

        def through_v_face(i, l, cell_i, cell_j, val):
            emit(i, l, +1.0, cell_i, cell_j, val)
            emit(i, l - 1, -1.0, cell_i, cell_j, val)

        # --- x faces k = 0..nx, rows j -------------------------------------------
        K, J = np.meshgrid(np.arange(nx + 1), np.arange(nv), indexing="ij")
        hx = self.hx[K]
        Bp, Bm = bernoulli(hx), bernoulli(-hx)
        Lc = np.pad(L2dx, (1, 1))          # index k -> cell k-1 at [k], cell k at [k+1]
        through_x_face(K, J, K - 1, J, mean_v[J] * Bp / (2 * hx) * Lc[K])
        through_x_face(K, J, K, J, -mean_v[J] * Bm / (2 * hx) * Lc[K + 1])
        # cross: c Lf [C_{k,j+1} - C_{k,j}], C_{k,J} = 1/4 [gp_{J-1} p-row J-1 + gm_J p-row J]
        c = cross * Lf[K]
        idxp = np.pad(inv_dx, (1, 1))
        for di in (-1, 0):
            w = 0.25 * idxp[K + 1 + di]
            through_x_face(K, J, K + di, J, c * w * gp[J])                       # C_{k,j+1}, row j
            through_x_face(K, J, K + di, J + 1, c * w * np.pad(gm, (0, 1))[J + 1])  # C_{k,j+1}, row j+1
            through_x_face(K, J, K + di, J - 1, -c * w * gp[np.maximum(J - 1, 0)] * (J >= 1))
            through_x_face(K, J, K + di, J, -c * w * gm[J])                      # C_{k,j}, row j
        # --- v faces l = 1..nv, columns i ------------------------------------------
        I, Lv = np.meshgrid(np.arange(nx), np.arange(1, nv + 1), indexing="ij")
        hv = self.hv[Lv]
        D = 0.5 * hp.xi ** 2
        w = -2.0 * hp.kappa * hv / hp.xi ** 2
        gmp = np.pad(gm, (0, 1))
        through_v_face(I, Lv, I, Lv - 1, D / hv * bernoulli(-w) * gp[Lv - 1])
        through_v_face(I, Lv, I, Lv, -D / hv * bernoulli(w) * gmp[Lv])
        # cross: c [Cv_{i+1,l} - Cv_{i,l}], Cv_{I,l} = 1/4 sum over cells (I-1|I, l-1|l) of V_l^b1 L p
        Ldxp = np.pad(Ldx, (1, 1))
        for face_off, sgn in ((1, +1.0), (0, -1.0)):
            for di in (-1, 0):
                ci = I + face_off + di
                wgt = sgn * cross * 0.25 * Ldxp[ci + 1]
                through_v_face(I, Lv, ci, Lv - 1, wgt * gp[Lv - 1])
                through_v_face(I, Lv, ci, Lv, wgt * gmp[Lv])
        n = nx * nv
        A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n)).tocsc()
        A.sum_duplicates()
        return A

    # ----- time stepping -------------------------------------------------------------
    def advance(self, m, A, dt, n_sub, *, first=False, scheme="cn"):
        """Advance the masses by dt in n_sub theta-steps under the frozen operator A."""
        if scheme not in ("cn", "be"):
            raise ValueError(f"unknown scheme {scheme!r}")
        if scheme == "be" or first:
            theta, n = 1.0, (2 * n_sub if scheme == "cn" else n_sub)
        else:
            theta, n = 0.5, n_sub
        tau = dt / n
        n_dof = A.shape[0]
        eye = sp.identity(n_dof, format="csc")
        lu = splu((eye - theta * tau * A).tocsc())
        explicit = None if theta == 1.0 else (eye + (1.0 - theta) * tau * A).tocsr()
        y = m.reshape(-1)
        for _ in range(n):
            rhs = y if explicit is None else explicit @ y
            y = lu.solve(rhs)
        return y.reshape(self.shape)
