"""Compute asymptotic formula and plot algebraic evolution (Figure 6)."""

from __future__ import annotations

import mpmath as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, rc
from scipy.special import jv

rc("font", family="Times New Roman")
plt.rcParams["mathtext.fontset"] = "cm"

mp.mp.dps = 15
FS = 14
R = 1.0e-6


def _besseli(order: complex, z: complex) -> complex:
    return complex(mp.besseli(mp.mpc(order), mp.mpc(complex(z))))


def _gamma(z: complex) -> complex:
    return complex(mp.gamma(mp.mpc(z)))


def _mp_apply1(func, a: object) -> np.ndarray:
    aa = np.asarray(a, dtype=np.complex128)
    out = np.empty(aa.shape, dtype=np.complex128)
    it = np.nditer(aa, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        out[idx] = func(aa[idx])
        it.iternext()
    return out


def I(order: complex, z: object) -> np.ndarray:
    return _mp_apply1(lambda zz: _besseli(order, zz), z)


def _sqrt_cont(p: np.ndarray) -> np.ndarray:
    pa = np.asarray(p, dtype=np.complex128)
    mag = np.abs(pa)
    ang = np.unwrap(np.angle(pa), axis=0)
    return np.sqrt(mag) * np.exp(0.5j * ang)


def main() -> None:
    n = 1
    nx, nz = 200, 100

    z_min, z_max = 0.38, 0.62
    dz = (z_max - z_min) / (nz - 1)

    xa = np.linspace(2.0, 20.0, nx)
    za = np.linspace(z_min, z_max, nz)
    x, z = np.meshgrid(xa, za)

    ot = 1.0
    on_c = n * ot + 1j * R
    u0, u1 = 0.0, 0.5
    ut = 0.1
    s = u1 - u0

    u = u0 + s * z
    nu = (1.0 / (u1 - u0) ** 2 - 0.25) ** 0.5
    
    k = n * on_c / u
    z0 = (k * u0 - on_c) / (u1 - u0)
    z1 = (k * u1 - on_c) / (u1 - u0)

    i_p_z1 = I(1j * nu, z1)
    i_m_z1 = I(-1j * nu, z1)
    i_p_z0 = I(1j * nu, z0)
    i_m_z0 = I(-1j * nu, z0)

    g1 = _gamma(1.0 - 1j * nu)
    alpha = -1j * _sqrt_cont(k * z1) * i_p_z1 / g1
    alpha *= (k / 2.0) ** (-1j * nu)

    term = i_m_z1 * i_p_z0 - i_p_z1 * i_m_z0
    dden = 1j * _sqrt_cont(z1 * z0) * term

    jj = jv(n, n * ut / u)

    g_p = _gamma(-0.5 + 1j * nu)
    g_m = _gamma(-0.5 - 1j * nu)

    t1 = (u**2 / (on_c * s)) ** (0.5 - 1j * nu)
    t1 *= np.exp(-np.pi * nu / 2.0) * alpha
    t1 *= x ** (1j * nu) / g_p

    t2 = (u**2 / (on_c * s)) ** (0.5 + 1j * nu)
    t2 *= np.exp(np.pi * nu / 2.0) * np.conjugate(alpha)
    t2 *= x ** (-1j * nu) / g_m

    psi = (t1 + t2) * (u0 - u) * jj
    psi *= 1j ** (-0.5) * x ** (-1.5)
    psi *= np.exp(1j * on_c * x / u) / dden

    fig = plt.figure(figsize=(6, 6))
    nc = 40
    y_min, y_max = 0.4, 0.6

    psi_r = np.real(psi)
    psi_max = np.abs(psi_r).max()
    clevels = np.linspace(-psi_max, psi_max, nc, endpoint=True)

    ax1 = fig.add_subplot(311)
    ax1.set_title(r"(a) $\psi$", fontsize=FS, loc="left")
    ax1.set_xlabel(r"$x$", fontsize=FS, labelpad=10)
    ax1.set_ylabel(r"$z$", fontsize=FS, labelpad=8)
    ax1.set_ylim(y_min, y_max)
    ax1.contourf(xa, za, psi_r, levels=clevels, cmap=cm.seismic)

    uvel = np.real(psi[1:, :] - psi[:-1, :]) / dz
    u_max = np.abs(uvel).max()
    clevels = np.linspace(-u_max, u_max, nc, endpoint=True)

    ax2 = fig.add_subplot(312)
    ax2.set_title(r"(b) $\psi_z$", fontsize=FS, loc="left")
    ax2.set_xlabel(r"$x$", fontsize=FS, labelpad=10)
    ax2.set_ylabel(r"$z$", fontsize=FS, labelpad=8)
    ax2.set_ylim(y_min, y_max)
    ax2.contourf(xa, za[1:] + dz / 2.0, uvel, levels=clevels, cmap=cm.seismic)

    zeta = np.real(uvel[1:, :] - uvel[:-1, :]) / dz
    zeta_max = np.abs(zeta).max()
    clevels = np.linspace(-zeta_max, zeta_max, nc, endpoint=True)

    ax3 = fig.add_subplot(313)
    ax3.set_title(r"(c) $\psi_{zz}$", fontsize=FS, loc="left")
    ax3.set_xlabel(r"$x$", fontsize=FS, labelpad=10)
    ax3.set_ylabel(r"$z$", fontsize=FS, labelpad=8)
    ax3.set_ylim(y_min, y_max)
    ax3.contourf(xa, za[1:-1], zeta, levels=clevels, cmap=cm.seismic)

    fig.tight_layout()
    plt.savefig("algebraic_evolution.eps")
    plt.savefig("algebraic_evolution.png")


if __name__ == "__main__":
    main()