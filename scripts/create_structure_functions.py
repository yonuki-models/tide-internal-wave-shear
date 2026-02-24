"""Create structure function figures."""

from __future__ import annotations

import mpmath as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, rc

rc("font", family="Times New Roman")
plt.rcParams["mathtext.fontset"] = "cm"

mp.mp.dps = 15
FS = 14
LEVELS = 64
R = 1.0e-6


def _besseli(order: complex, z: complex) -> mp.mpc:
    return mp.besseli(mp.mpc(order), mp.mpc(complex(z)))


def II(mu: complex, z1: object, z2: object) -> np.ndarray:
    z1a = np.asarray(z1, dtype=np.complex128)
    z2a = np.asarray(z2, dtype=np.complex128)
    z1b, z2b = np.broadcast_arrays(z1a, z2a)
    mu_mp = mp.mpc(mu)
    out = np.empty(z1b.shape, dtype=np.complex128)
    it = np.nditer(z1b, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        a = _besseli(-mu_mp, z1b[idx]) * _besseli(mu_mp, z2b[idx])
        b = _besseli(mu_mp, z1b[idx]) * _besseli(-mu_mp, z2b[idx])
        out[idx] = complex(1j * (a - b))
        it.iternext()
    return out


def _sqrt_cont(p: np.ndarray) -> np.ndarray:
    pa = np.asarray(p, dtype=np.complex128)
    mag = np.abs(pa)
    ang = np.unwrap(np.angle(pa))
    return np.sqrt(mag) * np.exp(0.5j * ang)


def structure_function(
    nu: float,
    u0: float,
    u1: float,
    k: float,
    omega: complex,
    z: np.ndarray,
) -> np.ndarray:
    z1 = (k * u1 - omega) / (u1 - u0)
    u = u1 * z + u0 * (1.0 - z)
    zz = (k * u - omega) / (u1 - u0)
    s = _sqrt_cont(z1 * zz)
    return s * II(1j * nu, z1, zz)


def structure_function_normal(
    nu: float,
    u0: float,
    u1: float,
    k: float,
    omega: complex,
    z: np.ndarray,
) -> np.ndarray:
    f = structure_function(nu, u0, u1, k, omega, z)
    return f / np.abs(f).max()


def _plot_case(
    ax: plt.Axes,
    nu: float,
    u0: float,
    u1: float,
    k: float,
    omega: complex,
) -> None:
    wl = 2.0 * np.pi / np.abs(k)
    x_max = 4.0 * wl
    z = np.linspace(0.0, 1.0, 1000)
    x = np.linspace(0.0, x_max, 200)
    psi_1d = structure_function_normal(nu, u0, u1, k, omega, z)
    psi = np.real(np.outer(psi_1d, np.exp(1j * k * x)))
    title = r"$k={:.1f}, \sigma={:.1f}$".format(k, float(np.real(omega)))
    ax.set_title(title, fontsize=FS, loc="left")
    ax.set_xlabel(r"$x$", fontsize=FS, labelpad=10)
    ax.set_ylabel(r"$z$", fontsize=FS, labelpad=8)
    ax.contourf(x, z, psi, levels=LEVELS, cmap=cm.seismic)


def main() -> None:
    buoy = 1.0
    u0, u1 = 0.0, 0.15
    ri = buoy**2 / (u0 - u1) ** 2
    nu = (ri - 0.25) ** 0.5

    fig = plt.figure(figsize=(8, 2.5))
    cases = [(-5.0, 1.5), (2.5, 0.6), (5.0, 0.5)]
    for i, (k, omega) in enumerate(cases, start=1):
        ax = fig.add_subplot(1, 3, i)
        _plot_case(ax, nu, u0, u1, k, omega + 1j * R)

    fig.tight_layout()
    plt.savefig("structure_functions.eps")
    plt.savefig("structure_functions.png")


if __name__ == "__main__":
    main()
