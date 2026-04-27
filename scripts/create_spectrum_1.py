"""Create spectrum figure (spectrum_1)."""

from __future__ import annotations

import mpmath as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

rc("font", family="Times New Roman")
plt.rcParams["mathtext.fontset"] = "cm"
mp.mp.dps = 15


def _besseli(order: complex, z: complex) -> mp.mpc:
    return mp.besseli(mp.mpc(order), mp.mpc(complex(z)))


def II(mu: complex, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    if z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have the same shape")
    mu_mp = mp.mpc(mu)
    out = np.empty(z1.shape, dtype=np.complex128)
    it = np.nditer(z1, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        a = _besseli(-mu_mp, z1[idx]) * _besseli(mu_mp, z2[idx])
        b = _besseli(mu_mp, z1[idx]) * _besseli(-mu_mp, z2[idx])
        out[idx] = complex(1j * (a - b))
        it.iternext()
    return out


def _k_axis(k_max: float, nk: int) -> np.ndarray:
    kp = np.logspace(-2.0, np.log10(k_max), nk)
    return np.concatenate((-kp[::-1], np.array([0.0]), kp))


def _omega_k(
    k_axis: np.ndarray,
    no: int,
    omega_max: float,
    u0: float,
    u1: float,
) -> tuple[np.ndarray, np.ndarray]:
    off = np.logspace(-3.0, np.log10(omega_max), no)
    omega = np.empty((k_axis.size, no), dtype=float)
    for j, kj in enumerate(k_axis):
        omega[j, :] = kj * (u0 if kj < 0 else u1) + off
    k = np.repeat(k_axis[:, None], no, axis=1)
    return omega, k


def main() -> None:
    buoy = 1.0
    u0, u1 = 0.0, 0.15
    ri = buoy**2 / (u0 - u1) ** 2
    nu = (ri - 0.25) ** 0.5

    nk, no = 100, 300
    k_max, omega_max = 10.0, buoy

    k_axis = _k_axis(k_max, nk)
    omega, k = _omega_k(k_axis, no, omega_max, u0, u1)

    z0 = (k * u0 - omega) / (u1 - u0)
    z1 = (k * u1 - omega) / (u1 - u0)
    func = II(1j * nu, z1, z0)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.grid()

    nk0 = nk
    ax.contour(
        k[nk0 + 1 :, :],
        omega[nk0 + 1 :, :],
        np.real(func[nk0 + 1 :, :]),
        colors=["r"],
        levels=[0],
    )
    ax.contour(
        k[:nk0, :],
        omega[:nk0, :],
        np.real(func[:nk0, :]),
        colors=["b"],
        levels=[0],
    )
    ax.contour(
        -k[:nk0, :],
        -omega[:nk0, :],
        np.real(func[:nk0, :]),
        colors=["b"],
        levels=[0],
    )

    ax.fill_between(
        k_axis[nk0:],
        k_axis[nk0:] * u0,
        k_axis[nk0:] * u1,
        color="purple",
        alpha=0.22,
    )
    ax.scatter(
        [-5.0, 2.5, 5.0],
        [1.5, 0.6, 0.5],
        marker="s",
        color="k",
        zorder=20,
    )

    ax.set_xlim([-k_max, k_max])
    ax.set_ylim([0.0, k_max * u1 + omega_max])
    ax.set_xlabel(r"$k$", fontsize=20, labelpad=10)
    ax.set_ylabel(r"$\sigma$", fontsize=20, labelpad=8)

    fig.tight_layout()
    plt.savefig("spectrum_1.pdf")


if __name__ == "__main__":
    main()
