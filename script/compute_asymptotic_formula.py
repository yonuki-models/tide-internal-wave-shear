import numpy as np
from scipy.special import gamma
from scipy.special import jv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm

rc('font', family='Times New Roman')
plt.rcParams['mathtext.fontset'] = 'cm'
fs = 14
eps_i = 1.e-10j


def Bessel_I(mu, z, N):
    I = 1.e0
    eps = np.sign(np.real(z)) * eps_i
    for m in range(1, N):
        a = 1
        for j in range(1, m+1):
            a = a * (z + eps)**2 / (4 * j * (j + mu))
        I = I + a
    I = (z / 2 + eps)**mu / gamma(mu + 1) * I
    return I


def D_func(U0, U1, k, omega, N):
    nu = (1 / (U1 - U0)**2 - 0.25)**0.5
    Z0 = (k * U0 - omega) / (U1 - U0)
    Z1 = (k * U1 - omega) / (U1 - U0)
    return np.sqrt(Z1 * Z0 + eps_i) * (Bessel_I(- 1j*nu, Z1, N)
            * Bessel_I(1j*nu, Z0, N)
            - Bessel_I(1j*nu, Z1, N)
            * Bessel_I(-1j*nu, Z0, N))


N = 100
Nx = 200
Nz = 100
z_max = 0.62
z_min = 0.38
dz = (z_max - z_min) / (Nz - 1)

xa = np.linspace(2, 20, Nx)
za = np.linspace(z_min, z_max, Nz)
x, z = np.meshgrid(xa, za)

OT = 1.  # omega_T
n = 1
U0 = 0.
U1 = 0.5
UT = 0.1
S = U1 - U0

U = U0 + S * z
nu = (1 / (U1 - U0)**2 - 0.25)**0.5

h = 1.

k = OT / U
Z0 = (k * U0 - OT) / (U1 - U0)
Z1 = (k * U1 - OT) / (U1 - U0)
alpha = (- np.sqrt(k * Z1) * Bessel_I(1j * nu, Z1, N) / (gamma(- 1j * nu + 1))
         * (k/2 + eps_i)**(- 1j * nu))

J = jv(n, n * UT / U)
D = D_func(U0, U1, n * OT / U, n * OT, N)

term1 = ((U**2 / (n * OT * S))**(0.5 - 1j * nu)
         * 1j**(-0.5) * np.exp(- np.pi * nu / 2) * alpha * x**(1j * nu)
         / gamma(- 0.5 + 1j * nu))
term2 = ((U**2 / (n * OT * S))**(0.5 + 1j * nu)
         * 1j**(-0.5) * np.exp(np.pi * nu / 2) * alpha.conjugate() * x**(- 1j * nu)
         / gamma(- 0.5 - 1j * nu))
psi = ((term1 + term2) * (U0 - U) * h * J * x**(-1.5)
       * np.exp(1j * n * OT * x / U) / D)

fig = plt.figure(figsize=(6, 6))
fs = 14
nc = 40
y_max = 0.6
y_min = 0.4

psi_r = np.real(psi)
psi_max = np.abs(psi_r).max()
clevels = np.linspace(-psi_max, psi_max, nc, endpoint=True)
ax1 = fig.add_subplot(311)
ax1.set_title(r"(a) $\psi$", fontsize=fs, loc='left')
ax1.set_xlabel(r"$x$", fontsize=fs, labelpad=10)
ax1.set_ylabel(r"$z$", fontsize=fs, labelpad=8)
ax1.set_ylim(y_min, y_max)
cf = ax1.contourf(xa, za, psi_r, levels=clevels, cmap=cm.seismic)

u = np.real(psi[1:,:] - psi[:-1, :]) / dz
u_max = np.abs(u).max()
clevels = np.linspace(-u_max, u_max, nc, endpoint=True)
ax2 = fig.add_subplot(312)
ax2.set_title(r"(b) $\psi_z$", fontsize=fs, loc='left')
ax2.set_xlabel(r"$x$", fontsize=fs, labelpad=10)
ax2.set_ylabel(r"$z$", fontsize=fs, labelpad=8)
ax2.set_ylim(y_min, y_max)
cf = ax2.contourf(xa, za[1:] + dz/2, u, levels=clevels, cmap=cm.seismic)

zeta = np.real(u[1:,:] - u[:-1, :]) / dz
zeta_max = np.abs(zeta).max()
clevels = np.linspace(-zeta_max, zeta_max, nc, endpoint=True)
ax3 = fig.add_subplot(313)
ax3.set_title(r"(c) $\psi_{zz}$", fontsize=fs, loc='left')
ax3.set_xlabel(r"$x$", fontsize=fs, labelpad=10)
ax3.set_ylabel(r"$z$", fontsize=fs, labelpad=8)
ax3.set_ylim(y_min, y_max)
cf = ax3.contourf(xa, za[1:-1], zeta, levels=clevels, cmap=cm.seismic)

fig.tight_layout()

file_name = 'algebraic_evolution'
plt.savefig(file_name + '.eps');
plt.savefig(file_name + '.png');
