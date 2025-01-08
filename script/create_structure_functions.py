import numpy as np
from scipy.special import gamma
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm

rc('font', family='Times New Roman')
plt.rcParams['mathtext.fontset'] = 'cm'
fs = 14
levels = 64
eps_i = 1.e-10j


def Bessel_I(nu, z, N):
    I = 1.e0
    eps = np.sign(np.real(z)) * eps_i
    for m in range(1, N):
        a = 1
        for j in range(1, m+1):
            a = a * (z + eps)**2 / (4 * j * (j + nu))            
        I = I + a
    I = (z / 2 + eps)**nu / gamma(nu + 1) * I
    return I


def II(nu, z1, z2, N):
    return (Bessel_I(nu, z1, N)
            * Bessel_I(-nu, z2, N)
            - Bessel_I(-nu, z1, N)
            * Bessel_I(nu, z2, N))


def D_func(U0, U1, k, omega, N):
    nu = 1j * (1 / (U1 - U0)**2 - 0.25)**0.5
    Z0 = (k * U0 - omega) / (U1 - U0)
    Z1 = (k * U1 - omega) / (U1 - U0)
    return np.sqrt(Z1 * Z0 + eps_i) * (Bessel_I(- nu, Z1, N)
            * Bessel_I(nu, Z0, N)
            - Bessel_I(nu, Z1, N)
            * Bessel_I(-nu, Z0, N))


def structure_function(U0, U1, k, omega, z, N):
    nu = 1j * (1 / (U1 - U0)**2 - 0.25)**0.5
    Z1 = (k * U1 - omega) / (U1 - U0)
    U = U1 * z + U0 *(1 - z)
    Z = (k * U - omega) / (U1 - U0)
    return ((Z + eps_i)**0.5 * II(nu, Z1, Z, N))


def structure_function_normal(U0, U1, k, omega, z, N):
    F = structure_function(U0, U1, k, omega, z, N)
    return F / np.abs(F).max()


buoyancy_frequency = 1.
U0 = 0
U1 = 0.15
Ri = buoyancy_frequency / (U0 - U1)**2
nu = 1j * (Ri - 0.25)**0.5

N = 100

fig = plt.figure(figsize=(8, 2.5))

k, omega = -5, 1.5
wave_length = 2 * np.pi / np.abs(k)
x_max = 4 * wave_length
z = np.linspace(0, 1, 1000)
x = np.linspace(0, x_max, 200)
psi_1d = structure_function_normal(U0, U1, k, omega, z, N)
psi =  np.real(np.outer(psi_1d, np.exp(1j * k * x)))
ax1 = fig.add_subplot(131)
ax1.set_title(r"$k={:.1f}, \ \sigma={:.1f}$".format(k, omega),
              fontsize=fs, loc='left')
ax1.set_xlabel(r"$x$", fontsize=fs, labelpad=10)
ax1.set_ylabel(r"$z$", fontsize=fs, labelpad=8)
cf = ax1.contourf(x, z, psi, levels=levels, cmap=cm.seismic)

k, omega = 2.5, 0.6
wave_length = 2 * np.pi / np.abs(k)
x_max = 4 * wave_length
z = np.linspace(0, 1, 1000)
x = np.linspace(0, x_max, 200)
psi_1d = structure_function_normal(U0, U1, k, omega, z, N)
psi =  np.real(np.outer(psi_1d, np.exp(1j * k * x)))
ax2 = fig.add_subplot(132)
ax2.set_title(r"$k={:.1f}, \ \sigma={:.1f}$".format(k, omega),
              fontsize=fs, loc='left')
ax2.set_xlabel(r"$x$", fontsize=fs, labelpad=10)
ax2.set_ylabel(r"$z$", fontsize=fs, labelpad=8)
cf = ax2.contourf(x, z, psi, levels=levels, cmap=cm.seismic)

k, omega = 5, 0.5
wave_length = 2 * np.pi / np.abs(k)
x_max = 4 * wave_length
z = np.linspace(0, 1, 1000)
x = np.linspace(0, x_max, 200)
psi_1d = structure_function_normal(U0, U1, k, omega, z, N)
psi =  np.real(np.outer(psi_1d, np.exp(1j * k * x)))
ax3 = fig.add_subplot(133)
ax3.set_title(r"$k={:.1f}, \ \sigma={:.1f}$".format(k, omega),
              fontsize=fs, loc='left')
ax3.set_xlabel(r"$x$", fontsize=fs, labelpad=10)
ax3.set_ylabel(r"$z$", fontsize=fs, labelpad=8)
cf = ax3.contourf(x, z, psi, levels=levels, cmap=cm.seismic)

fig.tight_layout()

file_name = 'structure_functions';
plt.savefig(file_name + '.eps');
plt.savefig(file_name + '.png');
