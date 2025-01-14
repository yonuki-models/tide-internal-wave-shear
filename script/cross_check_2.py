# Display structure functions for numerics in Biswas and Shukla (2022) Table 1
# 10.1103/PhysRevFluids.7.023904

import numpy as np
from scipy.special import gamma
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm

rc('font', family='Times New Roman')
plt.rcParams['mathtext.fontset'] = 'cm'
fs = 14
eps_i = 1.e-10j

# Note that the vertical domain was -1 < z < 1 in Biswas and Skukla,
# which differs from the present formulation by a factor of 2.
factor = 2


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


def II(mu, z1, z2, N):
    return (Bessel_I(mu, z1, N)
            * Bessel_I(-mu, z2, N)
            - Bessel_I(-mu, z1, N)
            * Bessel_I(mu, z2, N))


def structure_function(nu, U0, U1, k, omega, z, N):
    Z1 = (k * U1 - omega) / (U1 - U0)
    U = U1 * z + U0 *(1 - z)
    Z = (k * U - omega) / (U1 - U0)
    return ((Z + eps_i)**0.5 * II(1j * nu, Z1, Z, N))


def structure_function_normal(nu, U0, U1, k, omega, z, N):
    F = structure_function(nu, U0, U1, k, omega, z, N)
    return F / np.abs(F).max()


buoyancy_frequency = 2 * np.sqrt(5.)
U0 = -1.
U1 = 1.
Ri = buoyancy_frequency**2 / (U0 - U1)**2
nu = (Ri - 0.25)**0.5

N = 100

fig = plt.figure(figsize=(6, 6))
omega = 1.09730 * factor

k = 0.721779 * factor
Z0 = (k * U0 - omega) / (U1 - U0)
Z1 = (k * U1 - omega) / (U1 - U0)
z = np.linspace(0, 1, 1000)
psi = structure_function_normal(nu, U0, U1, k, omega, z, N)
ax1 = fig.add_subplot(221)
ax1.plot(np.real(psi), 2 * z - 1, 'k')
ax1.vlines(0, -1, 1, 'gray', ':')
ax1.set_title(r"$k={:.5f}, \ \sigma={:.5f}$".format(
    k / factor, omega / factor),
              fontsize=fs, loc='left')
ax1.set_xlabel(r"$\psi$", fontsize=fs, labelpad=10)
ax1.set_ylabel(r"$z$", fontsize=fs, labelpad=8)
ax1.set_xlim([-1, 1])
ax1.set_ylim([-1, 1])
print(k, np.real(psi)[0], np.real(psi)[-1])

k = 1. * factor
Z0 = (k * U0 - omega) / (U1 - U0)
Z1 = (k * U1 - omega) / (U1 - U0)
z = np.linspace(0, 1, 1000)
psi = structure_function_normal(nu, U0, U1, k, omega, z, N)
ax2 = fig.add_subplot(222)
ax2.plot(np.real(psi), 2 * z - 1, 'k')
ax2.vlines(0, -1, 1, 'gray', ':')
ax2.set_title(r"$k={:.5f}, \ \sigma={:.5f}$".format(
    k / factor, omega / factor),
              fontsize=fs, loc='left')
ax2.set_xlabel(r"$\psi$", fontsize=fs, labelpad=10)
ax2.set_ylabel(r"$z$", fontsize=fs, labelpad=8)
ax2.set_xlim([-1, 1])
ax2.set_ylim([-1, 1])
print(k, np.real(psi)[0], np.real(psi)[-1])

k = 1.07379 * factor
Z0 = (k * U0 - omega) / (U1 - U0)
Z1 = (k * U1 - omega) / (U1 - U0)
z = np.linspace(0, 1, 1000)
psi = structure_function_normal(nu, U0, U1, k, omega, z, N)
ax3 = fig.add_subplot(223)
ax3.plot(np.real(psi), 2 * z - 1, 'k')
ax3.vlines(0, -1, 1, 'gray', ':')
ax3.set_title(r"$k={:.5f}, \ \sigma={:.5f}$".format(
    k / factor, omega / factor),
              fontsize=fs, loc='left')
ax3.set_xlabel(r"$\psi$", fontsize=fs, labelpad=10)
ax3.set_ylabel(r"$z$", fontsize=fs, labelpad=8)
ax3.set_xlim([-1, 1])
ax3.set_ylim([-1, 1])
print(k, np.real(psi)[0], np.real(psi)[-1])

k = 1.09172 * factor
Z0 = (k * U0 - omega) / (U1 - U0)
Z1 = (k * U1 - omega) / (U1 - U0)
z = np.linspace(0, 1, 1000)
psi = structure_function_normal(nu, U0, U1, k, omega, z, N)
ax4 = fig.add_subplot(224)
ax4.plot(np.real(psi), 2 * z - 1, 'k')
ax4.vlines(0, -1, 1, 'gray', ':')
ax4.set_title(r"$k={:.5f}, \ \sigma={:.5f}$".format(
    k / factor, omega / factor),
              fontsize=fs, loc='left')
ax4.set_xlabel(r"$\psi$", fontsize=fs, labelpad=10)
ax4.set_ylabel(r"$z$", fontsize=fs, labelpad=8)
ax4.set_xlim([-1, 1])
ax4.set_ylim([-1, 1])
print(k, np.real(psi)[0], np.real(psi)[-1])

fig.tight_layout()

file_name = 'check_2';
plt.savefig(file_name + '.eps');
plt.savefig(file_name + '.png');
