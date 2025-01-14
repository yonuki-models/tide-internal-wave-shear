# Create figures to be compared with Biswas and Shukla (2022) Figure 1
# 10.1103/PhysRevFluids.7.023904

import numpy as np
from scipy.special import gamma
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', family='Times New Roman')
plt.rcParams['mathtext.fontset'] = 'cm'
fs = 20
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


buoyancy_frequency = 2 * np.sqrt(5.)
U0 = -1.
U1 = 1.
Ri = buoyancy_frequency**2 / (U0 - U1)**2
nu = (Ri - 0.25)**0.5

N = 100
NK = 100
NO = 300
k_max = 10.
omega_max = 10.

k_axis = np.logspace(-2, np.log10(k_max), NK)

omega_axis = np.zeros(NO)
omega, k = np.meshgrid(omega_axis, k_axis)
for j in range(NK):
    omega[j, :] = k_axis[j] * U1 + np.logspace(-3, np.log10(omega_max), NO)

Z0 = (k * U0 - omega) / (U1 - U0)
Z1 = (k * U1 - omega) / (U1 - U0)

func = (1j * II(1j * nu, Z0, Z1, N))

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(121)
ax1.grid()

ax1.contour(k / factor, omega / factor, np.real(func),
            colors=['k'], levels=[0])
ax1.set_xlim([0, 3])
ax1.set_ylim([0, 3])
ax1.set_xlabel(r"$k$", fontsize=20, labelpad=10)
ax1.set_ylabel(r"$\omega$", fontsize=20, labelpad=8);

ax2 = fig.add_subplot(122)
ax2.grid()

ax2.contour(k / factor, omega / k, np.real(func),
            colors=['k'], levels=[0])
ax2.set_xlim([0, 3.2])
ax2.set_ylim([0.95, 1.63])
ax2.set_xlabel(r"$k$", fontsize=20, labelpad=10)
ax2.set_ylabel(r"$\omega$", fontsize=20, labelpad=8);


fig.tight_layout()

file_name = 'check_1'

plt.savefig(file_name + '.eps')
plt.savefig(file_name + '.png')
