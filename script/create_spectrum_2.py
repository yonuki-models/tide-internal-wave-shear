import numpy as np
from scipy.special import gamma
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', family='Times New Roman')
plt.rcParams['mathtext.fontset'] = 'cm'
fs = 20
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


buoyancy_frequency = 1.
U0 = 0.05
U1 = 0.15
Ri = buoyancy_frequency / (U0 - U1)**2
nu = 1j * (Ri - 0.25)**0.5

N = 100
NK = 100
NO = 300
k_max = 10
omega_max = buoyancy_frequency

k_axis_plus = np.logspace(-2, np.log10(k_max), NK)
tmp = np.append(-k_axis_plus[::-1], 0)
k_axis = np.append(tmp, k_axis_plus)

omega_axis = np.zeros(NO)
omega, k = np.meshgrid(omega_axis, k_axis)
for j in range(2*NK+1):
    if k_axis[j] < 0:
        omega[j, :] = k_axis[j] * U0 + np.logspace(-3, np.log10(omega_max), NO)
    else:
        omega[j, :] = k_axis[j] * U1 + np.logspace(-3, np.log10(omega_max), NO)

Z0 = (k * U0 - omega) / (U1 - U0)
Z1 = (k * U1 - omega) / (U1 - U0)

func = (1j * II(nu, Z0, Z1, N))

fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(111)
ax1.grid()

ax1.contour(k[NK+1:, :], omega[NK+1:, :], np.real(func[NK+1:, :]),
            colors=['r'], levels=[0])
ax1.contour(k[:NK, :], omega[:NK, :], np.real(func[:NK, :]),
            colors=['b'], levels=[0])
ax1.contour(- k[:NK, :], - omega[:NK, :], np.real(func[:NK, :]),
            colors=['b'], levels=[0])
ax1.set_xlim([-k_max, k_max])
ax1.set_ylim([0, k_max * U1 + omega_max])
ax1.hlines(y=buoyancy_frequency, xmin=-100, xmax=100,
           colors='k', linestyles=':')
ax1.set_xlabel(r"$k$", fontsize=20, labelpad=10)
ax1.set_ylabel(r"$\sigma$", fontsize=20, labelpad=8);

fig.tight_layout()

file_name = 'spectrum_2'

plt.savefig(file_name + '.eps')
plt.savefig(file_name + '.png')
