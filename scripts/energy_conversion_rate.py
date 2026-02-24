#!/usr/bin/env python3
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.special import jv

# ------------------ parameters (paper figure) ------------------
N, U0, U1 = 1.0, 0.05, 0.20
omega_T, U_T, r = 0.20, 0.20, 1e-5
kmin, kmax = -20.0, 20.0

J, dk, Nk_cont = 10, 0.02, 1200       # roots / scan step / continuous grid
NK_spec, NO_spec = 120, 140           # panel (a) grid
sigma_max = 3.0 * omega_T             # panel (a) y-limit

mp.mp.dps = 70                        # mpmath precision

Ubar, S = 0.5 * (U0 + U1), (U1 - U0)
nu = mp.sqrt((mp.mpf(N) / mp.mpf(S)) ** 2 - mp.mpf("0.25"))
mu = mp.j * nu
sym = 2.0  # symmetry factor with respect to (k, omega) = (-k, -omega)

# ------------------ special functions & forcing ------------------
def psi_hat(k, n):
    """Fourier amplitude used in the paper formulas; here \\hat{h}(k)=1."""
    k = np.asarray(k, float)
    w = float(n) * omega_T
    J = jv(n, k * U_T / omega_T)
    out = np.zeros_like(k)
    m0 = np.abs(k) < 1e-14
    out[~m0] = ((k[~m0] * U0 - w) / k[~m0]) * J[~m0]
    return out

def Iprime(order, z):
    return mp.mpf("0.5") * (mp.besseli(order - 1, z) + mp.besseli(order + 1, z))

def F_parts(k, sig):
    """Return (Z0, F, FZ0, FZ1) for the Bessel determinant F and its partials."""
    Z0, Z1 = (k * U0 - sig) / S, (k * U1 - sig) / S

    I0p, I0m = mp.besseli(mu, Z0), mp.besseli(-mu, Z0)
    I1p, I1m = mp.besseli(mu, Z1), mp.besseli(-mu, Z1)

    I0pp, I0mp = Iprime(mu, Z0), Iprime(-mu, Z0)
    I1pp, I1mp = Iprime(mu, Z1), Iprime(-mu, Z1)

    F   = I1m * I0p  - I1p  * I0m
    FZ0 = I1m * I0pp - I1p  * I0mp
    FZ1 = I1mp * I0p - I1pp * I0m
    return Z0, F, FZ0, FZ1

# ------------------ root finding (Im F = 0) ------------------
def bisect(f, a, b):
    a, b = mp.mpf(a), mp.mpf(b)
    fa, fb = f(a), f(b)
    tol = mp.mpf(10) ** (-(mp.mp.dps - 12))
    for _ in range(250):
        m = (a + b) / 2
        fm = f(m)
        if abs(fm) <= tol or abs(b - a) <= tol:
            return m
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return m

def scan_roots(sig, k0, k1, nmax):
    """Scan k from k0 to k1 and bracket roots by sign change."""
    sig = float(sig)
    step = mp.mpf(abs(dk)) * (1 if k1 > k0 else -1)
    eps  = mp.mpf("1e-10") * (1 if k1 > k0 else -1)

    f = lambda kk: mp.im(F_parts(kk, mp.mpf(sig))[1])

    roots = []
    k = mp.mpf(k0)
    f0 = f(k)
    while (k <= k1 if step > 0 else k >= k1) and len(roots) < nmax:
        k1p = k + step
        f1 = f(k1p)

        if f0 == 0:
            roots.append(k)
            k = k + eps
            f0 = f(k)
            continue

        if f0 * f1 < 0:
            r0 = bisect(f, k, k1p)
            roots.append(r0)
            k = r0 + eps
            f0 = f(k)
        else:
            k, f0 = k1p, f1
    return roots

def lowk_roots(sig, nmax):
    """Resolve accumulation near kc1=sig/U1 by scanning k=kc1(1-exp(-t))."""
    sig = float(sig)
    kc1 = mp.mpf(sig) / mp.mpf(U1)
    f = lambda kk: mp.im(F_parts(kk, mp.mpf(sig))[1])

    roots = []
    eps_rel = mp.mpf("1e-6")
    dt = mp.mpf("0.01")
    while len(roots) < nmax and eps_rel >= mp.mpf("1e-14"):
        eps = eps_rel * kc1
        tmax = mp.log(kc1 / eps)

        t = dt
        k = kc1 * (1 - mp.e ** (-t))
        f0 = f(k)

        while t < tmax and len(roots) < nmax:
            t1 = min(t + dt, tmax)
            k1 = kc1 * (1 - mp.e ** (-t1))
            f1 = f(k1)

            if f0 * f1 < 0:
                r0 = bisect(f, k, k1)
                roots.append(r0)
                k = r0 * mp.mpf("1.000000001")
                f0 = f(k)
                t = t1
            else:
                t, k, f0 = t1, k1, f1

        eps_rel /= 10
    return roots, float(kc1)

# ------------------ spectral coefficients ------------------
def cg_chi(k, sig):
    Z0, F, FZ0, FZ1 = F_parts(mp.mpf(k), mp.mpf(sig))
    denom = FZ0 + FZ1
    cg = (mp.mpf(U1) * FZ1 + mp.mpf(U0) * FZ0) / denom
    cg = float(mp.re(cg))
    chi = -(mp.mpf(k) * mp.mpf(S)) * (FZ0 / denom) / (2 * abs(cg))
    return cg, float(mp.re(chi))

def R_cont(k, sig):
    Z0, F, FZ0, FZ1 = F_parts(mp.mpf(k), mp.mpc(sig, r))
    return float(-mp.im(mp.mpf(k) * (1 / (2 * Z0) + FZ0 / F)))

def compute_n(n):
    w = float(n) * omega_T
    if n == 0:
        ks = [float(x) for x in scan_roots(0.0, 1e-6, kmax, J)]
        cg = np.array([cg_chi(k, 0.0)[0] for k in ks], float)
        chi = np.array([cg_chi(k, 0.0)[1] for k in ks], float)
        return dict(omega=0.0, k=np.array(ks), cg=cg, chi=chi,
                    k_cont=np.array([]), R=np.array([]))

    sig = w
    k_neg = [float(x) for x in scan_roots(sig, -1e-8, kmin, J)]
    k_low, kc1 = lowk_roots(sig, J)
    k_low = [float(x) for x in k_low]

    kc2 = float(sig / U0)
    k_high = [float(x) for x in scan_roots(sig, kmax, kc2 + 1e-8, J)]

    k = np.array(sorted(k_neg + k_low + k_high), float)
    cg = np.array([cg_chi(kk, sig)[0] for kk in k], float)
    chi = np.array([cg_chi(kk, sig)[1] for kk in k], float)

    k_cont = np.linspace(kc1, kc2, Nk_cont)
    R = np.array([R_cont(kk, sig) for kk in k_cont], float)

    return dict(omega=w, k=k, cg=cg, chi=chi, k_cont=k_cont, R=R)

def disc_sums(d, n, mask):
    if not np.any(mask):
        return 0.0, 0.0, 0.0
    k = d["k"][mask]
    chi = d["chi"][mask]
    psi = psi_hat(k, n)
    w = d["omega"]
    Pt = w * chi * (psi ** 2) * sym
    Ps = (-Ubar * k) * chi * (psi ** 2) * sym
    return float(Pt.sum()), float(Ps.sum()), float((Pt + Ps).sum())

def cont_sums(d, n):
    k = d["k_cont"]
    if k.size == 0:
        return 0.0, 0.0, 0.0
    psi = psi_hat(k, n)
    w = d["omega"]
    fac = 1.0 / (2.0 * np.pi)
    R = d["R"]
    dPt = fac * w * R * (psi ** 2)
    dPs = fac * (-Ubar * k) * R * (psi ** 2)
    Pt = float(np.trapezoid(dPt, k)) * sym
    Ps = float(np.trapezoid(dPs, k)) * sym
    return Pt, Ps, Pt + Ps

# ------------------ plotting ------------------
def apply_style():
    rc("font", family="Times New Roman")
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "legend.fontsize": 16,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.4,
    })

def main(out="energy_conversion_rates.pdf"):
    apply_style()
    data = {n: compute_n(n) for n in (0, 1, 2)}

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    # (a) spectrum overview
    k_plus = np.logspace(np.log10(1e-2), np.log10(abs(kmax)), NK_spec)
    k_axis = np.r_[-k_plus[::-1], 0.0, k_plus]
    ds = np.logspace(np.log10(1e-6), np.log10(N), NO_spec)

    K = np.tile(k_axis[:, None], (1, NO_spec))
    SIG = (np.where(k_axis[:, None] < 0, U0, U1) * k_axis[:, None]) + ds[None, :]

    iFreal = np.vectorize(
        lambda kk, ss: float(mp.re(mp.j * F_parts(mp.mpf(kk), mp.mpf(ss))[1])),
        otypes=[float],
    )
    D = iFreal(K, SIG)
    NK = NK_spec

    # continuous-spectrum region (k>0, sigma in [kU0, kU1])
    k_fill_max = min(kmax, sigma_max / U0)
    kf = np.linspace(0.0, k_fill_max, 400)
    ax[0].fill_between(kf, kf * U0, np.minimum(kf * U1, sigma_max), color="purple", alpha=0.22, zorder=0)

    ax[0].contour(K[NK + 1 :, :], SIG[NK + 1 :, :], D[NK + 1 :, :], levels=[0], colors=["r"], linewidths=1.2)
    ax[0].contour(K[:NK, :],      SIG[:NK, :],      D[:NK, :],      levels=[0], colors=["b"], linewidths=1.2)
    ax[0].contour(-K[:NK, :],    -SIG[:NK, :],      D[:NK, :],      levels=[0], colors=["b"], linewidths=1.2)

    ax[0].plot([kmin, kmax], [Ubar * kmin, Ubar * kmax], "--", color="k", linewidth=1.0)
    for n in (1, 2):
        ax[0].axhline(n * omega_T, linestyle=":", color="k", linewidth=1.1)
        m = data[n]["cg"] < 0
        if np.any(m):
            ax[0].plot(data[n]["k"][m], np.full(m.sum(), data[n]["omega"]), "o", ms=3, color="b")

    ax[0].set_xlim(kmin, kmax)
    ax[0].set_ylim(0.0, sigma_max)
    ax[0].set_xlabel(r"$k$")
    ax[0].set_ylabel(r"$\sigma$")
    ax[0].set_title("(a)", loc="left")

    # (b) continuous-spectrum integrand (n=1)
    n = 1
    d = data[n]
    k = d["k_cont"]
    R = d["R"]
    psi = psi_hat(k, n)
    fac = 1.0 / (2.0 * np.pi)
    dPt = fac * d["omega"] * R * (psi ** 2)
    dPs = fac * (-Ubar * k) * R * (psi ** 2)
    dP = dPt + dPs

    ax[1].axhline(0.0, color="k", linewidth=0.8)
    ax[1].axvline(d["omega"] / Ubar, linestyle="--", color="k", linewidth=1.0)
    ax[1].plot(k, dPt, label="Tide")
    ax[1].plot(k, dPs, label="Steady")
    ax[1].plot(k, dP,  label="Sum")
    ax[1].set_xlim(float(k.min()), float(k.max()))
    ax[1].set_xlabel(r"$k$")
    ax[1].set_ylabel(r"$d\mathcal{P}/dk$")
    ax[1].ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
    ax[1].set_ylim(-1.0e-3, 1.0e-3)
    ax[1].legend(frameon=False)
    ax[1].set_title("(b)", loc="left")

    # (c) grouped sums (n=0,1,2)
    labels, Pt_all, Ps_all, P_all, seps = [], [], [], [], []
    PtT = PsT = PT = 0.0

    for n in (0, 1, 2):
        d = data[n]
        before = len(labels)

        m = d["cg"] < 0
        if np.any(m):
            Pt, Ps, P = disc_sums(d, n, m)
            labels.append(r"$K^d_{\omega_%d}$" % n + "\n" + r"$(c^g<0)$")
            Pt_all.append(Pt); Ps_all.append(Ps); P_all.append(P)

        m = d["cg"] > 0
        if np.any(m):
            Pt, Ps, P = disc_sums(d, n, m)
            labels.append(r"$K^d_{\omega_%d}$" % n + "\n" + r"$(c^g>0)$")
            Pt_all.append(Pt); Ps_all.append(Ps); P_all.append(P)

        if d["k_cont"].size:
            Pt, Ps, P = cont_sums(d, n)
            labels.append(r"$K^c_{\omega_%d}$" % n)
            Pt_all.append(Pt); Ps_all.append(Ps); P_all.append(P)

        if len(labels) > before:
            seps.append(len(labels))

        PtT += (Pt_all[-1] if len(Pt_all) else 0.0)
        PsT += (Ps_all[-1] if len(Ps_all) else 0.0)
        PT  += (P_all[-1]  if len(P_all)  else 0.0)

    # true totals across all categories
    PtT = float(np.sum(Pt_all))
    PsT = float(np.sum(Ps_all))
    PT  = float(np.sum(P_all))

    labels.append("TOTAL\n(n=0,1,2)")
    Pt_all.append(PtT); Ps_all.append(PsT); P_all.append(PT)

    x = np.arange(len(labels), dtype=float)
    w = 0.26
    ax[2].axhline(0.0, color="k", linewidth=0.8)
    ax[2].bar(x - w, Pt_all, w, label="Tide")
    ax[2].bar(x,     Ps_all, w, label="Steady")
    ax[2].bar(x + w, P_all,  w, label="Sum")
    for s in seps:
        if 0 < s < len(labels):
            ax[2].axvline(s - 0.5, color="k", alpha=0.25, linewidth=0.8)

    ax[2].set_ylim(-5e-3, 30e-3)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(labels)
    ax[2].set_ylabel(r"$\mathcal{P}$")
    ax[2].ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
    ax[2].legend(frameon=False)
    ax[2].set_title("(c)", loc="left")

    fig.savefig(out)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
