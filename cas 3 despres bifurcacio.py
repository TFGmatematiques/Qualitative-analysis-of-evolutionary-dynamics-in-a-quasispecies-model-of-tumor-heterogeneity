# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:39:36 2026

@author: noaga
"""

import numpy as np
import matplotlib.pyplot as plt

f1 = 0.4
f2 = 0.5
Delta_f = f1 - f2

mu21 = 0.2
mu12_c = 0.0

print("f1 =", f1)
print("f2 =", f2)
print("Delta_f =", Delta_f)
print("mu21 =", mu21)
print("mu12_c =", mu12_c)

z_inicial = 0.01
h = 0.5           
tmax = 2e6        
eps_radius = 1.0     
tol_camp = 1e-12    

def camp(z, mu):
    mu12 = mu12_c + mu

    return (-Delta_f * z**2 + (Delta_f - mu12 * f1 - mu21 * f2) * z + mu21 * f2)

def rk4_step(z, mu, h):
    k1 = camp(z, mu)
    k2 = camp(z + 0.5 * h * k1, mu)
    k3 = camp(z + 0.5 * h * k2, mu)
    k4 = camp(z + h * k3, mu)

    return z + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def integra_RK4_T(mu):
    t = 0.0
    z = z_inicial

    while t < tmax:

        if abs(z) >= eps_radius:
            return t, "radi epsilon", z

        z = rk4_step(z, mu, h)
        t += h

        if not np.isfinite(z):
            return t, "error numèric", z

    return tmax, "temps màxim", z

mu_abs_values = np.logspace(-5, -1, 80)
mu_values = -mu_abs_values

T_values = []
motius = []
z_finals = []

for i, mu in enumerate(mu_values):
    print(f"Calculant {i+1}/{len(mu_values)}: mu = {mu:.3e}")

    T, motiu, z_final = integra_RK4_T(mu)

    T_values.append(T)
    motius.append(motiu)
    z_finals.append(z_final)

T_values = np.array(T_values)
z_finals = np.array(z_finals)

print("\nMotius d'aturada:")
for mu, T, motiu in zip(mu_values, T_values, motius):
    print(f"mu = {mu:.3e}, T = {T:.3e}, motiu = {motiu}")

plt.figure(figsize=(8, 5))
plt.plot(mu_abs_values, T_values, "o", markersize=4, zorder=5)
plt.axhline(0, color="black", linestyle="--", lw=0.5)
plt.axvline(0, color="black", linestyle="--", lw=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r"$|\mu|$", fontsize=15)
plt.ylabel(r"$T(|\mu|)$", fontsize=15)
plt.grid(False)
plt.tight_layout()
plt.show()

mask_positive_T = T_values > 0
mask_finite_T = np.isfinite(T_values)

log_mu_all = np.log(mu_abs_values[mask_positive_T & mask_finite_T])
log_T_all = np.log(T_values[mask_positive_T & mask_finite_T])

mask_no_tmax = T_values < 0.95 * tmax

finestres = [
    (1e-5, 1e-2),
    (5e-5, 2e-2),
    (1e-4, 5e-2),
    (2e-4, 4e-2),
    (5e-4, 3e-2),
    (1e-3, 2e-2),
    (2e-3, 1e-2),
]

print("\nProva de robustesa de l'ajust:")

for a, b in finestres:
    mask_fit_test = (
        (mu_abs_values > a)
        & (mu_abs_values < b)
        & mask_positive_T
        & mask_finite_T
        & mask_no_tmax
    )

    n_punts = np.sum(mask_fit_test)

    if n_punts >= 5:
        pendent_test, ordenada_test = np.polyfit(
            np.log(mu_abs_values[mask_fit_test]),
            np.log(T_values[mask_fit_test]),
            1
        )

        print(
            f"Finestra [{a:.1e}, {b:.1e}] "
            f"-> pendent = {pendent_test:.4f}, punts = {n_punts}"
        )
    else:
        print(
            f"Finestra [{a:.1e}, {b:.1e}] "
            f"-> massa pocs punts: {n_punts}"
        )

mu_min_fit = 5e-4
mu_max_fit = 3e-2

mask_fit = (
    (mu_abs_values > mu_min_fit)
    & (mu_abs_values < mu_max_fit)
    & mask_positive_T
    & mask_finite_T
    & mask_no_tmax
)

log_mu_fit = np.log(mu_abs_values[mask_fit])
log_T_fit = np.log(T_values[mask_fit])

pendent, ordenada = np.polyfit(log_mu_fit, log_T_fit, 1)

print("\nAjust final:")
print("Finestra usada: [{:.1e}, {:.1e}]".format(mu_min_fit, mu_max_fit))
print("Nombre de punts usats a l'ajust:", np.sum(mask_fit))
print("Pendent final =", pendent)

plt.figure(figsize=(8, 5))
plt.plot(log_mu_all, log_T_all, "o", markersize=4, label="RK4")
plt.plot(
    log_mu_fit,
    pendent * log_mu_fit + ordenada,
    "--",
    linewidth=3,
    label=f"Slope = {pendent:.3f}"
)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r"$\log|\mu|$", fontsize=15)
plt.ylabel(r"$\log T$", fontsize=15)
plt.legend(fontsize=15, loc="best")
plt.grid(False)
plt.tight_layout()
plt.show()