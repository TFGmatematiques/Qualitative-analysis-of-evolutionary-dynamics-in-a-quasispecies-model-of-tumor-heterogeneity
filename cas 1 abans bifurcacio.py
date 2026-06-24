# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:38:23 2026

"""

import numpy as np
import matplotlib.pyplot as plt

f1 = 0.2
f2 = 0.18
Delta_f = f1 - f2

mu12_c = Delta_f / f1

print("Delta_f =", Delta_f)
print("mu12_c =", mu12_c)

z_inicial = 0.01
h = 0.5           
tmax = 2e6          
eps_radius = 1.0    
tol_eq = 1e-4        
tol_camp = 1e-12    

def camp(z, mu):
    return z * (f1 * mu - Delta_f * z)

def equilibri_estable(mu):
    return f1 * mu / Delta_f

def rk4_step(z, mu, h):
    k1 = camp(z, mu)
    k2 = camp(z + 0.5 * h * k1, mu)
    k3 = camp(z + 0.5 * h * k2, mu)
    k4 = camp(z + h * k3, mu)

    return z + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def integra_RK4_T(mu):
    t = 0.0
    z = z_inicial

    z_eq = equilibri_estable(mu)

    if abs(z - z_eq) < tol_eq:
        return 0.0, "ja era prop de l'equilibri", z

    while t < tmax:

        if abs(z - z_eq) < tol_eq:
            return t, "a prop de l'equilibri", z

        if abs(z) >= eps_radius:
            return t, "radi epsilon", z

        if abs(camp(z, mu)) < tol_camp and abs(z - z_eq) < 10 * tol_eq:
            return t, "camp petit i prop de l'equilibri", z

        z = rk4_step(z, mu, h)
        t += h

        if not np.isfinite(z):
            return t, "error numèric", z

    return tmax, "temps màxim", z

mu_values = np.logspace(-5, -1, 80)

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

mu_problematic = Delta_f * z_inicial / f1
print("\nValor on z_inicial coincideix amb l'equilibri:")
print("mu_problematic =", mu_problematic)

plt.figure(figsize=(8, 5))
plt.plot(mu_values, T_values, "o", markersize=4, zorder=5)
plt.axhline(0, color="black", linestyle="--", lw=0.5)
plt.axvline(0, color="black", linestyle="--", lw=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r"$\mu$", fontsize=15)
plt.ylabel(r"$T(\mu)$", fontsize=15)
plt.grid(False)
plt.tight_layout()
plt.show()

mask_positive_T = T_values > 0

log_mu_all = np.log(mu_values[mask_positive_T])
log_T_all = np.log(T_values[mask_positive_T])

mask_no_tmax = T_values < 0.95 * tmax

mask_no_problematic = np.abs(mu_values - mu_problematic) > 3e-4

finestres = [
    (1e-5, 1e-3),
    (2e-5, 2e-3),
    (5e-5, 5e-3),
    (1e-4, 1e-2),
    (5e-4, 3e-2),
    (1e-3, 5e-2),
]

print("\nProva de robustesa de l'ajust:")

for a, b in finestres:
    mask_fit_test = (
        (mu_values > a)
        & (mu_values < b)
        & (T_values > 0)
        & mask_no_tmax
        & mask_no_problematic
    )

    n_punts = np.sum(mask_fit_test)

    if n_punts >= 5:
        pendent_test, ordenada_test = np.polyfit(
            np.log(mu_values[mask_fit_test]),
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

mu_min_fit = 1e-4
mu_max_fit = 1e-2

mask_fit = (
    (mu_values > mu_min_fit)
    & (mu_values < mu_max_fit)
    & (T_values > 0)
    & mask_no_tmax
    & mask_no_problematic
)

log_mu_fit = np.log(mu_values[mask_fit])
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
plt.xlabel(r"$\log\mu$", fontsize=15)
plt.ylabel(r"$\log T$", fontsize=15)
plt.legend(fontsize=15, loc="best")
plt.grid(False)
plt.tight_layout()
plt.show()
