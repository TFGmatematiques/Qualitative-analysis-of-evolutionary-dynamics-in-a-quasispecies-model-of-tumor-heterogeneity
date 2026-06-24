# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:51:13 2026

"""

import numpy as np
import matplotlib.pyplot as plt

K = 1.0
mu12 = 0.05
mu21 = 0.0
d10 = 0.20      
d2 = 0.0         
omega = 0.1   
   
A_values = [0.0, 0.05, 0.10, 0.15, 0.19]

x1_inicial = 0.20
x2_inicial = 0.05

h = 0.05
tmax = 300.0
t_values = np.arange(0, tmax + h, h)

casos = {
    r"$\Delta f_0 > 0$": {
        "r1": 0.80,
        "r2": 0.35
    },
    r"$\Delta f_0 = 0$": {
        "r1": 0.55,
        "r2": 0.35
    },
    r"$\Delta f_0 < 0$": {
        "r1": 0.30,
        "r2": 0.35
    }
}

def d1_t(t, A):
    return d10 + A * np.sin(omega * t)

def f1(x1, x2, t, r1, A):
    N = x1 + x2
    return r1 * (1 - N / K) - d1_t(t, A)

def f2(x1, x2, r2):
    N = x1 + x2
    return r2 * (1 - N / K) - d2

def camp(t, X, r1, r2, A):
    x1, x2 = X

    f1_val = f1(x1, x2, t, r1, A)
    f2_val = f2(x1, x2, r2)

    dx1 = (1 - mu12) * f1_val * x1
    dx2 = mu12 * f1_val * x1 + f2_val * x2

    return np.array([dx1, dx2])

def rk4_step(t, X, h, r1, r2, A):
    k1 = camp(t, X, r1, r2, A)
    k2 = camp(t + 0.5*h, X + 0.5*h*k1, r1, r2, A)
    k3 = camp(t + 0.5*h, X + 0.5*h*k2, r1, r2, A)
    k4 = camp(t + h, X + h*k3, r1, r2, A)

    return X + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def integra_RK4(r1, r2, A):
    X = np.array([x1_inicial, x2_inicial], dtype=float)

    x1_values = []
    x2_values = []
    N_values = []
    z_values = []

    for t in t_values:

        x1, x2 = X
        N = x1 + x2

        x1_values.append(x1)
        x2_values.append(x2)
        N_values.append(N)

        if N > 0:
            z_values.append(x1 / N)
        else:
            z_values.append(0.0)

        X = rk4_step(t, X, h, r1, r2, A)

        X = np.maximum(X, 0.0)

        if not np.all(np.isfinite(X)):
            print("Error numèric")
            break

    return (
        np.array(x1_values),
        np.array(x2_values),
        np.array(N_values),
        np.array(z_values)
    )

fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

for ax, (nom_cas, params) in zip(axs, casos.items()):

    r1 = params["r1"]
    r2 = params["r2"]

    Delta_f0 = (r1 - d10) - r2

    print("\n======================================")
    print("Cas:", nom_cas)
    print("r1 =", r1)
    print("r2 =", r2)
    print("d10 =", d10)
    print("Delta_f0 = (r1 - d10) - r2 =", Delta_f0)
    print("======================================")


    for A in A_values:

        x1_values, x2_values, N_values, z_values = integra_RK4(r1, r2, A)

        ax.plot(
            t_values[:len(N_values)],
            N_values,
            linewidth=1.7,
            label=fr"$A={A}$"
        )

    ax.set_title(nom_cas, fontsize=15)
    ax.set_xlabel(r"$t$", fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=15, loc="best", frameon=True)
    ax.grid(False)

axs[0].set_ylabel(r"$N(t)$", fontsize=15)

plt.tight_layout()
plt.show()

r1 = casos[r"$\Delta f_0 > 0$"]["r1"]
r2 = casos[r"$\Delta f_0 > 0$"]["r2"]
A = 0.19

x1_values, x2_values, N_values, z_values = integra_RK4(r1, r2, A)

plt.figure(figsize=(8, 5))

plt.plot(
    t_values[:len(x1_values)],
    x1_values,
    linewidth=2,
    label=r"$x_1(t)$"
)

plt.plot(
    t_values[:len(x2_values)],
    x2_values,
    linewidth=2,
    label=r"$x_2(t)$"
)

plt.plot(
    t_values[:len(N_values)],
    N_values,
    "--",
    linewidth=2.2,
    label=r"$N(t)$"
)

plt.axhline(0, color="black", linestyle="--", lw=0.5)
plt.axvline(0, color="black", linestyle="--", lw=0.5)
plt.xlim(-5, tmax)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r"$t$", fontsize=15)
plt.ylabel("Population", fontsize=15)
plt.legend(fontsize=15, loc="best")
plt.grid(False)
plt.tight_layout()
plt.show()
