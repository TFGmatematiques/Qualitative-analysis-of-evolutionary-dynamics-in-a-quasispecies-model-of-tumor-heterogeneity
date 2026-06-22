# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:56:16 2026

@author: noaga
"""

import numpy as np
import matplotlib.pyplot as plt

K = 1.0
r1 = 0.5
r2 = 0.4
d1 = 0.20
mu21 = 0.0

d2_critic = r2 * d1 / r1

d2_values = [0.08, d2_critic, 0.24]
mu12_values = [0.0, 0.20, 0.60]

x1_min, x1_max = 0.0, 1.2
x2_min, x2_max = 0.0, 1.2

n_grid = 25
x1 = np.linspace(x1_min, x1_max, n_grid)
x2 = np.linspace(x2_min, x2_max, n_grid)
X1, X2 = np.meshgrid(x1, x2)

def f1(x1, x2):
    N = x1 + x2
    return r1 * (1 - N / K) - d1

def f2(x1, x2, d2):
    N = x1 + x2
    return r2 * (1 - N / K) - d2

def camp_2D(x1, x2, d2, mu12):
    f1_val = f1(x1, x2)
    f2_val = f2(x1, x2, d2)

    dx1 = (1 - mu12) * f1_val * x1
    dx2 = mu12 * f1_val * x1 + f2_val * x2

    return dx1, dx2

def punts_equilibri(d2):
    equilibris = []

    xi = d2 - r2 * d1 / r1

    equilibris.append((0.0, 0.0, "unstable"))

    if d1 < r1:
        x1_eq = K * (1 - d1 / r1)

        if xi > 0:
            tipus = "stable"
        elif xi < 0:
            tipus = "saddle"
        else:
            tipus = "line"

        equilibris.append((x1_eq, 0.0, tipus))

    if d2 < r2:
        x2_eq = K * (1 - d2 / r2)

        if xi < 0:
            tipus = "stable"
        elif xi > 0:
            tipus = "saddle"
        else:
            tipus = "line"

        equilibris.append((0.0, x2_eq, tipus))

    return equilibris

fig, axs = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)

for i, d2 in enumerate(d2_values):
    for j, mu12 in enumerate(mu12_values):

        ax = axs[i, j]

        U, V = camp_2D(X1, X2, d2, mu12)

        speed = np.sqrt(U**2 + V**2)
        speed[speed == 0] = 1.0

        ax.quiver(
            X1, X2,
            U / speed,
            V / speed,
            angles="xy",
            scale_units="xy",
            scale=18,
            width=0.003
        )

        if np.isclose(d2, d2_critic):
            N_line = K * (1 - d1 / r1)
            x_line = np.linspace(0, N_line, 200)
            y_line = N_line - x_line
            ax.plot(x_line, y_line, color="black", linewidth=2.0)

        for x_eq, y_eq, tipus in punts_equilibri(d2):
            if tipus == "stable":
                ax.plot(x_eq, y_eq, "o", color="black", markersize=7)
            elif tipus == "unstable":
                ax.plot(x_eq, y_eq, "o", markerfacecolor="white",
                        markeredgecolor="black", markersize=7)
            elif tipus == "saddle":
                ax.plot(x_eq, y_eq, "s", color="black", markersize=7)
            elif tipus == "line":
                ax.plot(x_eq, y_eq, "o", color="black", markersize=6)

        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)

        ax.set_title(fr"$d_2={d2:.2f}$, $\mu_{{12}}={mu12}$", fontsize=13)

        if i == 2:
            ax.set_xlabel(r"$x_1$", fontsize=13)
        if j == 0:
            ax.set_ylabel(r"$x_2$", fontsize=13)

        ax.tick_params(axis="both", labelsize=10)
        ax.grid(False)

plt.tight_layout()
plt.show()