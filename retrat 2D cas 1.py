import numpy as np
import matplotlib.pyplot as plt

K = 1.0
r1 = 0.5
r2 = 0.4
d2 = 0.0
mu21 = 0.0

d1_values = [0.02, 0.20, 0.45]
mu12_values = [0.0, 0.20, 0.60]

x1_min, x1_max = 0.0, 1.2
x2_min, x2_max = 0.0, 1.2

n_grid = 25
x1 = np.linspace(x1_min, x1_max, n_grid)
x2 = np.linspace(x2_min, x2_max, n_grid)
X1, X2 = np.meshgrid(x1, x2)

def f1(x1, x2, d1):
    N = x1 + x2
    return r1 * (1 - N / K) - d1

def f2(x1, x2):
    N = x1 + x2
    return r2 * (1 - N / K)

def camp_2D(x1, x2, d1, mu12):
    f1_val = f1(x1, x2, d1)
    f2_val = f2(x1, x2)

    dx1 = (1 - mu12) * f1_val * x1
    dx2 = mu12 * f1_val * x1 + f2_val * x2

    return dx1, dx2

def punts_equilibri(d1):
    equilibris = []

    equilibris.append((0.0, 0.0, "unstable"))
    equilibris.append((0.0, K, "stable"))

    if d1 < r1:
        equilibris.append((K * (1 - d1 / r1), 0.0, "saddle"))

    return equilibris

fig, axs = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)

for i, d1 in enumerate(d1_values):
    for j, mu12 in enumerate(mu12_values):

        ax = axs[i, j]

        U, V = camp_2D(X1, X2, d1, mu12)

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

        for x_eq, y_eq, tipus in punts_equilibri(d1):
            if tipus == "stable":
                ax.plot(x_eq, y_eq, "o", color="black", markersize=7)
            elif tipus == "unstable":
                ax.plot(x_eq, y_eq, "o", markerfacecolor="white",
                        markeredgecolor="black", markersize=7)
            elif tipus == "saddle":
                ax.plot(x_eq, y_eq, "s", color="black", markersize=7)

        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)

        ax.set_title(fr"$d_1={d1}$, $\mu_{{12}}={mu12}$", fontsize=13)

        if i == 2:
            ax.set_xlabel(r"$x_1$", fontsize=13)
        if j == 0:
            ax.set_ylabel(r"$x_2$", fontsize=13)

        ax.tick_params(axis="both", labelsize=10)
        ax.grid(False)

plt.tight_layout()
plt.show()
