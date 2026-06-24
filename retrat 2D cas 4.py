import numpy as np
import matplotlib.pyplot as plt

K = 1.0
r1 = 0.5
r2 = 0.4
d1 = 0.20
mu12 = 0.20

mu21_values = [0.20, 0.80, 0.95]

d2_critic = r2 * d1 / r1
d2_values = [0.08, d2_critic, 0.24]

row_titles = [r"$\xi<0$", r"$\xi=0$", r"$\xi>0$"]
col_titles = [r"$\sigma>0$", r"$\sigma=0$", r"$\sigma<0$"]

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

def camp_2D(x1, x2, d2, mu21):
    f1_val = f1(x1, x2)
    f2_val = f2(x1, x2, d2)

    dx1 = (1 - mu12) * f1_val * x1 + mu21 * f2_val * x2
    dx2 = mu12 * f1_val * x1 + (1 - mu21) * f2_val * x2

    return dx1, dx2

def jacobia(x1, x2, d2, mu21):
    N = x1 + x2

    f1_val = r1 * (1 - N / K) - d1
    f2_val = r2 * (1 - N / K) - d2

    df1 = -r1 / K
    df2 = -r2 / K

    J11 = (1 - mu12) * (df1 * x1 + f1_val) + mu21 * df2 * x2
    J12 = (1 - mu12) * df1 * x1 + mu21 * (df2 * x2 + f2_val)

    J21 = mu12 * (df1 * x1 + f1_val) + (1 - mu21) * df2 * x2
    J22 = mu12 * df1 * x1 + (1 - mu21) * (df2 * x2 + f2_val)

    return np.array([[J11, J12], [J21, J22]])

def classifica_equilibri(x1, x2, d2, mu21):
    eigvals = np.linalg.eigvals(jacobia(x1, x2, d2, mu21))
    real_parts = np.real(eigvals)

    tol = 1e-8

    if np.any(np.abs(real_parts) < tol):
        return "degenerate"
    if np.all(real_parts < 0):
        return "stable"
    if np.all(real_parts > 0):
        return "unstable"

    return "saddle"

def punts_equilibri(d2, mu21):
    equilibris = []
    sigma = 1 - mu12 - mu21

    if not np.isclose(sigma, 0.0):

        candidats = [(0.0, 0.0)]

        if d1 < r1:
            candidatos_x1 = (K * (1 - d1 / r1), 0.0)
            candidats.append(candidatos_x1)

        if d2 < r2:
            candidatos_x2 = (0.0, K * (1 - d2 / r2))
            candidats.append(candidatos_x2)

        for x_eq, y_eq in candidats:
            tipus = classifica_equilibri(x_eq, y_eq, d2, mu21)
            equilibris.append((x_eq, y_eq, tipus))

    return equilibris

def recta_equilibris_xi_zero(d2):
    N_line = K * (1 - d1 / r1)
    x_line = np.linspace(0, N_line, 200)
    y_line = N_line - x_line
    return x_line, y_line

def corba_equilibris_sigma_zero(X1, X2, d2):
    N = X1 + X2
    return (1 - N / K) * (r1 * X1 + r2 * X2) - d1 * X1 - d2 * X2

fig, axs = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)

for i, d2 in enumerate(d2_values):
    for j, mu21 in enumerate(mu21_values):

        ax = axs[i, j]

        sigma = 1 - mu12 - mu21
        xi = d2 - r2 * d1 / r1

        U, V = camp_2D(X1, X2, d2, mu21)

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

        if np.isclose(xi, 0.0) and not np.isclose(sigma, 0.0):
            x_line, y_line = recta_equilibris_xi_zero(d2)
            ax.plot(x_line, y_line, color="black", linewidth=2.0)

        if np.isclose(sigma, 0.0):
            E = corba_equilibris_sigma_zero(X1, X2, d2)
            ax.contour(X1, X2, E, levels=[0], colors="black", linewidths=2.0)

        for x_eq, y_eq, tipus in punts_equilibri(d2, mu21):
            if tipus == "stable":
                ax.plot(x_eq, y_eq, "o", color="black", markersize=7)
            elif tipus == "unstable":
                ax.plot(x_eq, y_eq, "o", markerfacecolor="white",
                        markeredgecolor="black", markersize=7)
            elif tipus == "saddle":
                ax.plot(x_eq, y_eq, "s", color="black", markersize=7)
            elif tipus == "degenerate":
                ax.plot(x_eq, y_eq, "o", color="black", markersize=7)

        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)

        ax.set_title(
            fr"{row_titles[i]}, {col_titles[j]}" + "\n" +
            fr"$d_2={d2:.2f}$, $\mu_{{21}}={mu21}$",
            fontsize=12
        )

        if i == 2:
            ax.set_xlabel(r"$x_1$", fontsize=13)
        if j == 0:
            ax.set_ylabel(r"$x_2$", fontsize=13)

        ax.tick_params(axis="both", labelsize=10)
        ax.grid(False)

plt.tight_layout()
plt.show()
