import numpy as np
import matplotlib.pyplot as plt

K = 1.0
r1 = 0.5
d1 = 0.2
d2 = 0.0
mu12 = 0.2

mu21_values = [0.2, 0.8, 0.95]
r2_values = [0.15, 0.30, 0.50]

row_titles = [r"$\Delta f>0$", r"$\Delta f=0$", r"$\Delta f<0$"]
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

def f2(x1, x2, r2):
    N = x1 + x2
    return r2 * (1 - N / K)

def camp_2D(x1, x2, r2, mu21):
    f1_val = f1(x1, x2)
    f2_val = f2(x1, x2, r2)

    dx1 = (1 - mu12) * f1_val * x1 + mu21 * f2_val * x2
    dx2 = mu12 * f1_val * x1 + (1 - mu21) * f2_val * x2

    return dx1, dx2

def jacobia(x1, x2, r2, mu21):
    N = x1 + x2

    f1_val = r1 * (1 - N / K) - d1
    f2_val = r2 * (1 - N / K)

    df1 = -r1 / K
    df2 = -r2 / K

    J11 = (1 - mu12) * (df1 * x1 + f1_val) + mu21 * df2 * x2
    J12 = (1 - mu12) * df1 * x1 + mu21 * (df2 * x2 + f2_val)

    J21 = mu12 * (df1 * x1 + f1_val) + (1 - mu21) * df2 * x2
    J22 = mu12 * df1 * x1 + (1 - mu21) * (df2 * x2 + f2_val)

    return np.array([[J11, J12], [J21, J22]])

def classifica_equilibri(x1, x2, r2, mu21):
    eigvals = np.linalg.eigvals(jacobia(x1, x2, r2, mu21))
    real_parts = np.real(eigvals)

    tol = 1e-8

    if np.any(np.abs(real_parts) < tol):
        return "degenerate"
    if np.all(real_parts < 0):
        return "stable"
    if np.all(real_parts > 0):
        return "unstable"

    return "saddle"

def punts_equilibri(r2, mu21):
    equilibris = []
    sigma = 1 - mu12 - mu21

    if not np.isclose(sigma, 0.0):
        candidats = [(0.0, 0.0), (0.0, K)]

        if d1 < r1:
            candidats.append((K * (1 - d1 / r1), 0.0))

        for x_eq, y_eq in candidats:
            tipus = classifica_equilibri(x_eq, y_eq, r2, mu21)
            equilibris.append((x_eq, y_eq, tipus))

    return equilibris

def corba_equilibris(X1, X2, r2):
    N = X1 + X2
    return (1 - N / K) * (r1 * X1 + r2 * X2) - d1 * X1

fig, axs = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)

for i, r2 in enumerate(r2_values):
    for j, mu21 in enumerate(mu21_values):

        ax = axs[i, j]

        sigma = 1 - mu12 - mu21

        U, V = camp_2D(X1, X2, r2, mu21)

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

        if np.isclose(sigma, 0.0):
            E = corba_equilibris(X1, X2, r2)
            ax.contour(X1, X2, E, levels=[0], colors="black", linewidths=2.0)

        for x_eq, y_eq, tipus in punts_equilibri(r2, mu21):
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
            fr"$r_2={r2}$, $\mu_{{21}}={mu21}$",
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
