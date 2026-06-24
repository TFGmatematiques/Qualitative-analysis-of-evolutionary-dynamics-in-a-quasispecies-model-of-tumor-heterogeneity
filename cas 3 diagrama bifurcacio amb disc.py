import numpy as np
import matplotlib.pyplot as plt

f1 = 0.4
f2 = 0.5
mu21 = 0.2
delta_f = f1 - f2
mu12_estudi = 0.05

mu12_valors = np.linspace(0, 0.055, 500)
disc, z1, z2 = [], [], []

for m in mu12_valors:
    a = -delta_f
    b = (delta_f - m * f1 - mu21 * f2)
    c = mu21 * f2
    
    D = b**2 - 4*a*c
    disc.append(D)
    z1.append((-b + np.sqrt(D)) / (2*a))
    z2.append((-b - np.sqrt(D)) / (2*a))

a_est = -delta_f
b_est = (delta_f - mu12_estudi * f1 - mu21 * f2)
c_est = mu21 * f2
D_est = b_est**2 - 4*a_est*c_est

z1_est = (-b_est + np.sqrt(D_est)) / (2 * a_est)
z2_est = (-b_est - np.sqrt(D_est)) / (2 * a_est)

plt.figure(figsize=(8, 5))
plt.plot(disc, z1, color='red', linestyle='--', linewidth=2)
plt.plot(disc, z2, color='blue', linewidth=2)
plt.plot(D_est, z1_est, 'ko', markersize=7, zorder=5, label=f'$\mu_{{12}}={mu12_estudi}$')
plt.plot(D_est, z2_est, 'ko', markersize=7, zorder=5)

b_zero = (delta_f - mu21 * f2)
D_zero = b_zero**2 - 4*a_est*(mu21 * f2)
z_zero = (-b_zero + np.sqrt(D_zero)) / (2 * a_est)
plt.plot(D_zero, z_zero, 'o', color='green', markersize=7, zorder=5)

plt.plot([], [], 'b-', lw=2, label='Stable')
plt.plot([], [], 'r--', lw=2, label='Unstable')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel(r'$\Delta$', fontsize=15)
plt.ylabel(r'$z^*$', fontsize=15)
plt.axhline(0, color='black', linestyle='--', lw=0.5)
plt.axvline(0, color='black', linestyle='--', lw=0.5)
plt.xlim(-0.0002, 0.0096)
plt.legend(
    fontsize=15, 
    loc='lower left', 
    bbox_to_anchor=(0.02, 0.05),
    labelspacing=0.2,  
    handletextpad=0.5, 
    borderaxespad=0.5  
)
plt.grid(False) 
plt.tight_layout()
plt.show()
