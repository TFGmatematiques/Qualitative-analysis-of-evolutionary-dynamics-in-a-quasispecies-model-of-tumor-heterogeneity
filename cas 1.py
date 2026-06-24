import numpy as np
import matplotlib.pyplot as plt

z_min = 0.0
z_max = 1.0
punts = 500

z = np.linspace(z_min, z_max, punts)

def funcio(z, f1, f2, mu12, mu21):
    delta_f = f1 - f2
    return -delta_f * (z**2) + (delta_f - mu12*f1 - mu21*f2)*z + mu21*f2

casos = [
    {'f1': 0.2, 'f2': 0.18, 'mu12': 0.15, 'mu21': 0, 'color': 'crimson',  'label': 'B'},
    {'f1': 0.15, 'f2': 0.15, 'mu12': 0.05, 'mu21': 0, 'color': 'forestgreen',  'label': 'C'},
    {'f1': 0.1, 'f2': 0.25, 'mu12': 0.05, 'mu21': 0, 'color': 'royalblue',  'label': 'D'}
]

plt.figure(figsize=(8, 5))

for c in casos:
    z_punt = funcio(z, c['f1'], c['f2'], c['mu12'], c['mu21'])
    plt.plot(z, z_punt, color=c['color'], label=c['label'], linewidth=2)
     
    for i in range(len(z)-1):
        if np.sign(z_punt[i]) != np.sign(z_punt[i+1]):
            z_eq = (z[i] + z[i+1]) / 2

            if z_punt[i] > z_punt[i+1]:
                color_punt = 'blue'  
            else:
                color_punt = 'red' 

            plt.plot(z_eq, 0, 'o', color=color_punt, markersize=7, zorder=5)

plt.axhline(0, color='black', linestyle='--', lw=0.5)
plt.axvline(0, color='black', linestyle='--', lw=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('$z$', fontsize=15)
plt.ylabel(r'$\dot{z}$', fontsize=15)
plt.legend(
    fontsize=15, 
    loc='lower left', 
    bbox_to_anchor=(0.02, 0),
    labelspacing=0.2,  
    handletextpad=0.5, 
    borderaxespad=0.5   
)

plt.grid(False) 
plt.xlim(z_min - 0.02, z_max + 0.02)
plt.tight_layout()
plt.show()
