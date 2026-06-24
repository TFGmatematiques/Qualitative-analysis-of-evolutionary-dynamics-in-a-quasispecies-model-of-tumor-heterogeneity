import numpy as np
import matplotlib.pyplot as plt

z_min = 0.0
z_max = 1.001
punts = 2000

z = np.linspace(z_min, z_max, punts)

def funcio(z, f1, f2, mu12, mu21):
    delta_f = f1 - f2
    return (delta_f/3)*(z**3)+(mu12*f1+mu21*f2-delta_f)*(z**2)/2-mu21*f2*z

casos = [
    {'f1': 0.3, 'f2': 0.15, 'mu12': 0.1, 'mu21': 0.01, 'color': 'darkorange',  'label': 'A'},
    {'f1': 0.3, 'f2': 0.3, 'mu12': 0.05, 'mu21': 0.05, 'color': 'crimson',  'label': 'B'},
    {'f1': 0.3, 'f2': 0.3, 'mu12': 0.08, 'mu21': 0.02, 'color': 'forestgreen',  'label': 'C'},
    {'f1': 0.45, 'f2': 0.5, 'mu12': 0.02, 'mu21': 0.01, 'color': 'royalblue',  'label': 'D'},
    {'f1': 0.4, 'f2': 0.5, 'mu12': 0, 'mu21': 0.2, 'color': 'navy',  'label': 'E'}
]

plt.figure(figsize=(8, 5))

for c in casos:
    U = funcio(z, c['f1'], c['f2'], c['mu12'], c['mu21'])
    plt.plot(z, U, color=c['color'], label=c['label'], linewidth=2)

    for i in range(1, len(z) - 1):
        pendent_abans = U[i] - U[i-1]
        pendent_despres = U[i+1] - U[i]

        if np.sign(pendent_abans) != np.sign(pendent_despres) and np.sign(pendent_abans) != 0:
            z_eq = z[i]
            U_eq = U[i]
            if pendent_abans < 0:
                plt.plot(z_eq, U_eq, 'o', color='blue', markersize=7, zorder=5)
            else:
                plt.plot(z_eq, U_eq, 'o', color='red', markersize=7, zorder=5)

        concavitat_abans = (U[i] - U[i-1]) - (U[i-1] - U[i-2]) if i > 1 else 0
        concavitat_despres = (U[i+1] - U[i]) - (U[i] - U[i-1])
        
        if c['label'] == 'E':
            if np.sign(concavitat_abans) != np.sign(concavitat_despres) and i > 1:
                if np.sign(pendent_abans) == np.sign(pendent_despres):
                    plt.plot(z[i], U[i], 'o', color='green', markersize=7, zorder=5)

plt.axhline(0, color='black', linestyle='--', lw=0.5)
plt.axvline(0, color='black', linestyle='--', lw=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$U(z)$', fontsize=15)
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
