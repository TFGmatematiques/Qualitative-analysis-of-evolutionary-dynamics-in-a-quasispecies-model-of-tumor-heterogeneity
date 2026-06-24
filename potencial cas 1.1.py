import numpy as np
import matplotlib.pyplot as plt

z_min = 0.0
z_max = 1.0
punts = 500

z = np.linspace(z_min, z_max, punts)

def funcio(z, f1, f2, mu12, mu21):
    delta_f = f1 - f2
    return (delta_f/3)*(z**3)-(delta_f-mu12*f1)*(z**2)/2

casos = [
    {'f1': 0.2, 'f2': 0.18, 'mu12': 0.02, 'mu21': 0, 'color': 'darkorange',  'label': 'A'},
]

plt.figure(figsize=(8, 5))

for c in casos:
    U = funcio(z, c['f1'], c['f2'], c['mu12'], c['mu21'])
    plt.plot(z, U, color=c['color'], label=c['label'], linewidth=2)

    for i in range(1, len(z) - 1):
        pendent_abans = U[i] - U[i-1]
        pendent_despres = U[i+1] - U[i]
        
        if np.sign(pendent_abans) != np.sign(pendent_despres):
            z_eq = z[i]
            U_eq = U[i]
            
            if pendent_abans < 0 and pendent_despres > 0:
                plt.plot(z_eq, U_eq, 'o', color='blue', markersize=7, zorder=5)
            elif pendent_abans > 0 and pendent_despres < 0:
                plt.plot(z_eq, U_eq, 'o', color='red', markersize=7, zorder=5)
                
    pendent_inicial = U[1] - U[0]
    
    if pendent_inicial > 0:
        plt.plot(z[0], U[0], 'o', color='blue', markersize=7, zorder=5)
    elif pendent_inicial < 0:
        plt.plot(z[0], U[0], 'o', color='red', markersize=7, zorder=5)

plt.axhline(0, color='black', linestyle='--', lw=0.5)
plt.axvline(0, color='black', linestyle='--', lw=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$U(z)$', fontsize=15)
plt.legend(
    fontsize=15, 
    loc='upper right', 
    bbox_to_anchor=(1, 0.95),
    labelspacing=0.2,
    handletextpad=0.5,  
    borderaxespad=0.5  
)

plt.grid(False) 
plt.xlim(z_min - 0.02, z_max + 0.02)
plt.tight_layout()
plt.show()
